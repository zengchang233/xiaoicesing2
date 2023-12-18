import os
import yaml
import argparse
import math
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from dataset import SVSDataset as Dataset
from dataset import SVSCollate as Collate
from models import Xiaoice2 as Generator
from models import Discriminator as Discriminator
from loss import FastSpeech2Loss, FeatLoss, LSGANGLoss, LSGANDLoss
import pyutils
from pyutils import (
    load_checkpoint,
    save_checkpoint,
    clean_checkpoints,
    latest_checkpoint_path,
    melspecplot,
    get_logger
)

import wandb
import logging

logging.basicConfig(format = "%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s", level = logging.INFO)

python_script = os.path.realpath(__file__)

class Trainer():
    def __init__(self, rank, args, data_configs, model_configs, train_configs):
        self.rank = rank
        self.device = torch.device('cuda:{:d}'.format(rank))
        trainset = Dataset(data_configs)
        collate_fn = Collate()
        sampler = torch.utils.data.DistributedSampler(trainset) if args.num_gpus > 1 else None
        self.trainloader = DataLoader(
            trainset,
            shuffle = False,
            sampler = sampler,
            collate_fn = collate_fn,
            batch_size = train_configs['batch_size'],
            pin_memory = True,
            num_workers = train_configs['num_workers'],
            prefetch_factor = 10
        )

        model_configs['generator']['transformer']['encoder']['n_src_vocab'] = trainset.get_phone_number() + 1
        model_configs['generator']['spk_num'] = trainset.get_spk_number()
        
        if args.num_gpus > 1:
            self.models = (
                nn.parallel.DistributedDataParallel(
                    Generator(
                        data_configs,
                        model_configs['generator']
                    ).to(self.device),
                    device_ids = [rank]
                ),
                nn.parallel.DistributedDataParallel(
                    Discriminator().to(self.device),
                    device_ids = [rank]
                )
            )
        else:
            self.models = (
                Generator(
                    data_configs,
                    model_configs['generator']
                ).to(self.device),
                Discriminator().to(self.device)
            )
        self.data_configs = data_configs
        self.model_configs = model_configs
        self.train_configs = train_configs
        self.args = args

        self.g_optimizer = getattr(
            torch.optim, train_configs['g_optimizer']
        )(self.models[0].parameters(), **train_configs['g_optimizer_args'])
        
        self.g_scheduler = getattr(
            pyutils.scheduler, train_configs['g_scheduler']
        )(self.g_optimizer, **train_configs['g_scheduler_args'])
        
        self.d_optimizer = getattr(
            torch.optim, train_configs['d_optimizer']
        )(self.models[1].parameters(), **train_configs['d_optimizer_args'])
        
        self.d_scheduler = getattr(
            pyutils.scheduler, train_configs['d_scheduler']
        )(self.d_optimizer, **train_configs['d_scheduler_args'])

        self.fs2loss = FastSpeech2Loss(data_configs)
        self.feat_loss = FeatLoss(train_configs['feat_loss_weight'])
        self.adv_g_loss = LSGANGLoss(train_configs['adv_g_loss_weight'])
        self.adv_d_loss = LSGANDLoss()

        if self.rank == 0:
            self._make_exp_dir()
            self.logger = get_logger(os.path.join(self.args.exp_name, 'logs/train.log'))

        try:
            latest_gckpt_path = latest_checkpoint_path(
                os.path.join(self.args.exp_name, 'models'),
                'G_*.pth'
            )
            latest_dckpt_path = latest_checkpoint_path(
                os.path.join(self.args.exp_name, 'models'),
                'D_*.pth'
            )
            _, _, _, _, epoch_str = load_checkpoint(
                    latest_gckpt_path,
                    self.models[0],
                    self.g_optimizer,
                    self.g_scheduler,
                    False
            )
            _, _, _, _, epoch_str = load_checkpoint(
                    latest_dckpt_path,
                    self.models[1],
                    self.d_optimizer,
                    self.d_scheduler,
                    False
            )
            self.start_epoch = max(epoch_str, 1)
            name = latest_gckpt_path
            self.total_step = int(name[name.rfind("_")+1:name.rfind(".")])+1
        except Exception:
            print("Load old checkpoint failed...")
            print("Start a new training...")
            self.start_epoch = 1
            self.total_step = 0

        self.epochs = self.train_configs['epochs']
        self.start_disc_steps = self.train_configs['start_disc_steps']
        
    def _dump_args_and_config(self, filename, config):
        with open(os.path.join(self.args.exp_name, 'conf', filename) + '.yaml', 'w') as f:
            yaml.dump(config, f)
        
    def _make_exp_dir(self):
        os.makedirs(self.args.exp_name, exist_ok=True)
        os.makedirs(os.path.join(self.args.exp_name, 'conf'), exist_ok=True)
        os.makedirs(os.path.join(self.args.exp_name, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.args.exp_name, 'audios'), exist_ok=True)
        os.makedirs(os.path.join(self.args.exp_name, 'spectrograms'), exist_ok=True)
        os.makedirs(os.path.join(self.args.exp_name, 'melspectrograms'), exist_ok=True)
        os.makedirs(os.path.join(self.args.exp_name, 'eval_results'), exist_ok=True)
        os.makedirs(os.path.join(self.args.exp_name, 'logs'), exist_ok = True)
        with open(os.path.join(self.args.exp_name, 'model_arch.txt'), 'w') as f:
            for model in self.models:
                print(model, file = f)
        self._dump_args_and_config('args', vars(self.args))
        self._dump_args_and_config('data', self.data_configs)
        self._dump_args_and_config('model', self.model_configs)
        self._dump_args_and_config('train', self.train_configs)
        basename = os.path.basename(python_script)
        shutil.copyfile(python_script, os.path.join(self.args.exp_name, basename))
    
    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1, 1):
            self.train_epoch(epoch)
    
    def train_epoch(self, epoch):
        self.total_g_loss = 0.0
        self.total_d_loss = 0.0
        for batch_idx, data in enumerate(self.trainloader):
            output, postnet_output = self.train_batch(data, epoch, batch_idx)
            self.g_scheduler.step()
            self.d_scheduler.step()
            
            if self.rank == 0 and self.total_step % self.train_configs['save_interval'] == 0:
                gckpt_path = os.path.join(self.args.exp_name, 'models', 'G_{}.pth'.format(self.total_step))
                save_checkpoint(
                        self.models[0],
                        self.g_optimizer,
                        self.g_scheduler,
                        self.g_scheduler.get_lr()[0],
                        epoch,
                        gckpt_path
                )
                dckpt_path = os.path.join(self.args.exp_name, 'models', 'D_{}.pth'.format(self.total_step))
                save_checkpoint(
                        self.models[1],
                        self.d_optimizer,
                        self.d_scheduler,
                        self.d_scheduler.get_lr()[0],
                        epoch,
                        dckpt_path
                )
                length = data['mel_lens'][0]
                real_pic_path = os.path.join(self.args.exp_name, 'melspectrograms', '{}_{}.png'.format(data['uttids'][0], self.total_step))
                before_pic_path = os.path.join(self.args.exp_name, 'melspectrograms', 'before_{}_{}.png'.format(data['uttids'][0], self.total_step))
                after_pic_path = os.path.join(self.args.exp_name, 'melspectrograms', 'after_{}_{}.png'.format(data['uttids'][0], self.total_step))
                melspecplot(data['mels'][0][:length, :].transpose(1, 0).numpy(), real_pic_path) # (n_mels, T)
                melspecplot(output[0][:length, :].transpose(1, 0).detach().cpu().numpy(), before_pic_path)
                melspecplot(postnet_output[0][:length, :].transpose(1, 0).detach().cpu().numpy(), after_pic_path)

            if self.rank == 0:
                clean_checkpoints(
                    os.path.join(self.args.exp_name, 'models'),
                    n_ckpts_to_keep = self.train_configs['ckpt_clean']
                )

    def _move_to_device(self, data):
        new_data = {}
        for k, v in data.items():
            if type(v) is torch.Tensor:
                new_data[k] = v.to(self.device)
        return new_data
    
    def train_batch(self, data, epoch, step):
        for model in self.models:
            model.train()
        new_data = self._move_to_device(data)
        
        #  loss, report_keys, output, postnet_output = self.models[0](**new_data)
        self.g_optimizer.zero_grad()
        g_loss, report_keys, output, postnet_output = self.models[0](**new_data)
        if self.total_step >= self.train_configs['start_disc_steps']:
            d_fake, random_N = self.models[1](output, new_data['mel_lens'], new_data['mels'])
            feat_loss, feat_loss_report_keys = self.feat_loss(d_fake)
            adv_g_loss, adv_gloss_report_keys = self.adv_g_loss(d_fake)
            g_loss += feat_loss
            g_loss += adv_g_loss
            report_keys.update(feat_loss_report_keys)
            report_keys.update(adv_gloss_report_keys)
        g_loss.backward()
        
        grad_norm = nn.utils.clip_grad_norm_(self.models[0].parameters(), self.train_configs['grad_clip'])
        if math.isnan(grad_norm):
            raise ZeroDivisionError('Grad norm is nan')
        self.g_optimizer.step()
        self.total_g_loss += g_loss.item()
        
        if self.total_step >= self.train_configs['start_disc_steps']:
            self.d_optimizer.zero_grad()
            d_fake, _ = self.models[1](
                            output.detach(),
                            new_data['mel_lens'],
                            new_data['mels'],
                            random_N
                        )
            adv_d_loss, adv_dloss_report_keys = self.adv_d_loss(d_fake)
            adv_d_loss.backward()
            report_keys.update(adv_dloss_report_keys) 
            grad_norm = nn.utils.clip_grad_norm_(self.models[1].parameters(), self.train_configs['grad_clip'])
            if math.isnan(grad_norm):
                raise ZeroDivisionError('Grad norm is nan')
            self.d_optimizer.step()
            self.total_d_loss += adv_d_loss.item()
        
        self.total_step += 1
        if self.rank == 0:
            self.print_msg(epoch, step, report_keys) #, accuracy.item())
            wandb_log_dict = {
                    'train/avg_g_loss': self.total_g_loss / (step + 1),
                    'train/avg_d_loss': self.total_d_loss / (step + 1),
                    'train/g_lr': self.g_scheduler.get_lr()[0],
                    'train/d_lr': self.d_scheduler.get_lr()[0]
                }
            for k, v in report_keys.items():
                wandb_log_dict['train/' + k] = v
            if self.train_configs['wandb']:
                wandb.log(wandb_log_dict)
        return output, postnet_output   

    def print_msg(self, epoch, step, report_keys):
        if self.total_step % self.train_configs['log_interval'] == 0:
            temp = ''
            for k, v in report_keys.items():
                temp += '{}: {:.6f} '.format(k, v)
            message = ('[Epoch: {} Step: {} Total steps: {}] ' + temp).format(
                               epoch,   step + 1,     self.total_step
                            )
            self.logger.info(message)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-config', dest = 'data_config', type = str, default = './conf/data.yaml')
    parser.add_argument('--model-config', dest = 'model_config', type = str, default = './conf/model.yaml')
    parser.add_argument('--train-config', dest = 'train_config', type = str, default = './conf/train.yaml')
    parser.add_argument('--num-gpus', dest = 'num_gpus', type = int, default = 1)
    #  parser.add_argument('--exp-name', dest = 'exp_name', type = str, default = 'default')
    parser.add_argument('--dist-backend', dest = 'dist_backend', type = str, default = 'nccl')
    parser.add_argument('--dist-url', dest = 'dist_url', type = str, default = 'tcp://localhost:30302')
    return parser.parse_args()

def main(rank, args, configs):
    if args.num_gpus > 1:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
                backend = args.dist_backend,
                init_method = args.dist_url,
                world_size = args.num_gpus,
                rank = rank
        )
        
    data_configs, model_configs, train_configs = configs
    args.exp_name = train_configs['wandb_args']['group'] + '-' + \
                    train_configs['wandb_args']['job_type'] + '-' + \
                    train_configs['wandb_args']['name']
    args.exp_name = os.path.join('exp', args.exp_name)

    # wandb initialization
    if train_configs['wandb']:
        wandb_configs = vars(args)
        for config in configs:
            wandb_configs.update(config)
        wandb.init(
            **train_configs['wandb_args'],
            config = wandb_configs
        )
    
    trainer = Trainer(rank, args, data_configs, model_configs, train_configs)
    trainer.train()

    if train_configs['wandb']:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_configs = yaml.load(f, Loader = yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_configs = yaml.load(f, Loader = yaml.FullLoader)
    with open(args.train_config, 'r') as f:
        train_configs = yaml.load(f, Loader = yaml.FullLoader)
    configs = (data_configs, model_configs, train_configs)
    
    num_gpus = torch.cuda.device_count()
    if args.num_gpus > 1:
        mp.spawn(main, nprocs = num_gpus, args = (args, configs))
    else:
        main(0, args, configs)