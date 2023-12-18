import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import sys
sys.path.insert(0, '/home/zengchang/code/acoustic_v2')

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from pyutils import pad_list, str_to_int_list, remove_outlier
from dataset.texts import text_to_sequence

def f02pitch(f0):
    #f0 =f0 + 0.01
    return np.log2(f0 / 27.5) * 12 + 21

def pitch2f0(pitch):
    f0 =  np.exp2((pitch - 21 ) / 12) * 27.5
    for i in range(len(f0)):
        if f0[i] <= 10:
            f0[i] = 0
    return f0

def pitchxuv(pitch, uv, to_f0 = False):
    result = pitch * uv
    if to_f0:
        result = pitch2f0(result)
    return result

def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode="constant")

def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant")

class TTSDataset(Dataset):
    def __init__(self, config):
        audio_manifest = config['audio_manifest']
        raw_text_manifest = config['raw_text_manifest']
        duration_manifest = config['duration_manifest']
        spk_manifest = config['spk_manifest']
        self.sampling_rate = config['sampling_rate']
        self.utt2path = {}
        self.utt2text = {}
        self.utt2duration = {}
        self.utt2raw_text = {}
        self.utt2spk = {}
        with open(audio_manifest, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                self.utt2path[line[0]] = line[1]
        with open(duration_manifest, 'r') as f:
            for line in f:
                line = line.rstrip().split('|')
                self.utt2text[line[0]] = ' '.join(line[2].split(' ')[0::2])
                self.utt2duration[line[0]] = ' '.join(line[2].split(' ')[1::2])
        with open(raw_text_manifest, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                self.utt2raw_text[line[0]] = line[1]
        with open(spk_manifest, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                self.utt2spk[line[0]] = line[1]
        self.utt = list(self.utt2path.keys())
        self.spk2int = {spk: idx for idx, spk in enumerate(set(self.utt2spk.values()))}
        self.int2spk = {idx: spk for spk, idx in self.spk2int.items()}
        
        self.use_phonemes = config['use_phonemes']
        self.tts_cleaner_names = config['tts_cleaner_names']
        self.eos = config['eos']
        
    def get_spk_number(self):
        return len(self.spk2int)
    
    def _norm_mean_std(self, x, mean, std, is_remove_outlier=False):
        if is_remove_outlier:
            x = remove_outlier(x)
        zero_idxs = np.where(x == 0.0)[0]
        x = (x - mean) / std
        x[zero_idxs] = 0.0
        return x
    
    def __len__(self):
        return len(self.utt)
    
    def __getitem__(self, idx):
        #  set_trace()
        uttid = self.utt[idx]
        mel_path = self.utt2path[uttid].replace('.wav', '.mel.npy')
        f0_path = self.utt2path[uttid].replace('.wav', '.f0.npy')
        energy_path = self.utt2path[uttid].replace('.wav', '.en.npy')
        
        mel = np.load(mel_path) #.transpose(1, 0)
        f0 = np.load(f0_path)
        # pitch = f02pitch(f0)
        energy = np.load(energy_path)
        raw_text = self.utt2raw_text[uttid]
        phone_text = self.utt2text[uttid]
        phone_seq = np.array(text_to_sequence(phone_text, self.tts_cleaner_names))
        duration = np.array(str_to_int_list(self.utt2duration[uttid]))
        spk = self.spk2int[self.utt2spk[uttid]]
        
        mel_len = mel.shape[0]
        duration = duration[: len(phone_seq)]
        duration[-1] = duration[-1] + (mel.shape[0] - sum(duration))
        assert mel_len == sum(duration), f'{mel_len} != {sum(duration)}'
        
        return {
            'uttid': uttid,
            'raw_text': raw_text,
            'text': phone_seq,
            'mel': mel,
            'duration': duration,
            'f0': f0,
            'energy': energy,
            'spk': spk
        }
    
class TTSCollate():
    def __init__(self):
        pass
    
    def __call__(self, batch):
        ilens = torch.from_numpy(np.array([x['text'].shape[0] for x in batch])).long()
        olens = torch.from_numpy(np.array([y['mel'].shape[0] for y in batch])).long()
        ids = [x['uttid'] for x in batch]
        raw_texts = [x['raw_text'] for x in batch]

        # perform padding and conversion to tensor
        inputs = pad_list([torch.from_numpy(x['text']).long() for x in batch], 0)
        mels = pad_list([torch.from_numpy(y['mel']).float() for y in batch], 0)

        durations = pad_list([torch.from_numpy(x['duration']).long() for x in batch], 0)
        energys = pad_list([torch.from_numpy(y['energy']).float() for y in batch], 0).squeeze(-1)
        f0 = pad_list([torch.from_numpy(y['f0']).float() for y in batch], 0).squeeze(-1)
        # pitch = pad_list([torch.from_numpy(y['pitch']).float() for y in batch], 0).squeeze(-1)
        
        spks = torch.tensor([x['spk'] for x in batch], dtype = torch.int64)

        return {
            'uttids': ids,
            'spks': spks,
            'raw_texts': raw_texts,
            'texts': inputs,
            'src_lens': ilens,
            'max_src_len': ilens.max(), 
            'mels': mels,  
            'mel_lens': olens, 
            'max_mel_len': olens.max(),
            'p_targets': f0, 
            'e_targets': energys, 
            'd_targets': durations
        }
    
if __name__ == '__main__':
    import yaml
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    with open('./configs/data.yaml', 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    dataset = TTSDataset(config)
    print(dataset[0]['text'])
    print(dataset[0]['duration'])
    collate_fn = TTSCollate()
    dataloader = DataLoader(dataset, batch_size = 64, shuffle = True, collate_fn = collate_fn, num_workers = 8)
    for data in tqdm(dataloader):
        assert data['texts'].shape[1] == data['d_targets'].shape[1], "{} != {}".format(data['texts'].shape[1], data['d_targets'].shape[1])
        pass
        # print(data['texts'].shape)
        # print(data['texts'])
        # print(data['input_len'])
        # print(data['mels'].shape)
        # print(data['labels'].shape)
        # print(data['output_len'])
        # print(data['uttids'])
        # print(data['durations'].shape)
        # print(data['durations'].sum(dim = 1))
        # print(data['energys'].shape)
        # print(data['f0s'].shape)
        # print(data['raw_texts'])
        # break
    # print(dataset[0]['mel'].shape)
