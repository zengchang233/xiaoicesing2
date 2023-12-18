import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import sys
sys.path.insert(0, '/home/smg/zengchang/code/xhs/acoustic')

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pyutils import pad_list, remove_outlier
from librosa import note_to_midi

import json
import os

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

def interpolate_f0(f0):
    data = np.reshape(f0, (f0.size, 1))

    vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0

    ip_data = data

    frame_number = data.size
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i + 1
            for j in range(i + 1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number - 1:
                if last_value > 0.0:
                    step = (data[j] - data[i - 1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i - 1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i] # this may not be necessary
            last_value = data[i]

    return ip_data[:,0], vuv_vector[:,0]

class SVSDataset(Dataset):
    def __init__(self, configs):
        audio_manifest = configs['audio_manifest']
        transcription_manifest = configs['svs_manifest']
        spk_manifest = configs['spk_manifest']
        self.sampling_rate = configs['sampling_rate']
        self.utt2path = {}
        self.utt2raw_text = {}
        self.utt2phone_seq = {}
        self.utt2note_pitch = {}
        self.utt2note_dur = {}
        self.utt2dur = {}
        self.utt2spk = {}
        hop_length = configs['hop_length'] / self.sampling_rate
        with open(audio_manifest, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                self.utt2path[line[0]] = line[1]
        with open(transcription_manifest, 'r') as f:
            for line in f:
                line = line.rstrip().split('|')
                self.utt2raw_text[line[0]] = line[1]
                self.utt2phone_seq[line[0]] = line[2].split(' ')
                self.utt2note_pitch[line[0]] = [note_to_midi(note.split('/')[0]) if note != 'rest' else 0 for note in line[3].split(' ')]
                self.utt2note_dur[line[0]] = [round(eval(dur) / hop_length) for dur in line[4].split(' ')]
                self.utt2dur[line[0]] = [round(eval(dur) / hop_length) for dur in line[5].split(' ')]
        with open(spk_manifest, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                self.utt2spk[line[0]] = line[1]
        if not os.path.exists(configs['phone_set']):
            phone_set = set()
            for phone_seq in self.utt2phone_seq.values():
                phone_set.update(phone_seq)
            phone_set = list(phone_set)
            phone_set.sort()
            with open(configs['phone_set'], 'w') as f:
                json.dump(phone_set, f)
            self.phone_set = phone_set
        else:
            with open(configs['phone_set'], 'r') as f:
                self.phone_set = json.load(f)
                
        self.spk2int = {spk: idx for idx, spk in enumerate(set(self.utt2spk.values()))}
        self.int2spk = {idx: spk for spk, idx in self.spk2int.items()}
        self.phone2idx = {phone: idx for idx, phone in enumerate(self.phone_set)}
        self.utt = list(self.utt2path.keys())
    
    def _norm_mean_std(self, x, mean, std, is_remove_outlier=False):
        if is_remove_outlier:
            x = remove_outlier(x)
        zero_idxs = np.where(x == 0.0)[0]
        x = (x - mean) / std
        x[zero_idxs] = 0.0
        return x

    def get_spk_number(self):
        return len(self.spk2int)

    def get_phone_number(self):
        return len(self.phone2idx)
    
    def __len__(self):
        return len(self.utt)
    
    def __getitem__(self, idx):
        uttid = self.utt[idx]
        mel_path = self.utt2path[uttid].replace('.wav', '.mel.npy')
        f0_path = self.utt2path[uttid].replace('.wav', '.f0.npy')
        energy_path = self.utt2path[uttid].replace('.wav', '.en.npy')
        
        mel = np.load(mel_path) #.transpose(1, 0)
        f0 = np.load(f0_path)
        f0, uv = interpolate_f0(f0)
        #  unnormalized_f0 = self.f0_std * f0 + self.f0_mean
        pitch = f02pitch(f0)
        energy = np.load(energy_path)
        #  energy = self.energy_std * energy + self.energy_mean

        raw_text = self.utt2raw_text[uttid]
        phone_text = self.utt2phone_seq[uttid]
        phone_seq = np.array([self.phone2idx[phone] for phone in phone_text])
        note_pitch = np.array(self.utt2note_pitch[uttid])
        note_duration = np.array(self.utt2note_dur[uttid])
        duration = np.array(self.utt2dur[uttid])
        
        mel_len = mel.shape[0]
        duration = duration[: len(phone_seq)]
        duration[-1] = duration[-1] + (mel.shape[0] - sum(duration))
        assert mel_len == sum(duration), f'{mel_len} != {sum(duration)}'
        
        return {
            'uttid': uttid,
            'raw_text': raw_text,
            'text': phone_seq,
            'note_pitch': note_pitch,
            'note_duration': note_duration,
            'mel': mel,
            'duration': duration,
            'pitch': pitch,
            'uv': uv,
            'energy': energy
        }
    
class SVSCollate():
    def __init__(self):
        pass
    
    def __call__(self, batch):
        ilens = torch.from_numpy(np.array([x['text'].shape[0] for x in batch])).long()
        olens = torch.from_numpy(np.array([y['mel'].shape[0] for y in batch])).long()
        ids = [x['uttid'] for x in batch]
        raw_texts = [x['raw_text'] for x in batch]

        # perform padding and conversion to tensor
        inputs = pad_list([torch.from_numpy(x['text']).long() for x in batch], 0)
        note_pitchs = pad_list([torch.from_numpy(x['note_pitch']).long() for x in batch], 0)
        note_durations = pad_list([torch.from_numpy(x['note_duration']).long() for x in batch], 0)
        
        mels = pad_list([torch.from_numpy(y['mel']).float() for y in batch], 0)
        durations = pad_list([torch.from_numpy(x['duration']).long() for x in batch], 0)
        energys = pad_list([torch.from_numpy(y['energy']).float() for y in batch], 0).squeeze(-1)
        pitchs = pad_list([torch.from_numpy(y['pitch']).float() for y in batch], 0).squeeze(-1)
        uvs = pad_list([torch.from_numpy(y['uv']).float() for y in batch], 0).squeeze(-1)

        return {
            'uttids': ids,
            'raw_texts': raw_texts,
            'texts': inputs,
            'note_pitchs': note_pitchs,
            'note_durations': note_durations,
            'src_lens': ilens,
            'max_src_len': ilens.max(),
            'mels': mels,
            'mel_lens': olens,
            'max_mel_len': olens.max(),
            'p_targets': pitchs,
            'e_targets': energys,
            'uv_targets': uvs,
            'd_targets': durations
        }

if __name__ == '__main__':
    import yaml
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    with open('./configs/data.yaml', 'r') as f:
        configs = yaml.load(f, Loader = yaml.FullLoader)
    dataset = SVSDataset(configs)
    collate_fn = SVSCollate()
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True, collate_fn = collate_fn, num_workers = 8)
    for data in tqdm(dataloader):
        assert data['note_pitchs'].shape[-1] == data['note_durations'].shape[-1]
        assert data['uv_targets'].shape[1] == data['p_targets'].shape[-1]
        pass
