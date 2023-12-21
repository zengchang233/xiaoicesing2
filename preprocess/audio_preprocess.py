import argparse
import os
import sys
import librosa
import pyworld
import parselmouth
import soundfile as sf
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from ipdb import set_trace
from pyutils import f02pitch, pitch2f0, pitchxuv
cwd=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, cwd)

def resample_wav(wav, src_sr, tgt_sr):
    return librosa.resample(wav, orig_sr=src_sr, target_sr=tgt_sr)

def _resize_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)), source)
    res = np.nan_to_num(target)
    return res

def compute_f0_dio(wav, p_len=None, sampling_rate=48000, hop_length=240):
    if p_len is None:
        p_len = wav.shape[0]//hop_length
    f0, t = pyworld.dio(
        wav.astype(np.double),
        fs=sampling_rate,
        f0_ceil=800,
        frame_period=1000 * hop_length / sampling_rate
    )
    f0 = pyworld.stonemask(wav.astype(np.double), f0, t, sampling_rate)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return _resize_f0(f0, p_len)

def compute_f0_parselmouth(wav, p_len=None, sampling_rate=48000, hop_length=240):
    x = wav
    if p_len is None:
        p_len = x.shape[0]//hop_length
    else:
        assert abs(p_len-x.shape[0]//hop_length) < 4, "pad length error"
    time_step = hop_length / sampling_rate * 1000
    f0_min = 50
    f0_max = 1100
    f0 = parselmouth.Sound(x, sampling_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
        f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
    return f0

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

def read_scp(scp_file):
    filelists = []
    with open(scp_file, 'r') as f:
        for line in f:
            line = line.rstrip()
            filelists.append(line)
    return filelists

def spec_normalize(feat):
    '''
    params:
        feat: T, F
    '''
    return (feat - feat.mean(axis = 0, keepdims = True)) / (feat.std(axis = 0, keepdims = True) + 2e-12)

def pad_wav(wav, config):
    padded_wav = np.pad(wav, (int((config['n_fft']-config['hop_length'])/2), int((config['n_fft']-config['hop_length'])/2)), mode='reflect')
    return padded_wav

def extract_spec_with_energy(wav, filepath, config, spec_scaler = None, energy_scaler = None):
    '''
    (T, F/C)
    '''
    wav = pad_wav(wav, config)
    stft = librosa.stft(
            wav,
            n_fft = config['n_fft'],
            hop_length = config['hop_length'],
            win_length = config['win_length'],
            window = 'hann',
            center = False,
            pad_mode = 'reflect'
    )
    # set_trace()
    spec = np.abs(stft).transpose(1, 0)
    energy = np.sqrt((spec**2).sum(axis = 1))
    energy = energy.reshape(-1, 1)
    if spec_scaler is not None:
        spec_scaler.partial_fit(spec)
    if energy_scaler is not None:
        energy_scaler.partial_fit(energy)
    suffix = filepath.split('.')[-1]
    spec_filepath = filepath.replace(f'.{suffix}', '.spec.npy')
    np.save(spec_filepath, spec)
    suffix = filepath.split('.')[-1]
    energy_filepath = filepath.replace(f'.{suffix}', '.en.npy')
    np.save(energy_filepath, energy)
    # return spec, energy

def extract_mel(wav, filepath, config, mel_scaler):
    '''
    log mel + spec normalization
    (T, F/C)
    '''
    wav = pad_wav(wav, config)
    mel_spec = librosa.feature.melspectrogram(
            y = wav,
            sr = config['sampling_rate'],
            n_fft = config['n_fft'],
            hop_length = config['hop_length'],
            win_length = config['win_length'],
            window = 'hann',
            n_mels = config['n_mels'],
            fmin = config['fmin'],
            fmax = config['fmax'],
            center = False,
            pad_mode = 'reflect'
    )
    log_mel_spec = np.log(mel_spec + 1e-9).transpose(1, 0)
    #  normalized_log_mel_spec = spec_normalize(log_mel_spec)
    mel_scaler.partial_fit(log_mel_spec)
    suffix = filepath.split('.')[-1]
    mel_filepath = filepath.replace(f'.{suffix}', '.mel.npy')
    np.save(mel_filepath, log_mel_spec)

def extract_f0(filepath, config, f0_scaler = None):
    '''
    (T, 1)
    '''
    wav, sr = sf.read(filepath)
    wav = resample_wav(wav, sr, config['sampling_rate'])
    sr = config['sampling_rate']
    assert sr == config['sampling_rate'], "Sampling rate ({}) != {}, please fix it!".format(sr, config['sampling_rate'])
    #  wav = pad_wav(wav, config) # don't padding for computing f0
    f0 = compute_f0_dio(
            wav,
            sampling_rate = config["sampling_rate"],
            hop_length = config["hop_length"]
    )
    f0, uv = interpolate_f0(f0)
    f0 = f0.reshape(-1, 1)
    if f0_scaler is not None:
        f0_scaler.partial_fit(f0)
    suffix = filepath.split('.')[-1]
    f0_filepath = filepath.replace(f'.{suffix}', '.f0.npy')
    uv_filepath = filepath.replace(f'.{suffix}', '.uv.npy')
    np.save(f0_filepath, f0)
    np.save(uv_filepath, uv)

def process_one_utterance_spec(filepath, config, spec_scaler, mel_scaler, energy_scaler = None):
    wav, sr = sf.read(filepath)
    wav = resample_wav(wav, sr, config['sampling_rate'])
    sr = config['sampling_rate']
    assert sr == config['sampling_rate'], "Sampling rate ({}) != {}, please fix it!".format(sr, config['sampling_rate'])
    if args.spec:
        extract_spec_with_energy(wav, filepath, config, spec_scaler, energy_scaler)
    if args.mel:
        extract_mel(wav, filepath, config, mel_scaler)

def normalize(filelists, mean, std, feature = 'f0'):
    '''
    normalize spec/mel_spec
    unnormalize f0/energy
    '''
    min_value = np.finfo(np.float64).max
    max_value = np.finfo(np.float64).min
    for filepath in filelists:
        suffix = filepath.split('.')[-1]
        filepath = filepath.replace(f'.{suffix}', f'.{feature}.npy')
        values = np.load(filepath)
        if feature in ['f0', 'en']:
            min_value = min(min_value, min(values))
            max_value = max(max_value, max(values))
        else:
            values = (np.load(filepath) - mean) / std
            np.save(filepath, values)
    return np.array([min_value, max_value]).reshape(1, -1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", dest = "data_config", type = str, default = "", help = "data config path")
    parser.add_argument("--spec", action = "store_true", help = "extract stft spec feature")
    parser.add_argument("--mel", action = "store_true", help = "extract mel feature")
    parser.add_argument("--f0", action = "store_true", help = "extract f0 and uv")
    parser.add_argument("--energy", action = "store_true", help = "extract energy")
    parser.add_argument("--stat", action = "store_true", help = "Count the statistical numbers (mean and std) for energy and f0")

    args = parser.parse_args()
    return args

def main(args):
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader = yaml.FullLoader)
    filelists = []
    with open(data_config['audio_manifest'], 'r') as f:
        for line in f:
            line = line.rstrip().split(' ')[-1]
            filelists.append(line)
    args.scp_file = data_config['audio_manifest']

    spec_scaler = StandardScaler()
    mel_scaler = StandardScaler()
    f0_scaler = None
    energy_scaler = None
    if args.stat:
        f0_scaler = StandardScaler()
        energy_scaler = StandardScaler()

    print("Extracting features...")
    for filepath in tqdm(filelists):
        if args.spec or args.mel:
            try:
                process_one_utterance_spec(filepath, data_config, spec_scaler, mel_scaler, energy_scaler)
            except:
                print(filepath)
        if args.f0:
            try:
                extract_f0(filepath, data_config, f0_scaler)
            except:
                print(filepath)
                
    if args.stat:
        if args.spec:
            spec_mean = spec_scaler.mean_.reshape(1, -1)
            spec_std = spec_scaler.scale_.reshape(1, -1)
            np.save(os.path.join(os.path.dirname(args.scp_file), 'spec_mean.npy'), spec_mean)
            np.save(os.path.join(os.path.dirname(args.scp_file), 'spec_std.npy'), spec_std)
            normalize(filelists, spec_mean, spec_std, feature = 'spec')
            
        if args.mel:
            mel_mean = mel_scaler.mean_.reshape(1, -1)
            mel_std = mel_scaler.scale_.reshape(1, -1)
            np.save(os.path.join(os.path.dirname(args.scp_file), 'mel_mean.npy'), mel_mean)
            np.save(os.path.join(os.path.dirname(args.scp_file), 'mel_std.npy'), mel_std)
            normalize(filelists, mel_mean, mel_std, feature = 'mel')

        if args.f0:
            print("Calculating f0 stats...")
            f0_mean = f0_scaler.mean_.reshape(1, -1)
            f0_std = f0_scaler.scale_.reshape(1, -1)
            np.save(os.path.join(os.path.dirname(args.scp_file), 'f0_mean.npy'), f0_mean)
            np.save(os.path.join(os.path.dirname(args.scp_file), 'f0_std.npy'), f0_std)
            f0_min_max = normalize(filelists, f0_mean, f0_std, feature = 'f0')
            np.save(os.path.join(os.path.dirname(args.scp_file), 'f0_min_max.npy'), f0_min_max)

        if args.energy:
            print("Calculating energy stats...")
            energy_mean = energy_scaler.mean_.reshape(1, -1)
            energy_std = energy_scaler.scale_.reshape(1, -1)
            np.save(os.path.join(os.path.dirname(args.scp_file), 'energy_mean.npy'), energy_mean)
            np.save(os.path.join(os.path.dirname(args.scp_file), 'energy_std.npy'), energy_std)
            energy_min_max = normalize(filelists, energy_mean, energy_std, feature = 'en')
            np.save(os.path.join(os.path.dirname(args.scp_file), 'energy_min_max.npy'), energy_min_max)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
