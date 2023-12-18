import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import numpy as np
import logging
import argparse
logging.getLogger('matplotlib.font_manager').disabled = True

def specplot(spec,
             pic_path = 'exp/test/melspectrograms/spec.png',
             **kwargs):
    """Plot the log mel spectrogram of audio."""
    fig = plt.figure()
    plt.imshow(spec, origin = 'lower', cmap = plt.cm.magma, aspect='auto')
    plt.colorbar()
    fig.savefig(pic_path)
    plt.close()

def specplot_from_audio(filename = None,
                        audio = None,
                        rate = None,
                        rotate=False,
                        n_ffts = 1024,
                        pic_path = 'exp/test/spectrograms/spec.png',
                        **kwargs):
    """Plot the log magnitude spectrogram of audio."""
    if filename is not None:
        audio, rate = sf.read(filename)
    hop_length = kwargs.get('hop_length', None)
    win_length = kwargs.get('win_length', None)
    stft = librosa.stft(
            audio,
            n_fft = n_ffts,
            hop_length = hop_length,
            win_length = win_length
            )
    mag, phase = librosa.magphase(stft)
    logmag = np.log10(mag)
    fig = plt.figure()
    plt.imshow(logmag, cmap = plt.cm.magma, origin = 'lower', aspect = 'auto')
    plt.colorbar()
    fig.savefig(pic_path)
    plt.close()

def melspecplot(mel_spec,
                pic_path = 'exp/test/melspectrograms/melspec.png',
                **kwargs):
    """Plot the log mel spectrogram of audio."""
    fig = plt.figure()
    plt.imshow(mel_spec, origin = 'lower', cmap = plt.cm.magma, aspect='auto')
    plt.colorbar()
    fig.savefig(pic_path)
    plt.close()

def melspecplot_from_audio(filename = None,
                           audio = None,
                           rate = None,
                           rotate = False,
                           n_ffts = 1024,
                           pic_path = 'exp/test/melspectrograms/melspec.png',
                           **kwargs):
    """Plot the log mel spectrogram of audio."""
    if filename is not None:
        audio, rate = sf.read(filename)
    hop_length = kwargs.get('hop_length', None)
    win_length = kwargs.get('win_length', None)
    n_mels = kwargs.get('n_mels', 23)
    mel_spec = librosa.feature.melspectrogram(
                y = audio,
                sr = rate,
                n_fft = n_ffts,
                hop_length = hop_length,
                win_length = win_length,
                n_mels = n_mels
            )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    plt.imshow(mel_spec, cmap = plt.cm.magma, origin = 'lower', aspect = 'auto')
    plt.savefig(pic_path)
    plt.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--mean', type=str, default=None)
    parser.add_argument('--std', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    import numpy as np
    # specplot_from_audio(args.filename, pic_path = args.output + '.aspec.png')
    data = np.load(args.filename).T
    mean = np.load(args.mean).T # (n_fft + 1, T)
    std = np.load(args.std).T
    if 'spec' in args.filename:
        specplot(data, mean, std, args.output + '.spec.png')
    if 'mel' in args.filename:
        melspecplot(data, args.output + '.melspec.png')