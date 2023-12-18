from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from pyutils import get_mask_from_lengths, pad

def f02pitch(f0):
    #f0 =f0 + 0.01
    return np.log2(f0 / 27.5) * 12 + 21

class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, data_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(**model_config['variance_predictor'])
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(**model_config['variance_predictor'])
        self.uv_predictor = VariancePredictor(**model_config['variance_predictor'])
        self.energy_predictor = VariancePredictor(**model_config['variance_predictor'])

        self.uv_threshold = model_config['uv_threshold']

        self.pitch_feature_level = data_config["pitch"]["feature"]
        self.energy_feature_level = data_config["energy"]["feature"]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        pitch_min_max = f02pitch(np.load(data_config['f0_min_max']))
        pitch_min, pitch_max = pitch_min_max[0][0], pitch_min_max[0][1]
        # print(np.load(data_config['energy_min_max']))

        energy_min_max = np.load(data_config['energy_min_max'])
        energy_min, energy_max = energy_min_max[0][0] + 1e-4, energy_min_max[0][1]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min + 1e-6), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder"]["d_word_vec"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder"]["d_word_vec"]
        )
        self.uv_embedding = nn.Embedding(
                2, model_config['transformer']['encoder']['d_word_vec']
        )

    def get_uv_embedding(self, x, target, mask, control=1.0):
        prediction = self.uv_predictor(x, mask)
        if target is not None:
            embedding = self.uv_embedding(target.to(torch.int64))
        else:
            prediction = prediction * control
            prediction = torch.sigmoid(prediction)
            for i in range(prediction.shape[0]):
                prediction[i] = prediction[i] >= self.uv_threshold # (B, max_frames, 1)

            embedding = self.uv_embedding(prediction.long())

        return prediction, embedding

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        uv_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        uv_control=1.0
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        uv_prediction, uv_embedding = self.get_uv_embedding(
            x, uv_target, mel_mask, uv_control
        )
        x = x + uv_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            uv_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        device = x.device
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(
        self, input_size, filter_size,
        kernel_size, dropout
    ):
        super(VariancePredictor, self).__init__()

        # self.input_size = model_config["transformer"]["encoder_hidden"]
        # self.filter_size = model_config["variance_predictor"]["filter_size"]
        # self.kernel = model_config["variance_predictor"]["kernel_size"]
        # self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        # self.dropout = model_config["variance_predictor"]["dropout"]
        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel = kernel_size
        self.conv_output_size = filter_size
        self.dropout = dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

if __name__ == "__main__":
    import yaml
    with open('/home/zengchang/code/acoustic_v2/configs/data.yaml', 'r') as f:
        data_config = yaml.load(f, Loader = yaml.FullLoader)
    with open('/home/zengchang/code/acoustic_v2/configs/model.yaml', 'r') as f:
        model_config = yaml.load(f, Loader = yaml.FullLoader)
    model = VarianceAdaptor(data_config, model_config['generator'])
    print(model)
