from .save_and_load import *
from .plot import *
from .logger import *
from .mask import *
from .logger import *
from .optimizer import *
from . import scheduler
import torch
import numpy as np

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

def initialize(model, init_type="pytorch"):
    """Initialize Transformer module

    :param torch.nn.Module model: core instance
    :param str init_type: initialization type
    """
    if init_type == "pytorch":
        return

    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init_type)
    # bias init
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()

    # reset some loss with default init
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.LayerNorm)):
            m.reset_parameters()

def get_mask_from_lengths(lengths, max_len=None):
    device = lengths.device
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
