import torch
import torch.nn as nn
import numpy as np

try:
    import modules_v2.transformer.Constants as Constants
    from modules_v2.transformer.Layers import FFTBlock
    from dataset.texts.symbols import symbols
except:
    import sys
    import os
    filepath = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
    print(filepath)
    sys.path.insert(0, filepath)
    import modules_v2.transformer.Constants as Constants
    from modules_v2.transformer.Layers import FFTBlock
    from dataset.texts.symbols import symbols


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """ Encoder """

    def __init__(
        self, max_seq_len, n_src_vocab, d_word_vec,
        n_layers, n_head, d_model, d_inner, max_note_pitch,
        max_note_duration, kernel_size, dropout=0.1
    ):
        super(Encoder, self).__init__()

        n_position = max_seq_len + 1
        n_src_vocab = n_src_vocab
        d_word_vec = d_word_vec
        n_layers = n_layers
        n_head = n_head
        d_k = d_v = (d_word_vec // n_head)
        d_model = d_model
        d_inner = d_inner
        kernel_size = kernel_size
        dropout = dropout

        # self.max_seq_len = config["max_seq_len"]
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx = Constants.PAD
        )
        self.note_pitch_emb = nn.Embedding(
            max_note_pitch, d_word_vec, padding_idx = Constants.PAD
        )
        self.note_duration_emb = nn.Embedding(
            max_note_duration, d_word_vec, padding_idx = Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, note_pitchs, note_durations, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            # print("training!!!!!!!!!!")
            enc_output = self.src_word_emb(src_seq) \
                       + self.note_pitch_emb(note_pitchs) \
                       + self.note_duration_emb(note_durations) \
                       + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(
        self, max_seq_len, d_word_vec,
        n_layers, n_head, d_model, d_inner,
        kernel_size, dropout=0.1
    ):
        super(Decoder, self).__init__()

        n_position = max_seq_len + 1
        d_word_vec = d_word_vec
        n_layers = n_layers
        n_head = n_head
        d_k = d_v = (d_word_vec // n_head)
        d_model = d_model
        d_inner = d_inner
        kernel_size = kernel_size
        dropout = dropout

        # self.max_seq_len = config["max_seq_len"]
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask

if __name__ == "__main__":
    encoder = Encoder({"max_seq_len": 100, "transformer": {"encoder_hidden": 256, "encoder_layer": 6, "encoder_head": 4, "conv_filter_size": 1024, "conv_kernel_size": [3,3], "encoder_dropout": 0.1}})
    decoder = Decoder({"max_seq_len": 100, "transformer": {"decoder_hidden": 256, "decoder_layer": 6, "decoder_head": 4, "conv_filter_size": 1024, "conv_kernel_size": [3,3], "decoder_dropout": 0.1}})
    src_seq = torch.randint(0, 100, (2, 100))
    mask = torch.ones((2, 100)).bool()
    enc_output = encoder(src_seq, mask)
    dec_output, mask = decoder(enc_output, mask)
    print(dec_output.shape, mask.shape)
