import torch.nn as nn
import torch.nn.functional as F

from loss import FastSpeech2Loss

from modules.transformer import Encoder, Decoder, PostNet
from modules.variance.modules import VarianceAdaptor
from pyutils import get_mask_from_lengths

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, data_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(**model_config['transformer']['encoder'])
        self.variance_adaptor = VarianceAdaptor(data_config, model_config)
        self.decoder = Decoder(**model_config['transformer']['decoder'])
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder"]['d_word_vec'],
            data_config["n_mels"],
        )
        self.postnet = PostNet(data_config['n_mels'], **model_config['postnet'])

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            n_speaker = model_config['spk_num']
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder"]["d_word_vec"],
            )
        self.loss = FastSpeech2Loss(data_config, model_config)

    def forward(
        self,
        spks,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(spks).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        outputs = (output, postnet_output, p_predictions, e_predictions, log_d_predictions, d_rounded, src_masks, mel_masks, src_lens, mel_lens)
        (total_loss, mel_loss, post_mel_loss, pitch_loss, energy_loss, duration_loss) = self.loss((mels, mel_lens, max_mel_len, p_targets, e_targets, d_targets), outputs)
        report_keys = {
            'loss': total_loss,
            'mel_loss': mel_loss,
            'post_mel_loss': post_mel_loss,
            'pitch_loss': pitch_loss,
            'energy_loss': energy_loss,
            'duration_loss': duration_loss
        }
        return total_loss, report_keys, output, postnet_output

if __name__ == "__main__":
    import yaml
    with open('/home/zengchang/code/acoustic_v2/configs/data.yaml', 'r') as f:
        data_config = yaml.load(f, Loader = yaml.FullLoader)
    with open('/home/zengchang/code/acoustic_v2/configs/model.yaml', 'r') as f:
        model_config = yaml.load(f, Loader = yaml.FullLoader)
    model = FastSpeech2(data_config, model_config['generator'])
    print(model)
