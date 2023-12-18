import torch
import torch.nn as nn

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, data_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = data_config["pitch"]["feature"]
        self.energy_feature_level = data_config["energy"]["feature"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            uv_targets,
            duration_targets,
        ) = inputs
        
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            uv_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        if not uv_targets is None:
            uv_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        if not uv_targets is None:
            uv_predictions = uv_predictions.masked_select(mel_masks)
            uv_targets = uv_targets.masked_select(mel_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        total_loss = (mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss)
        uv_loss = None
        if not uv_targets is None:
            uv_loss = self.mse_loss(uv_predictions, uv_targets)
            total_loss = (
                mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + 0.1 * uv_loss
            )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            0 if uv_loss is None else uv_loss,
            duration_loss,
        )

class FeatLoss(nn.Module):
    '''
    feature loss (multi-band discriminator) 
    '''
    def __init__(self, feat_loss_weight = (1.0, 1.0, 1.0)):
        super(FeatLoss, self).__init__()
        self.loss_d = nn.MSELoss() #.to(self.device)
        self.feat_loss_weight = feat_loss_weight

    def forward(self, D_fake):
        feat_g_loss = 0.0
        feat_loss = [0.0] * len(D_fake)
        report_keys = {}
        for j in range(len(D_fake)):
            for k in range(len(D_fake[j][0])):
                for n in range(len(D_fake[j][0][k][1])):
                    if len(D_fake[j][0][k][1][n].shape) == 4:
                        t_batch = D_fake[j][0][k][1][n].shape[0]
                        t_length = D_fake[j][0][k][1][n].shape[-1]
                        D_fake[j][0][k][1][n] = D_fake[j][0][k][1][n].view(t_batch, t_length,-1)
                        D_fake[j][1][k][1][n] = D_fake[j][1][k][1][n].view(t_batch, t_length,-1)
                    feat_loss[j] += self.loss_d(D_fake[j][0][k][1][n], D_fake[j][1][k][1][n]) * 2
                feat_loss[j] /= (n + 1)
            feat_loss[j] /= (k + 1)
            feat_loss[j] *= self.feat_loss_weight[j]
            report_keys['feat_loss_' + str(j)] = feat_loss[j]
            feat_g_loss += feat_loss[j]

        return feat_g_loss, report_keys

class LSGANGLoss(nn.Module):
    def __init__(self, adv_loss_weight):
        super(LSGANGLoss, self).__init__()
        self.loss_d = nn.MSELoss() #.to(self.device)
        self.adv_loss_weight = adv_loss_weight

    def forward(self, D_fake):
        adv_g_loss = 0.0
        adv_loss = [0.0] * len(D_fake)
        report_keys = {}
        for j in range(len(D_fake)):
            for k in range(len(D_fake[j][0])):
                adv_loss[j] += self.loss_d(D_fake[j][0][k][0], D_fake[j][0][k][0].new_ones(D_fake[j][0][k][0].size()))
            adv_loss[j] /= (k + 1)
            adv_loss[j] *= self.adv_loss_weight[j]
            report_keys['adv_g_loss_' + str(j)] = adv_loss[j]
            adv_g_loss += adv_loss[j]
        return adv_g_loss, report_keys

class LSGANDLoss(nn.Module):
    def __init__(self):
        super(LSGANDLoss, self).__init__()
        self.loss_d = nn.MSELoss()

    def forward(self, D_fake):
        adv_d_loss = 0.0
        adv_loss = [0.0] * len(D_fake)
        real_loss = [0.0] * len(D_fake)
        fake_loss = [0.0] * len(D_fake)
        report_keys = {}
        for j in range(len(D_fake)):
            for k in range(len(D_fake[j][0])):
                real_loss[j] += self.loss_d(D_fake[j][1][k][0], D_fake[j][1][k][0].new_ones(D_fake[j][1][k][0].size()))
                fake_loss[j] += self.loss_d(D_fake[j][0][k][0], D_fake[j][0][k][0].new_zeros(D_fake[j][0][k][0].size()))
            real_loss[j] /= (k + 1)
            fake_loss[j] /= (k + 1)
            adv_loss[j] = 0.5 * (real_loss[j] + fake_loss[j])
            report_keys['adv_d_loss_' + str(j)] = adv_loss[j]
            adv_d_loss += adv_loss[j]
        return adv_d_loss, report_keys
