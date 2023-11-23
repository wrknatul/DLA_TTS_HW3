import torch.nn as nn
import torch


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_loss = nn.MSELoss()
        self.duration_loss = nn.MSELoss()
        self.energy_loss = nn.MSELoss()
        self.pitch_loss = nn.MSELoss()

    def forward(self, mel, log_duration_predicted, pitch_pred, energy_pred,
                mel_target, duration_predictor_target, pitch_target, energy_target):
        mel_loss = self.mel_loss(mel, mel_target)

        duration_predictor_loss = self.duration_loss(log_duration_predicted,
                                                     torch.log(duration_predictor_target.float() + 1))
        pitch_loss = self.pitch_loss(pitch_pred, torch.log(pitch_target + 1))
        energy_loss = self.energy_loss(energy_pred, torch.log(energy_target + 1))
        return mel_loss, duration_predictor_loss, pitch_loss, energy_loss
