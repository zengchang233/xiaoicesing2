import torch
from torch import nn
from typing import Optional

class PitchPredictorLoss(nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(PitchPredictorLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        # NOTE: We convert the output in log domain low error value
        # print("Output :", outputs[0])
        # print("Before Output :", targets[0])
        # targets = torch.log(targets.float() + self.offset)
        # print("Before Output :", targets[0])
        # outputs = torch.log(outputs.float() + self.offset)
        loss = self.criterion(outputs, targets)
        # print(loss)
        return loss


class EnergyPredictorLoss(nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(EnergyPredictorLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        # NOTE: outputs is in log domain while targets in linear
        # targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss

class DurationPredictorLoss(nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        # NOTE: outputs is in log domain while targets in linear
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss