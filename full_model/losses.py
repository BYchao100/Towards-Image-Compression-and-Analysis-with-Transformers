"""
Implements the loss functions
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

import math

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class DenormalizedMSELoss(nn.Module):

    def __init__(self, scale=255.):
        super().__init__()
        self.imagenet_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1,3,1,1)
        self.scale = scale

    def forward(self, x, y):
        diff = (x - y) * self.imagenet_std.to(x.device) * self.scale
        mse_loss = torch.mean(diff ** 2)

        return mse_loss


class JointLoss(nn.Module):

    def __init__(self, base_criterion: torch.nn.Module, alpha: float, beta: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.alpha = alpha
        self.d_mse = DenormalizedMSELoss()
        self.beta = beta

    def forward(self, inputs, outputs, labels):
        B, _, H, W = inputs.size()
        num_pixels = B * H * W

        outputs_cls, outputs_rec, outputs_y_likelihoods, outputs_z_likelihoods = outputs

        cls_loss = self.base_criterion(outputs_cls, labels)
        mse_loss = self.d_mse(outputs_rec, inputs)
        bpp_loss = (torch.log(outputs_y_likelihoods).sum() + torch.log(outputs_z_likelihoods).sum()) / (-math.log(2) * num_pixels)

        return self.alpha * cls_loss + self.beta * mse_loss + bpp_loss

