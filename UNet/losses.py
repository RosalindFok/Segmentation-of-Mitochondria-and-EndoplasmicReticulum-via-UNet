"""
Losses:
    - Binary Cross Entropy Loss (BCELoss)
    - DICE Loss (DICELoss)
    - Intersection over Union Loss (IoULoss)
"""

import torch
from torch.nn.modules.loss import _Loss

__all__ = ['BCELoss', 'DICELoss', 'IoULoss']


class BCELoss(_Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input, target):
        bceloss = torch.nn.BCELoss()
        return bceloss(input, target)

class DICELoss(_Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input, target):
        # Define a small constant eps to avoid zero division
        eps = 1e-6
        intersection = torch.sum(torch.mul(input, target)) 
        union = torch.sum(input) + torch.sum(target) + eps
        dice = 2 * intersection / union 
        dice_loss = 1 - dice
        return dice_loss

class IoULoss(_Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input, target):
        # Define a small constant eps to avoid zero division
        eps = 1e-6
        intersection = torch.sum(torch.mul(input, target))
        union = torch.sum(input) + torch.sum(target) - intersection + eps
        iou = intersection / union
        iou_loss = 1 - iou
        return iou_loss
