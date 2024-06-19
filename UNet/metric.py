"""
Metrics:
    - DICE coefficient
    - Intersection over Union (IoU)
    - Accuracy
    - Specificity
    - Sensitivity
    - Hausdorff distance
"""

import torch
import numpy as np
import scipy.spatial.distance as spd


__all__ = ['DICE', 'IoU', 'ACCURACY', 'SPECIFICITY', 'SENSITIVITY', 'HausdorffDistance']

# def dice_coef(output, target):
#     smooth = 1e-5
#     output = output.view(-1).data.cpu().numpy()
#     target = target.view(-1).data.cpu().numpy()
#     intersection = (output * target).sum()
#     return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

class DICE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        # Define a small constant eps to avoid zero division
        eps = 1e-6
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        intersection = torch.sum(torch.mul(output, target))
        return (2. * intersection) / (torch.sum(output) + torch.sum(target) + eps)

class IoU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        # Define a small constant eps to avoid zero division
        eps = 1e-6
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        intersection = torch.sum(torch.mul(output, target))
        union = torch.sum(output) + torch.sum(target) - intersection + eps
        return intersection / union

class ACCURACY(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        output = torch.squeeze(output).view(-1)
        target = torch.squeeze(target).view(-1)
        return 1 - torch.mean(torch.abs(output - target))

class SPECIFICITY(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        output = torch.squeeze(output).view(-1)
        target = torch.squeeze(target).view(-1)
        neg_output = 1 - output
        neg_target = 1 - target
        neg_intersection = torch.sum(torch.mul(neg_output, neg_target))
        return neg_intersection / torch.sum(neg_target)

class SENSITIVITY(torch.nn.Module): 
    def __init__(self):
        super().__init__()

    def forward(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        pos_output = output
        pos_target = target
        pos_intersection = torch.sum(torch.mul(pos_output, pos_target))
        return pos_intersection / torch.sum(pos_target)

class HausdorffDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output : torch.Tensor, target : torch.Tensor) -> float:
        output = torch.squeeze(output).cpu().numpy()
        target = torch.squeeze(target).cpu().numpy()
        output_binary = np.where(output > 0.5, 1, 0)
        set_A = np.argwhere(output_binary)
        set_B = np.argwhere(target)
        return max(spd.directed_hausdorff(set_A, set_B)[0], spd.directed_hausdorff(set_B, set_A)[0])