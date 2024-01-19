import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetBinaryCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetBinaryCrossEntropy, self).__init__()

    def forward(self, x, target):
        x = x.sigmoid()
        loss = -(target * torch.log(x) + (1-target) * torch.log(1-x))
        return loss.mean()
