from torch import nn
import torch
from torch.nn import functional as F

class ShadowNet(nn.Module):

    def __init__(self):
        super(ShadowNet, self).__init__()

    def forward(self, x):
        return x