import torch
from torch import nn
import torch.nn.functional as F


def binary_shadow(shadowless, true_shadowed_image, threshold):
    true_shadow = shadowless - true_shadowed_image
    binary_true_shadow = (true_shadow > 0.1).long()
    return binary_true_shadow[:,2,:,:]

def shadow_loss(binary_true_shadow, binary_estimated_shadow):
    return torch.nn.CrossEntropyLoss()(binary_estimated_shadow, binary_true_shadow)

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def binary_shadow_to_image(shadowless, binary_estimated_shadow):
    binary_estimated_shadow = F.softmax(binary_estimated_shadow, dim=1)
    binary_estimated_shadow = (binary_estimated_shadow > 0.5).float()
    shadow = binary_estimated_shadow[:,1,:,:].unsqueeze(1)
    shadow = shadow.expand(-1, 3, -1, -1)*0.45
    return shadowless*(1 - shadow)
