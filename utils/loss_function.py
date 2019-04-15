import torch
from torch import nn
import torch.nn.functional as F


def binary_shadow(shadowless, true_shadowed_image, threshold):
    true_shadow = shadowless - true_shadowed_image
    binary_true_shadow = (true_shadow > 0.1).long()
    return binary_true_shadow[:,2,:,:]

def shadow_loss(binary_true_shadow, binary_estimated_shadow):
    return torch.nn.CrossEntropyLoss()(binary_estimated_shadow, binary_true_shadow)

def kl_divergence(binary_estimated_shadow, binary_true_shadow):
    binary_true_shadow = torch.unsqueeze(binary_true_shadow, 1)
    inverse_shadow = 1 - binary_true_shadow
    combined = torch.cat((inverse_shadow, binary_true_shadow), dim=1)
    softmax_estimate = F.softmax(binary_estimated_shadow, dim=1)
    return torch.nn.KLDivLoss()(softmax_estimate.log(), combined)

def binary_shadow_to_image(shadowless, binary_estimated_shadow):
    binary_estimated_shadow = F.softmax(binary_estimated_shadow, dim=1)
    binary_estimated_shadow = (binary_estimated_shadow > 0.5).float()
    shadow = binary_estimated_shadow[:,1,:,:].unsqueeze(1)
    shadow = shadow.expand(-1, 3, -1, -1)*0.45
    return shadowless*(1 - shadow)
