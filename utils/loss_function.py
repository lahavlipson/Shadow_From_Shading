import torch
from torch import nn
import torch.nn.functional as F


def shadow_loss(shadowless, binary_estimated_shadow, true_shadowed_image):
    true_shadow = shadowless - true_shadowed_image
    binary_true_shadow = (true_shadow > 0.0005).long()
    binary_true_shadow = binary_true_shadow[:,2,:,:]
    return torch.nn.CrossEntropyLoss()(binary_estimated_shadow, binary_true_shadow)

def binary_shadow_to_image(shadowless, binary_estimated_shadow):
    binary_estimated_shadow = F.softmax(binary_estimated_shadow, dim=1)
    binary_estimated_shadow = (binary_estimated_shadow > 0.1).float()
    shadow = binary_estimated_shadow[:,1,:,:].unsqueeze(1)
    shadow = shadow.expand(-1, 3, -1, -1)*0.45
    return shadowless*(1 - shadow)
                        
