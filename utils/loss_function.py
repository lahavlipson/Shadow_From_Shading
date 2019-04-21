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
                        

# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss(shadowless, binary_estimated_shadow, true_shadowed_image, mu, logvar):
    BCE = shadow_loss(shadowless, binary_estimated_shadow, true_shadowed_image)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD =0 * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD
