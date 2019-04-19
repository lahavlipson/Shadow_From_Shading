import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, dim=128):
        return input.view(input.size(0), 1, dim, dim)

class ShadowVAE(nn.Module):
    def __init__(self):
        super(ShadowVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 5, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 9, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 12, 9, padding=4),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 1, 9, padding=4),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(128**2, 200)
        self.fc2 = nn.Linear(128**2, 200)
        self.fc3 = nn.Linear(200, 128**2)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.Conv2d(1, 12, 5, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 9, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 12, 9, padding=4),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 2, 7, padding=3)
        )

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(-30, 30)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        eps = eps.cuda()
        return mu + eps * std

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar
