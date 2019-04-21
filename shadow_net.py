from torch import nn
import torch
from torch.nn import functional as F

class ShadowNet(nn.Module):

    def __init__(self):
        super(ShadowNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 12, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(12, 16, 7, padding=3)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 24, 9, padding=4)
        self.bn3 = nn.BatchNorm2d(24)

        self.conv4 = nn.Conv2d(24, 6, 9, padding=4)
        self.bn4 = nn.BatchNorm2d(6)

        self.fc1 = nn.Linear(6*128**2, 400)
        self.fc2 = nn.Linear(6*128**2, 400)
        self.fc3 = nn.Linear(400, 6*128**2)

        self.conv5 = nn.Conv2d(6, 24, 9, padding=4)
        self.bn5 = nn.BatchNorm2d(24)

        self.conv6 = nn.Conv2d(24, 24, 9, padding=4)
        self.bn6 = nn.BatchNorm2d(24)

        self.conv7 = nn.Conv2d(24, 16, 9, padding=4)
        self.bn7 = nn.BatchNorm2d(16)

        self.conv8 = nn.Conv2d(16, 12, 7, padding=3)
        self.bn8 = nn.BatchNorm2d(12)

        self.conv9 = nn.Conv2d(12, 2, 5, padding=2)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x):
        #Encode
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 6*128*128)
        mu = self.fc1(x)
        logvar = self.fc2(x)

        #Sample
        x = self.reparameterize(mu, logvar)

        #Decode
        x = self.fc3(x)
        x = x.view(-1, 6, 128, 128)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.conv9(x)
        return x, mu, logvar
