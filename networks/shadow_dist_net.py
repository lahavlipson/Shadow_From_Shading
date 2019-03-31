from torch import nn
import torch

class ShadowDistNet(nn.Module):

    def __init__(self, dimension):
        super(ShadowDistNet, self).__init__()
        self.dim = dimension

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(3, 12, 5)
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(12, 16, 7)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 24, 7)
        self.bn3 = nn.BatchNorm2d(24)

        self.conv4 = nn.Conv2d(24, 24, 7)
        self.bn4 = nn.BatchNorm2d(24)

        self.conv5 = nn.Conv2d(24, 1, 7)
        self.bn5 = nn.BatchNorm2d(1)

        self.fc_cov = nn.Linear(17 * 17, self.dim)
        self.fc_means = nn.Linear(17 * 17, self.dim)



    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.conv5(x)
        x = x.view(-1, 17**2)
        means = self.fc_means(x)
        cov = self.fc_cov(x)
        cov = torch.diag_embed(cov)
        return means, cov