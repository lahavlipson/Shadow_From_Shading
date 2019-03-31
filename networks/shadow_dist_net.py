from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch

class ShadowDistNet(nn.Module):

    def __init__(self, dimension, samples):
        super(ShadowDistNet, self).__init__()
        self.dim = dimension
        self.samps = samples

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

        self.fc_cov = nn.Linear(100**2, self.dim)
        self.fc_means = nn.Linear(100**2, self.dim)

        ###########SAMPLER

        self.fc6 = nn.Linear(self.dim, 128**2)
        self.bn6 = nn.BatchNorm1d(128**2)

        self.conv7 = nn.Conv2d(1, 12, 5, padding=2)
        self.bn7 = nn.BatchNorm2d(12)

        self.conv8 = nn.Conv2d(12, 12, 7, padding=3)
        self.bn8 = nn.BatchNorm2d(12)

        self.conv9 = nn.Conv2d(12, 1, 7, padding=3)
        self.bn9 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = x.view(-1, 100**2)
        means = self.fc_means(x)
        cov = self.fc_cov(x)
        cov = torch.diag_embed(cov)

        distrib = MultivariateNormal(loc=abs(means), covariance_matrix=abs(cov))
        x = distrib.rsample([self.samps])
        x = x.view(-1, self.dim)

        x = self.relu(self.bn6(self.fc6(x)))
        x = x.view(-1, 1, 128, 128)
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = (x*255).expand(-1,3,-1,-1)
        print(x.shape)

        assert False
        return x