from torch import nn
import torch
from torch.nn import functional as F
from torchvision import models

class ShadowNet(nn.Module):

    def __init__(self):
        super(ShadowNet, self).__init__()

        self.relu = nn.ReLU()

        self.backbone = models.vgg16()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.converter = nn.Linear(1000, 900)
        self.upsample_1 = nn.ConvTranspose2d(1, 3, 4, stride=2)
        self.upsample_2 = nn.ConvTranspose2d(3, 3, 2, stride=2)

        self.conv1 = nn.Conv2d(3, 12, 5, padding=4)
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(12, 16, 7, padding=3)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 24, 9, padding=4)
        self.bn3 = nn.BatchNorm2d(24)

        self.conv4 = nn.Conv2d(24, 24, 9, padding=4)
        self.bn4 = nn.BatchNorm2d(24)

        self.conv5 = nn.Conv2d(24, 24, 9, padding=4)
        self.bn5 = nn.BatchNorm2d(24)

        self.conv6 = nn.Conv2d(24, 24, 9, padding=4)
        self.bn6 = nn.BatchNorm2d(24)

        self.conv7 = nn.Conv2d(24, 16, 9, padding=4)
        self.bn7 = nn.BatchNorm2d(16)

        self.conv8 = nn.Conv2d(16, 12, 7, padding=3)
        self.bn8 = nn.BatchNorm2d(12)

        self.conv9 = nn.Conv2d(12, 2, 5, padding=2)


    def forward(self, x):
        x = self.backbone(x)
        x = self.converter(x)
        x = x.view(x.shape[0], 1, 30, 30)
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.conv9(x)
        return x
