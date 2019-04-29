from torch import nn
import torch
from torch.nn import functional as F

class ShapeNet(nn.Module):

    def __init__(self):
        super(ShapeNet, self).__init__()

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(400, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 200)
        self.bn3 = nn.BatchNorm1d(200)
        self.fc4 = nn.Linear(200, 200)
        self.bn4 = nn.BatchNorm1d(200)
        self.fc5 = nn.Linear(200, 200)
        self.bn5 = nn.BatchNorm1d(200)
        self.fc_type = nn.Linear(200, 4)
        self.fc_loc = nn.Linear(200, 3)
        self.fc_scale = nn.Linear(200, 1)
        self.fc_rot = nn.Linear(200, 3)



    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.bn5(self.fc5(x)))
        shape_type = self.fc_type(x)
        shape_loc = self.fc_loc(x)
        shape_scale = self.fc_scale(x)
        shape_rot = self.fc_rot(x)
        return shape_type, shape_loc, shape_scale, shape_rot
