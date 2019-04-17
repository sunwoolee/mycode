import torch.nn as nn
import torch.nn.functional as F
from .building_blocks import Conv2d_FA, Linear_FA

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2d_FA(3,6,5)
        self.conv2 = Conv2d_FA(6,16,5)
        self.fc1 = Linear_FA(16*5*5, 120)
        self.fc2 = Linear_FA(120, 84)
        self.fc3 = Linear_FA(84, 10)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
