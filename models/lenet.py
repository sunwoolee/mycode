import torch.nn as nn
import torch.nn.functional as F
from .building_blocks import Conv2d_FA, Linear_FA, Linear_IFA, Feedback_Reciever

class LeNet_FA(nn.Module):
    def __init__(self):
        super(LeNet_FA, self).__init__()
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

class LeNet_DFA(nn.Module):
    def __init__(self):
        super(LeNet_DFA, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv1_fb = Feedback_Reciever(10)
        self.conv2 = nn.Conv2d(6,16,5)
        self.conv2_fb = Feedback_Reciever(10)
        self.fc1 = nn.Linear_FA(16*5*5, 120)
        self.fc1_fb = Feedback_Reciever(10)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_fb = Feedback_Reciever(10)
        self.fc3 = Linear_IFA(84, 10)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out, dm1 = self.conv1_fb(out)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out, dm2 = self.conv2_fb(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out, dm3 = self.fc1_fb(out)
        out = F.relu(self.fc2(out))
        out, dm4 = self.fc2_fb(out)
        out = self.fc3(out, dm1, dm2, dm3, dm4)
        return out