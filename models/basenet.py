import torch.nn as nn
import torch.nn.functional as F
from .building_blocks import Conv2d_FA, Conv2d_IFA, Linear_FA, Linear_IFA, Feedback_Reciever

class BaseNet_FA(nn.Module):
    def __init__(self):
        super(BaseNet_FA, self).__init__()
        self.conv1 = Conv2d_FA(3,96,5, padding=2)
        self.conv2 = Conv2d_FA(96,128,5, padding=2)
        self.conv3 = Conv2d_FA(128,256,5, padding=2)
        self.fc1 = Linear_FA(256*3*3, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)
    
    def forward(self, x):
        out = F.tanh(self.conv1(x))
        out = F.max_pool2d(out, 3, 2)
        out = F.tanh(self.conv2(out))
        out = F.max_pool2d(out, 3, 2)
        out = F.tanh(self.conv3(out))
        out = F.max_pool2d(out, 3, 2)
        out = out.view(out.size(0), -1)
        out = F.tanh(self.fc1(out))
        out = F.tanh(self.fc2(out))
        out = self.fc3(out)
        return out

class BaseNet_DFA(nn.Module):
    def __init__(self):
        super(BaseNet_DFA, self).__init__()
        self.conv1 = nn.Conv2d(3,96,5, padding=2)
        self.conv1_fb = Feedback_Reciever(10)
        self.conv2 = nn.Conv2d(96,128,5, padding=2)
        self.conv2_fb = Feedback_Reciever(10)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2)
        self.conv3_fb = Feedback_Reciever(10)
        self.fc1 = nn.Linear(256*3*3, 2048)
        self.fc1_fb = Feedback_Reciever(10)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc2_fb = Feedback_Reciever(10)
        self.fc3 = Linear_IFA(2048, 10)
    
    def forward(self, x):
        out = F.tanh(self.conv1(x))
        out = F.max_pool2d(out, 3, 2)
        out, dm1 = self.conv1_fb(out)
        out = F.tanh(self.conv2(out))
        out = F.max_pool2d(out, 3, 2)
        out, dm2 = self.conv2_fb(out)
        out = F.tanh(self.conv3(out))
        out = F.max_pool2d(out, 3, 2)
        out, dm3 = self.conv3_fb(out)
        out = out.view(out.size(0), -1)
        out = F.tanh(self.fc1(out))
        out, dm4 = self.fc1_fb(out)
        out = F.tanh(self.fc2(out))
        out, dm5 = self.fc2_fb(out)
        out = self.fc3(out, dm1, dm2, dm3, dm4, dm5)
        return out


class BaseNet_IFA(nn.Module):
    def __init__(self):
        super(BaseNet_IFA, self).__init__()
        self.conv1 = nn.Conv2d(3,96,5, padding=2)
        self.conv1_fb = Feedback_Reciever(256*7*7)
        self.conv2 = nn.Conv2d(96,128,5, padding=2)
        self.conv2_fb = Feedback_Reciever(256*7*7)
        self.conv3 = Conv2d_IFA(128, 256, 5, padding=2)
        #self.conv3_fb = Feedback_Reciever(2048)
        self.fc1 = nn.Linear(256*3*3, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)
    
    def forward(self, x):
        out = F.tanh(self.conv1(x))
        out = F.max_pool2d(out, 3, 2)
        out, dm1 = self.conv1_fb(out)
        out = F.tanh(self.conv2(out))
        out = F.max_pool2d(out, 3, 2)
        out, dm2 = self.conv2_fb(out)
        out = F.tanh(self.conv3(out, dm1, dm2))
        out = F.max_pool2d(out, 3, 2)
        #out, dm3 = self.conv3_fb(out)
        out = out.view(out.size(0), -1)
        out = F.tanh(self.fc1(out))
        out = F.tanh(self.fc2(out))
        out = self.fc3(out)
        return out
