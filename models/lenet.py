import torch.nn as nn
import torch.nn.functional as F
from .building_blocks import * # Conv2d_FA, Linear_FA, Linear_IFA, Feedback_Reciever
import math

class LeNet_FA(nn.Module):
    def __init__(self):
        super(LeNet_FA, self).__init__()
        self.conv1 = Conv2d_FA(3,6,5)
        self.conv2 = Conv2d_FA(6,16,5)
        self.fc1 = Linear_FA(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
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
        self.fc1 = nn.Linear(16*5*5, 120)
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

class BigNet(nn.Module):
    def __init__(self):
        super(BigNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,7)
        self.conv2 = nn.Conv2d(6,16,7)
        self.conv3 = nn.Conv2d(16,32,7)
        self.conv4 = nn.Conv2d(32,64,7)
        self.fc1 = nn.Linear(64*8*8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class BigNet_DFA(nn.Module):
    def __init__(self):
        super(BigNet_DFA, self).__init__()
        self.conv1 = nn.Conv2d(3,6,7)   # 26
        self.conv2 = nn.Conv2d(6,16,7)  # 20
        self.conv3 = nn.Conv2d(16,32,7) # 14
        self.conv4 = Conv2d_IFA(32,64,7) # 8
        self.fc1 = nn.Linear(64*8*8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
        self.fb1 = Conv_Feedback_Reciever(6, 64, 19, 1, 0, (64,8,8))
        self.fb2 = Conv_Feedback_Reciever(16, 64, 13, 1, 0, (64,8,8))
        self.fb3 = Conv_Feedback_Reciever(32, 64, 7, 1, 0, (64,8,8))
    def forward(self, x):
        out, dm1 = self.fb1(F.relu(self.conv1(x)))
        out, dm2 = self.fb2(F.relu(self.conv2(out)))
        out, dm3 = self.fb3(F.relu(self.conv3(out)))
        out = F.relu(self.conv4(out, dm1, dm2, dm3))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def eval_alignment(self):
        T3 = self.conv4.weight.clone()
        T2 = F.conv2d(self.conv3.weight.flip(2,3).permute(1,0,2,3), T3, padding=6).flip(2,3).permute(1,0,2,3)
        T1 = F.conv2d(self.conv2.weight.flip(2,3).permute(1,0,2,3), T2, padding=12).flip(2,3).permute(1,0,2,3)

        deg3 = 180 * math.acos(F.cosine_similarity(T3.view(1,-1), self.fb3.weight_fb.view(1,-1))) / math.pi
        deg2 = 180 * math.acos(F.cosine_similarity(T2.view(1,-1), self.fb2.weight_fb.view(1,-1))) / math.pi
        deg1 = 180 * math.acos(F.cosine_similarity(T1.view(1,-1), self.fb1.weight_fb.view(1,-1))) / math.pi
        return deg1, deg2, deg3 




