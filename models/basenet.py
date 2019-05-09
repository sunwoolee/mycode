import torch.nn as nn
import torch.nn.functional as F
from .building_blocks import *
#Conv2d_FA, Conv2d_IFA, Linear_FA, Linear_IFA, Feedback_Reciever, Conv_Feedback_Reciever


def merge_weights(w1, w2, s1, s2, p1, p2):
    w_merge = F.conv_transpose2d(w1.permute(1,0,2,3), w2.permute(1,0,2,3), dilation=s1).permute(1,0,2,3)
    stride_merge = s1 * s2
    padding_merge = p1 + s1 * p2
    return w_merge.contiguous(), stride_merge, padding_merge


class BaseNet_FA(nn.Module):
    def __init__(self):
        super(BaseNet_FA, self).__init__()
        self.conv1 = Conv2d_FA(3,96,5, padding=2)
        self.conv2 = Conv2d_FA(96,128,5, padding=2)
        self.conv3 = Conv2d_FA(128,256,5, padding=2)
        self.fc1 = Linear_FA(256*3*3, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
    
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
    def __init__(self, use_bn=False, activation=nn.ReLU()):
        super(BaseNet_IFA, self).__init__()
        fb_features_size = (256, 4, 4)
        self.conv1 = nn.Conv2d(3,96,3, padding=1, stride=1) # 96, 32, 32
        self.bn1 = nn.BatchNorm2d(96)
        self.conv1_fb = Conv_Feedback_Reciever(96, 256, 3, 8, 1, fb_features_size)
        self.conv2 = nn.Conv2d(96,128,3, padding=1, stride=2) # 96, 16, 16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_fb = Conv_Feedback_Reciever(128, 256, 3, 4, 1, fb_features_size)
        self.conv3 = nn.Conv2d(128, 256, 3,padding=1, stride=2) # 128, 8, 8
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_fb = Conv_Feedback_Reciever(256, 256, 3, 2, 1, fb_features_size)
        self.conv4 = Conv2d_IFA(256, 256, 3, stride=2, padding=1) # 256, 4, 4
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256*4*4, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, dm1 = self.conv1_fb(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out, dm2 = self.conv2_fb(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out, dm3 = self.conv3_fb(out)
        out = F.relu(self.bn4(self.conv4(out, dm1, dm2, dm3)))
        #out, dm3 = self.conv3_fb(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
#%%
#net = BaseNet_IFA()
##loss = net(torch.randn(1,3,32,32)).sum()
#loss.backward()

class BaseNet_IFA_v2(nn.Module):
    def __init__(self, use_bn=False, activation=nn.ReLU()):
        super(BaseNet_IFA_v2, self).__init__()
        fb_features_size = (256, 4, 4)
        self.conv1 = nn.Conv2d(3,96,3, padding=1, stride=1) # 96, 32, 32
        self.bn1 = nn.BatchNorm2d(96)
        self.conv1_fb = Conv_Feedback_Reciever(96, 256, 3, 8, 1, fb_features_size)
        self.conv2 = nn.Conv2d(96,128,3, padding=1, stride=2) # 96, 16, 16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_fb = Conv_Feedback_Reciever(128, 256, 3, 4, 1, fb_features_size)
        self.conv3 = nn.Conv2d(128, 256, 3,padding=1, stride=2) # 128, 8, 8
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_fb = Conv_Feedback_Reciever(256, 256, 3, 2, 1, fb_features_size)
        self.conv4 = Conv2d_IFA(256, 256, 3, stride=2, padding=1) # 256, 4, 4
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_fb = Feedback_Reciever(10)
        
        self.fc1 = nn.Linear(256*4*4, 2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fc1_fb = Feedback_Reciever(10)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.fc2_fb = Feedback_Reciever(10)
        self.fc3 = Linear_IFA(2048, 10)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, dm1 = self.conv1_fb(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out, dm2 = self.conv2_fb(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out, dm3 = self.conv3_fb(out)
        out = F.relu(self.bn4(self.conv4(out, dm1, dm2, dm3)))
        #out, dm3 = self.conv3_fb(out)
        out = out.view(out.size(0), -1)
        out, dm4 = self.conv4_fb(out)
        out = F.relu(self.bn5(self.fc1(out)))
        out, dm5 = self.fc1_fb(out)
        out = F.relu(self.bn6(self.fc2(out)))
        out, dm6 = self.fc2_fb(out)
        out = self.fc3(out, dm4, dm5, dm6)
        return out


class BaseNet_IFA_v3(nn.Module):
    def __init__(self, use_bn=False, activation=nn.ReLU()):
        super(BaseNet_IFA_v3, self).__init__()
        fb_features_size = (256, 4, 4)
        self.conv1 = nn.Conv2d(3,96,3, padding=1, stride=1) # 96, 32, 32
        self.bn1 = nn.BatchNorm2d(96)
        self.conv1_fb = Conv_Feedback_Reciever(96, 256, 3, 8, 1, fb_features_size)
        self.conv2 = nn.Conv2d(96,128,3, padding=1, stride=2) # 96, 16, 16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_fb = Conv_Feedback_Reciever(128, 256, 3, 4, 1, fb_features_size)
        self.conv3 = nn.Conv2d(128, 256, 3,padding=1, stride=2) # 128, 8, 8
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_fb = Conv_Feedback_Reciever(256, 256, 3, 2, 1, fb_features_size)
        self.conv4 = Conv2d_IFA(256, 256, 3, stride=2, padding=1) # 256, 4, 4
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_fb = Feedback_Reciever(2048)
        
        self.fc1 = Linear_IFA(256*4*4, 2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fc1_fb = Feedback_Reciever(10)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.fc2_fb = Feedback_Reciever(10)
        self.fc3 = Linear_IFA(2048, 10)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, dm1 = self.conv1_fb(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out, dm2 = self.conv2_fb(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out, dm3 = self.conv3_fb(out)
        out = F.relu(self.bn4(self.conv4(out, dm1, dm2, dm3)))
        #out, dm3 = self.conv3_fb(out)
        out = out.view(out.size(0), -1)
        out, dm4 = self.conv4_fb(out)
        out = F.relu(self.bn5(self.fc1(out, dm4)))
        out, dm5 = self.fc1_fb(out)
        out = F.relu(self.bn6(self.fc2(out)))
        out, dm6 = self.fc2_fb(out)
        out = self.fc3(out, dm5, dm6)
        return out
#
#%%
class BaseNet_FA_v2(nn.Module):
    def __init__(self, use_bn=False, activation=nn.ReLU()):
        super(BaseNet_FA_v2, self).__init__()
        fb_features_size = (256, 4, 4)
        self.conv1 = Conv2d_FA(3,96,3, padding=1, stride=1) # 96, 32, 32
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = Conv2d_FA(96,128,3, padding=1, stride=2) # 96, 16, 16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = Conv2d_FA(128, 256, 3,padding=1, stride=2) # 128, 8, 8
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = Conv2d_FA(256, 256, 3, stride=2, padding=1) # 256, 4, 4
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = Linear_FA(256*4*4, 2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fc2 = Linear_FA(2048, 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.fc3 = Linear_IFA(2048, 10)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        #out, dm3 = self.conv3_fb(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn5(self.fc1(out)))
        out = F.relu(self.bn6(self.fc2(out)))
        out = self.fc3(out)
        return out

    def sign_sym(self):
        self.conv4.weight_fa.data = self.conv4.weight_fa.abs() * self.conv4.weight.sign().float()
        self.conv3.weight_fa.data = self.conv3.weight_fa.abs() * self.conv3.weight.sign().float()
        self.conv2.weight_fa.data = self.conv2.weight_fa.abs() * self.conv2.weight.sign().float()
        self.conv1.weight_fa.data = self.conv1.weight_fa.abs() * self.conv1.weight.sign().float()


class BaseNet_IFA_v4(nn.Module):
    def __init__(self, use_bn=False, activation=nn.ReLU()):
        super(BaseNet_IFA_v4, self).__init__()
        fb_features_size = (256, 4, 4)
        self.conv1 = nn.Conv2d(3,96,3, padding=1, stride=1) # 96, 32, 32
        self.bn1 = nn.BatchNorm2d(96)
        self.conv1_fb = Conv_Feedback_Reciever(96, 256, 3, 8, 1, fb_features_size)
        self.conv2 = nn.Conv2d(96,128,3, padding=1, stride=2) # 96, 16, 16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_fb = Conv_Feedback_Reciever(128, 256, 3, 4, 1, fb_features_size)
        self.conv3 = nn.Conv2d(128, 256, 3,padding=1, stride=2) # 128, 8, 8
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_fb = Conv_Feedback_Reciever(256, 256, 3, 2, 1, fb_features_size)
        self.conv4 = Conv2d_IFA(256, 256, 3, stride=2, padding=1) # 256, 4, 4
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_fb = Feedback_Reciever(10)
        
        self.fc1 = nn.Linear(256*4*4, 2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fc1_fb = Feedback_Reciever(10)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.fc2_fb = Feedback_Reciever(10)
        self.fc3 = Linear_IFA(2048, 10)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, dm1 = self.conv1_fb(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out, dm2 = self.conv2_fb(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out, dm3 = self.conv3_fb(out)
        out = F.relu(self.bn4(self.conv4(out, dm1, dm2, dm3)))
        #out, dm3 = self.conv3_fb(out)
        out = out.view(out.size(0), -1)
        out, dm4 = self.conv4_fb(out)
        out = F.relu(self.bn5(self.fc1(out)))
        out, dm5 = self.fc1_fb(out)
        out = F.relu(self.bn6(self.fc2(out)))
        out, dm6 = self.fc2_fb(out)
        out = self.fc3(out, dm4,dm5, dm6)
        return out

class BaseNet_IFA_v5(nn.Module):
    def __init__(self, use_bn=False, activation=nn.ReLU()):
        super(BaseNet_IFA_v5, self).__init__()
        fb_features_size = (256, 4, 4)
        self.conv1 = nn.Conv2d(3,96,3, padding=1, stride=1) # 96, 32, 32
        self.bn1 = nn.BatchNorm2d(96)
        self.conv1_fb = Conv_Feedback_Reciever(96, 256, kernel_size=15, stride=8, padding=7, fb_features_size=fb_features_size)
        self.conv2 = nn.Conv2d(96,128,3, padding=1, stride=2) # 128, 16, 16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_fb = Conv_Feedback_Reciever(128, 256, kernel_size=7, stride=4, padding=3, fb_features_size=fb_features_size)
        self.conv3 = nn.Conv2d(128, 256, 3,padding=1, stride=2) # 256, 8, 8
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_fb = Conv_Feedback_Reciever(256, 256, kernel_size=3, stride=2, padding=1, fb_features_size=fb_features_size)
        self.conv4 = Conv2d_IFA(256, 256, 3, padding=1, stride=2) # 256, 4, 4
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_fb = Feedback_Reciever(10)
        
        self.fc1 = nn.Linear(256*4*4, 2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fc1_fb = Feedback_Reciever(10)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.fc2_fb = Feedback_Reciever(10)
        self.fc3 = Linear_IFA(2048, 10)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, dm1 = self.conv1_fb(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out, dm2 = self.conv2_fb(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out, dm3 = self.conv3_fb(out)
        out = F.relu(self.bn4(self.conv4(out, dm1, dm2, dm3)))
        #out, dm3 = self.conv3_fb(out)
        out = out.view(out.size(0), -1)
        out, dm4 = self.conv4_fb(out)
        out = F.relu(self.bn5(self.fc1(out)))
        out, dm5 = self.fc1_fb(out)
        out = F.relu(self.bn6(self.fc2(out)))
        out, dm6 = self.fc2_fb(out)
        out = self.fc3(out, dm4,dm5, dm6)
        return out

    def eval_alignment(self):
        T3 = self.conv4.weight.clone()
        T2, s_t2, p_t2 = merge_weights(self.conv3.weight, T3, s1=self.conv3.stride[0], s2=self.conv4.stride, p1=self.conv3.padding[0], p2=self.conv4.padding) # 256, 128, 7, 7
        T1 , s_t1, p_t1 = merge_weights(self.conv2.weight, T2, s1=self.conv2.stride[0], s2=s_t2, p1=self.conv2.padding[0], p2=p_t2) # 256, 96, 15, 15
        deg3 = 180 * math.acos(F.cosine_similarity(T3.view(1,-1),self.conv3_fb.weight_fb.view(1,-1))) / math.pi
        deg2 = 180 * math.acos(F.cosine_similarity(T2.view(1,-1),self.conv2_fb.weight_fb.view(1,-1))) / math.pi
        deg1 = 180 * math.acos(F.cosine_similarity(T1.view(1,-1),self.conv1_fb.weight_fb.view(1,-1))) / math.pi
        return deg3, deg2, deg1

        

#
#%%

