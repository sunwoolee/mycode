# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:12:36 2019

@author: Jeongwoo Park
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .building_blocks import * 


class MyBlock(nn.Module):
    def __init__(self, in_planes, planes, num_block, stride, fb_features_size):
        super(MyBlock, self).__init__()
        strides = [stride] + [1]*(num_block - 1)
        self.num_block = num_block
        self.layers = []
        self.bns = []
        self.fbs = []
        for stride, next_stride in zip(strides[:-1], strides[1:]):
            self.layers.append(nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(planes))
            # Stride for fbs should be changed
            self.fbs.append(Conv_Feedback_Reciever(planes, planes, 3, stride=next_stride, padding=1, fb_features_size=fb_features_size))
            in_planes = planes
        self.layers.append(Conv2d_IFA(planes, planes, 3, stride=1, padding=1, bias=False))
        self.bns.append(nn.BatchNorm2d(planes))
    
    def forward(self, input):
        # Dummies : List of 
        out = input
        dummies = []
        #out = self.bns[0](self.layers[0](out))
        #out, dm1 = self.fbs[0](out)
        for i in range(self.num_block - 1):
            out = self.layers[i](out)
            out = self.bns[i](out)
            out, dm = self.fbs[i](out)
            dummies.append(dm)
        self.dummies = dummies
        out = self.layers[-1](out, *dummies)
        out = self.bns[-1](out)
        out = F.relu(out)
        return out

class Block_3(nn.Module):
    def __init__(self, in_planes, planes, stride, fb_features_size):
        super(MyBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.conv3 = Conv2d_IFA(planes, planes, 3, 1, padding=1, bias=False)
        self.fb1 = Conv_Feedback_Reciever(planes, planes, 3, 1, padding=1, fb_features_size=fb_features_size)
        self.fb2 = Conv_Feedback_Reciever(planes, planes, 3, 1, padding=1, fb_features_size=fb_features_size)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
    
    def forward(self, input):
        # Dummies : List of 
        out, dm1 = self.fb1(self.bn1(self.conv1(input)))
        out, dm2 = self.fb2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out, dm1, dm2))
        return out


class MyNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(MyNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        """
        self.layer1 = MyBlock(64, 64, num_blocks[0], stride=1, fb_features_size=(64,32,32))
        self.l1_fb = Conv_Feedback_Reciever(64, 128, 3, stride=2, padding=1, fb_features_size=(128,16,16))
        self.layer2 = MyBlock(64, 128, num_blocks[1], stride=2, fb_features_size=(128,16,16))
        self.l2_fb = Conv_Feedback_Reciever(128, 256, 3, stride=2, padding=1, fb_features_size=(256,8,8))
        self.layer3 = MyBlock(128, 256, num_blocks[2], stride=2, fb_features_size=(256,8,8))
        self.l3_fb = Conv_Feedback_Reciever(256, 512, 3, stride=2, padding=1, fb_features_size=(512,4,4))
        self.layer4 = MyBlock(256, 512, num_blocks[3], stride=2, fb_features_size=(512,4,4))
        """
        self.layer1 = Block_3(64, 64, 1, (64,32,32))
        self.layer2 = Block_3(64,128, 2, (128,16,16))
        self.layer3 = Block_3(128,256,2, (256,8,8))
        self.layer4 = Block_3(256,512,2, (512,4,4))
        self.linear = nn.Linear(512, num_classes)
    
    def forward(self, input):
        out = F.relu(self.bn1(self.conv1(input)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    #%%



