# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:43:15 2019

@author: jeffj
"""

import torch
import torch.nn
import torch.nn.functional as F

# a, b, k, k : c, b, k, k
# out, in, k, k
w1 = torch.randn(2,1,3,3)
w2 = torch.randn(5,2,7,7)

# Add for strided cases
s1 = 2
s2 = 4
p1 = 1
p2 = 3 
#w_comb = F.conv2d(w1.flip(2,3).permute(1,0,2,3), w2, padding=4+p2, dilation=s1).flip(2,3).permute(1,0,2,3)
w_tr = F.conv_transpose2d(w1.permute(1,0,2,3), w2.permute(1,0,2,3), dilation=s1).permute(1,0,2,3)
input = torch.randn(1,1,32,32)

res = F.conv2d(F.conv2d(input, w1, stride=s1, padding=p1), w2, stride=s2, padding=p2)
#res_comb = F.conv2d(input, w_comb, stride=s1*s2, padding=p1)
res_tr = F.conv2d(input, w_tr, stride=s1 * s2, padding=p1+s1*p2)
#print((res - res_comb).abs().sum())
print((res - res_tr).abs().sum())

# Dilation should be sM*sM+1*...sN-1 for Layer M to Layer N
# Padding should be pM + (sM*pM+1) + (sM * sM+1 * pM+2) ... 
# 
#%%
def merge_weights(w1, w2, s1, s2, p1, p2):
    w_merge = F.conv_transpose2d(w1.permute(1,0,2,3), w2.permute(1,0,2,3), dilation=s1).permute(1,0,2,3)
    stride_merge = s1 * s2
    padding_merge = p1 + s1 * p2
    return w_merge, stride_merge, padding_merge
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
import numpy as np
degrees = np.load('data/degrees_MNIST.npy').transpose()
x = np.arange(6000)
for i in range(4):
    plt.plot(x, degrees[i])
    plt.title('Layer %i '%(i+1))
    plt.xlabel('# of update steps')
    plt.ylabel('Degrees (°)')
    plt.ylim(40, 100)
    plt.savefig('img/MNIST_DFA_l%i.png'%(i+1))
    plt.close()

#%%
showrange = 6000
fig = plt.figure(1)
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)
x = np.arange(showrange)
lines = []
for i in range(4):
    l, = ax.plot(x, degrees[i][:showrange])
    lines.append(l)
ax.legend(lines, ['layer %i'%i for i in range(4)])
ax.set_xlabel('# of update steps')
ax.set_ylabel('Degrees (°)')
plt.savefig('img/MNIST_DFA_%isteps.png'%showrange)
    


