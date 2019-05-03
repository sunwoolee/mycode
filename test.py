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
w1 = torch.randn(3,2,4,4)
w2 = torch.randn(5,3,7,7)

w_comb = F.conv2d(w1.flip(2,3).permute(1,0,2,3), w2, padding=6).flip(2,3).permute(1,0,2,3)
w_tr = F.conv_transpose2d(w1.permute(1,0,2,3), w2.permute(1,0,2,3), stride=2).permute(1,0,2,3)
input = torch.randn(1,2,32,32)

res = F.conv2d(F.conv2d(input, w1), w2)
res_comb = F.conv2d(input, w_comb)
res_tr = F.conv2d(input, w_tr, stride=2)
print((res - res_comb).abs().sum())
#print((res - res_tr).abs().sum())
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
    


