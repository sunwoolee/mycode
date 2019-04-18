#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class linear_ifa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, *dummies):
        """
        Add dummies from the activations of layers you want the ifa to transfer errors
        """
        output = F.linear(input, weight, bias)
        ctx.save_for_backward(input, weight, bias, *dummies)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, *dummies = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_weight = F.linear(input.t(), grad_output.t()).t()
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
        grad_input = torch.Tensor(input.size()).zero_().to(input.device)
        grad_dummies = [grad_output.clone() for dummy in dummies]
        return tuple([grad_input, grad_weight, grad_bias, *grad_dummies])


class feedback_reciever(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_fb):
        """
        Dummy should be attached to the IFA layer for proper IFA error calculation
        weight_fb : of shape (input, ifa_neurons)
        grad_dummy: of shape (batch_size, ifa_neurons)
        """
        output = input.clone()
        dummy = torch.Tensor(input.size()[0],weight_fb.size()[0]).zero_().to(input.device)
        ctx.save_for_backward(input, weight_fb)
        return output, dummy
    
    @staticmethod
    def backward(ctx, grad_output, grad_dummy):
        input, weight_fb = ctx.saved_tensors
        grad_weight_fb = None
        grad_input = torch.mm(grad_dummy, weight_fb).view(input.size()) # Batch_size, input
        return grad_input, grad_weight_fb
        

class Linear_IFA(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_IFA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).zero_())
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight)
    
    def forward(self, input, *dummies):
        return linear_ifa.apply(input, self.weight, self.bias, *dummies)
        

class Feedback_Reciever(nn.Module):
    def __init__(self, in_features, connect_features):
        super(Feedback_Reciever, self).__init__()
        self.in_features = in_features
        self.connect_features = connect_features
        self.weight_fb = nn.Parameter(torch.Tensor(connect_features, in_features))
        nn.init.kaiming_uniform_(self.weight_fb)
    
    def forward(self, input):
        return feedback_reciever.apply(input, self.weight_fb)
        
    
class linear_fa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, weight_fa):
        output = F.linear(input, weight, bias)
        ctx.save_for_backward(input, weight, bias, weight_fa)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, weight_fa = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_weight_fa = None
        grad_weight = F.linear(input.t(), grad_output.t()).t()
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
        grad_input = F.linear(grad_output, weight_fa.t())
        return grad_input, grad_weight, grad_bias, grad_weight_fa


class Linear_FA(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_FA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #k = math.sqrt(1. / self.in_features)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).zero_())
        else:
            self.register_parameter('bias', None)
        self.weight_fa = nn.Parameter(torch.FloatTensor(out_features, in_features).uniform_(-1,1), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight)
        #nn.init.kaiming_uniform_(self.weight_fa)
        
    def forward(self, input):
        return linear_fa.apply(input, self.weight, self.bias, self.weight_fa)

class conv2d_fa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, weight_fa, stride=1, padding=0):
        output = F.conv2d(input, weight, bias, stride, padding)
        ctx.stride = stride
        ctx.padding = padding
        ctx.save_for_backward(input, weight, bias, weight_fa)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, weight_fa = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        grad_input = grad_weight = grad_bias = grad_weight_fa = None
        grad_weight = F.conv2d(input.permute(1,0,2,3), grad_output.permute(1,0,2,3), bias=None, stride=stride, padding=padding).permute(1,0,2,3)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        grad_input = F.conv_transpose2d(grad_output, weight_fa, bias=None, stride=stride, padding=padding)
        return grad_input, grad_weight, grad_bias, grad_weight_fa, None, None


class Conv2d_FA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_FA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels).zero_())
        else:
            self.register_parameter('bias', None)
        self.weight_fa = nn.Parameter(torch.FloatTensor(out_channels, in_channels, kernel_size, kernel_size).uniform_(-1,1), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight)
        #nn.init.kaiming_uniform_(self.weight_fa)
    
    def forward(self, input):
        return conv2d_fa.apply(input, self.weight, self.bias, self.weight_fa, self.stride, self.padding)

#%%
"""Sanity Check"""
conv1_check = Conv2d_FA(3,4,5)
conv1_base = nn.Conv2d(3,4,5)
conv2_check = Conv2d_FA(4,10,3)
conv2_base = nn.Conv2d(4,10,3)
lin1_check = Linear_FA(10,10)
lin1_base = nn.Linear(10,10)
lin2_check = Linear_FA(10,10)
lin2_base = nn.Linear(10,10)
for param_check, param_base in zip([conv1_check, conv2_check, lin1_check, lin2_check], [conv1_base, conv2_base, lin1_base, lin2_base]):
    param_check.weight_fa.data = param_base.weight.data.clone()
    for p1, p2 in zip(param_check.parameters(), param_base.parameters()):
        p1.data = p2.data.clone()
inputx = torch.Tensor(2,3,7,7).uniform_()
y = F.log_softmax(lin2_check(lin1_check(conv2_check(conv1_check(inputx)).view(-1,10))))
y_t = torch.Tensor([2,4]).long()
loss = F.nll_loss(y, y_t)
y_base = F.log_softmax(lin2_check(lin1_check((conv2_base(conv1_base(inputx)).view(-1,10)))))
y_t_base = torch.Tensor([2,4]).long()
loss_base = F.nll_loss(y_base, y_t_base)
loss_base.backward()
loss.backward()
print((conv1_check.weight.grad - conv1_base.weight.grad).abs().sum())
print((conv1_base.bias.grad-conv1_check.bias.grad).abs().sum())
