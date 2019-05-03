import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from models.building_blocks import *
from torch.autograd import Variable
import numpy as np
import math
import time
#%%
# Hyper Parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MNIST Dataset 
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = Linear_IFA(hidden_size, num_classes)
        self.fb1 = Feedback_Reciever(10)
        self.fb2 = Feedback_Reciever(10)
        self.fb3 = Feedback_Reciever(10)
        self.fb4 = Feedback_Reciever(10)
    
    def forward(self, x):
        out = self.fc1(x)
        out, dm1 = self.fb1(F.tanh(out))
        out = self.fc2(out)
        out, dm2 = self.fb2(F.tanh(out))
        out = self.fc3(out)
        out, dm3 = self.fb3(F.tanh(out))
        out = self.fc4(F.tanh(out))
        out, dm4 = self.fb4(F.tanh(out))
        out = F.log_softmax(self.fc5(out, dm1, dm2, dm3, dm4))
        return out
    
    def eval_alignment(self):
        T4 = self.fc5.weight.clone()    # 10, hidden_size
        T3 = torch.mm(T4, self.fc4.weight.clone())
        T2 = torch.mm(T3, self.fc3.weight.clone())
        T1 = torch.mm(T2, self.fc2.weight.clone())
        deg4 = 180 * math.acos(F.cosine_similarity(T4.view(1,-1),self.fb4.weight_fb.view(1,-1))) / math.pi
        deg3 = 180 * math.acos(F.cosine_similarity(T3.view(1,-1),self.fb3.weight_fb.view(1,-1))) / math.pi
        deg2 = 180 * math.acos(F.cosine_similarity(T2.view(1,-1),self.fb2.weight_fb.view(1,-1))) / math.pi
        deg1 = 180 * math.acos(F.cosine_similarity(T1.view(1,-1),self.fb1.weight_fb.view(1,-1))) / math.pi
        return deg1, deg2, deg3, deg4

class ByPassNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ByPassNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = Linear_IFA(hidden_size, num_classes)
        self.fb1 = Feedback_Reciever(10)
        self.fb2 = Feedback_Reciever(10)
        self.fb3 = Feedback_Reciever(10)
        self.fb4 = Feedback_Reciever(10)
    
    def forward(self, x):
        out = self.fc1(x)
        out, dm1 = self.fb1(F.tanh(out))
        out = self.fc2(out)
        out, dm2 = self.fb2(F.tanh(out))
        out = self.fc3(out)
        out, dm3 = self.fb3(F.tanh(out))
        out = self.fc4(F.tanh(out))
        out, dm4 = self.fb4(F.tanh(out))
        out = F.log_softmax(self.fc5(out, dm1, dm2, dm3, dm4))
        return out
    
    def sym_params(self):
        T4 = self.fc5.weight.clone()    # 10, hidden_size
        T3 = torch.mm(T4, self.fc4.weight.clone())
        T2 = torch.mm(T3, self.fc3.weight.clone())
        T1 = torch.mm(T2, self.fc2.weight.clone())
        self.fb4.weight_fb.data = T4   # 10, hidden_size
        self.fb3.weight_fb.data = T3
        self.fb2.weight_fb.data = T2
        self.fb1.weight_fb.data = T1
        





#net = Net(input_size, 256, num_classes)
net = ByPassNet(input_size, 256, num_classes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net.to(device) 
    #%%
# Loss and Optimizer
criterion = nn.NLLLoss()  
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)  
t0 = time.time()
degrees = np.zeros([num_epochs * 600, 4])
out1 = net(torch.randn(1,784).to(device))
# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        net.sym_params()
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28).to(device))
        labels = Variable(labels.to(device))
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        # Save alignment info
        #degrees[600*epoch + i] = net.eval_alignment()

        # Run backward pass
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
                   
#np.save('data/degrees_MNIST.npy', degrees)
# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28)).to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %.2f %%' % (100.0 * float(correct) / total))
print('Total Time: %.2f'%(time.time()-t0))
#%%
# Save the Model
torch.save(net.state_dict(), 'model.pkl')
