import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as trnsforms

# Create tensors 
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Creat fully connected layer 
linear = nn.Linear(3,2)
print('w: ', linear.weight)
print('b: ', linear.bias)
# print('Params: ',linear.parameters)

# Build loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward passing
pred = linear(x)

# Compute loss
loss = criterion(pred, y)

# Backward passing
loss.backward()

# Gradients 
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# One step gradient descent
optimizer.step()

# loss after 1-step gradient descent
pred = linear(x)
loss = criterion(pred, y)
print('loss after inital optimization: ',loss.item())

# 2nd step gradient descent
optimizer.step()

# loss after 2-step gradient descent
pred = linear(x)
loss = criterion(pred, y)
print('loss after 2nd optimization: ',loss.item())

# 3rd step gradient descent
optimizer.step()

# loss after 3-step gradient descent
pred = linear(x)
loss = criterion(pred, y)
print('loss after 3rd optimization: ',loss.item())
