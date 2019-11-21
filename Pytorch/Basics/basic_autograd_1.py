import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as trnsforms


# Create Tensor
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(1., requires_grad=True)

# Create a computaional graph 
y = w * x + b 

# Comput gradients using backward() 
y.backward()

print(x.grad, w.grad, b.grad)