import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

input_size = 1
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Random sample data
x_data =np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_data = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]],dtype=np.float32)
                    
# Define linear regression model 
model = nn.Linear(input_size,output_size)


# Define Loss function and optimizer 
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Model Training 
for epoch in range(num_epochs):
    # Conversion : numpy to tensor 
    inputs = torch.from_numpy(x_data)
    targets = torch.from_numpy(y_data)
    
    #Forward pass 
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1)%5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        
# Plot the original vs fitted line 
predicted = model(torch.from_numpy(x_data)).detach().numpy()
plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, predicted, label='Fitted line')
plt.legend()
plt.show()

# save model ccheckpoint 
torch.save(model.state_dict(), './Linear Regression/model.ckpt')
        