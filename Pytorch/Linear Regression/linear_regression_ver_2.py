import torch 
from torch.autograd import Variable

num_epochs = 100
learning_rate = 0.01

x_data = Variable(torch.Tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]]))
y_data = Variable(torch.Tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]]))

class LinearRegressionModel(torch.nn.Module):
    
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
        
model = LinearRegressionModel()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    
    pred_y = model(x_data)
    loss = criterion(pred_y, y_data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1)%10 == 0:
        print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch+1,num_epochs, loss.item()))
        
new_x = Variable(torch.Tensor([[7.5]]))
new_pred = model(new_x)
print("Prediction for input {} is {:.4f}: ".format(new_x.detach().numpy()[0][0], new_pred.detach().numpy()[0][0]))