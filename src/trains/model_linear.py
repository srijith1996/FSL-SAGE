import torch
import torch.nn as nn
import torch.nn.functional as F


######################## Cifar ##########################
# Client-side Model
class Client_model_cifar(nn.Module):
    def __init__(self):
        super(Client_model_cifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.LocalResponseNorm(size=4, alpha=0.001 / 9.0,
                                          beta=0.75, k=1.0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=5, padding=2)
        self.norm2 = nn.LocalResponseNorm(size=4, alpha=0.001 / 9.0,
                                          beta=0.75, k=1.0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        
        return x
    
# Auxiliary Model
class Auxiliary_model_cifar(nn.Module):
    def __init__(self, bias=True):
        super(Auxiliary_model_cifar, self).__init__()
        self.bias = bias
        self.fc = nn.Linear(in_features=6 * 6 * 64, out_features=2304, bias=bias) # Estimator of the gradients   

    def set_weight(self, W, c=None):
        self.fc.weight.data = W #nn.Parameter(W)
        if self.bias: self.fc.bias.data = c

    def align_loss(self, X, Y, lambda_reg=1e-3):
        loss = self.unc_loss(X, Y)
        if self.bias: 
            c = self.fc.bias.data[:, None]
            loss += (lambda_reg / 2) * torch.norm(c)**2
        loss += (lambda_reg / 2) * torch.norm(self.fc.weight.data, p='fro')**2
        return loss

    def unc_loss(self, X, Y):
        if self.bias:
            c = self.fc.bias.data[:, None]
            one = torch.ones((X.shape[0], 1)).to(X.device)
            loss = (1 / (2 * X.shape[0])) * torch.norm(self.fc.weight.data @ X.T
                                       + c @ one.T - Y.T, p='fro')**2
        else:
            loss = (1 / (2 * X.shape[0])) * torch.norm(self.fc.weight.data @ X.T
                                       - Y.T, p='fro')**2
        return loss
            
    def align(self, X, Y, lambda_reg=1e-1):
        print(f"Loss Before aux update: {self.unc_loss(X, Y)}")
        XtX = X.T @ X + X.shape[0] * lambda_reg * torch.eye(X.shape[1]).to(X.device)
        P = torch.linalg.solve(XtX, X, left=False)
        if self.bias:
            one = torch.ones((X.shape[0], 1)).to(X.device)
            q = P @ X.T @ one
            c = Y.T @ (one - q) / (X.shape[0] * (1 + lambda_reg) - one.T @ q)
            W = (Y.T - c @ one.T) @ P
            c = torch.flatten(c)
        else:
            W = Y.T @ P
            c = None
        self.set_weight(W, c)
        print(f"Loss After aux update: {self.unc_loss(X, Y)}")

    def forward(self, x):
        # print("AUX MODEL WEIGHT SHAPE")
        # print(x.size())
        # print(self.fc.weight.size())
        #with torch.no_grad():
        #    grad_estimate = torch.matmul(x, self.fc.weight)
        #return grad_estimate
        return self.fc(x)
 
 # Server-side Model
class Server_model_cifar(nn.Module):
    def __init__(self):
        super(Server_model_cifar, self).__init__()
        self.fc1 = nn.Linear(in_features=6 * 6 * 64, out_features=384)
        self.fc2 = nn.Linear(in_features=384, out_features=192)
        self.olayer = nn.Linear(in_features=192, out_features=10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.olayer(x), dim=1)
        
        return x

    
# # Not touching Femnist model for now

# ######################## Femnist ##########################
# # Client-side Model
# class Client_model_femnist(nn.Module):
#     def __init__(self):
#         super(Client_model_femnist, self).__init__()
#         self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
#         self.max_pooling = nn.MaxPool2d(2, stride=2)
#         self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
#         self.dropout_1 = nn.Dropout(0.25)
#         self.flatten = nn.Flatten()
        
#     def forward(self, x):
#         x = torch.unsqueeze(x, 1)
#         x = self.conv2d_1(x)
#         x = self.conv2d_2(x)
#         x = self.max_pooling(x)
#         x = self.dropout_1(x)
#         x = self.flatten(x)
        
#         return x

# # Auxiliary Model
# class Auxiliary_model_femnist(nn.Module):
#     def __init__(self):
#         super(Auxiliary_model_femnist, self).__init__()
#         self.fc = nn.Linear(in_features=9216, out_features=62)        

#     def forward(self, x):
#         x = self.fc(x)
#         return x 
    
# # Server-side Model
# class Server_model_femnist(nn.Module):
#     def __init__(self, only_digits=False):
#         super(Server_model_femnist, self).__init__()
#         self.linear_1 = nn.Linear(9216, 128)
#         self.dropout_2 = nn.Dropout(0.5)
#         self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
#         self.relu = nn.ReLU()
#         # self.softmax = nn.Softmax(dim=1)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.relu(self.linear_1(x))
#         x = self.dropout_2(x)
#         x = self.softmax(self.linear_2(x))
        
#         return x