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
    def __init__(self):
        super(Auxiliary_model_cifar, self).__init__()
        self.fc = nn.Linear(in_features=6 * 6 * 64, out_features=10)        

    def forward(self, x):
        x = F.log_softmax(self.fc(x), dim=1)
        
        return x
 
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
#         x = F.log_softmax(self.fc(x), dim=1)       
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