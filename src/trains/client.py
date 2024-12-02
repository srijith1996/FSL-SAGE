# ------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# ------------------------------------------------------------------------------
class Client():
    def __init__(self, id, train_loader, client, auxiliary, c_args, device='cpu'):
        self.model = client 
        self.auxiliary_model = auxiliary
        self.criterion = nn.NLLLoss().to(device)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=c_args["lr"])       
        self.optimizer = optim.Adam(self.model.parameters(), lr=c_args["lr"])       
        #self.auxiliary_criterion = nn.NLLLoss().to(device)
        #self.auxiliary_optimizer = optim.SGD(
        #    self.auxiliary_model.parameters(), lr=c_args["lr"]
        #)
        
        self.train_loader = train_loader
        self.epochs = c_args['epoch']
        self.dataset_size = len(self.train_loader) * c_args["batch_size"] 

# ------------------------------------------------------------------------------
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

    def conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        return x

    def forward(self, x):
        x = self.conv_forward(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        
        return x

# ------------------------------------------------------------------------------