# ------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

from models import register_client_server_pair

# ------------------------------------------------------------------------------
class Client_model_cifar(nn.Module):
    def __init__(self, n_channels=3):
        super(Client_model_cifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64,
                               kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.LocalResponseNorm(size=4, alpha=0.001 / 9.0,
                                          beta=0.75, k=1.0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=5, padding=2)
        self.norm2 = nn.LocalResponseNorm(size=4, alpha=0.001 / 9.0,
                                          beta=0.75, k=1.0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)        

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0., std=0.05)
            nn.init.normal_(module.bias, mean=0., std=0.05)
            #if module.bias is not None:
            #    nn.init.zeros_(module.bias)

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
class Server_model_cifar(nn.Module):
    def __init__(self, n_input=6*6*64, n_output=10):
        super(Server_model_cifar, self).__init__()
        self.fc1 = nn.Linear(in_features=n_input, out_features=384)
        #self.fc1 = nn.Linear(in_features=4096, out_features=384)
        self.fc2 = nn.Linear(in_features=384, out_features=192)
        self.olayer = nn.Linear(in_features=192, out_features=n_output)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0., std=0.05)
            nn.init.normal_(module.bias, mean=0., std=0.05)
            #if module.bias is not None:
            #    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.olayer(x)
        
        return x

# ------------------------------------------------------------------------------
register_client_server_pair(
    'simple_conv', Client_model_cifar, Server_model_cifar
)

# ------------------------------------------------------------------------------