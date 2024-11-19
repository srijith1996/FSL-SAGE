# ------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# ------------------------------------------------------------------------------
class Server():
    def __init__(self, server, s_args, device='cpu'):
        #if c_args['dataset'] == "cifar":
        self.model = server
        # elif c_args['dataset'] == "femnist":
        #     self.model = model.Server_model_femnist()
        self.criterion = nn.NLLLoss().to(device)
        self.alignLoss = nn.MSELoss().to(device)

        # Srijith: If we optimize the server once for every client, we want to
        # divide the learning rate by the number of active clients
        #self.optimizer = optim.SGD(self.model.parameters(), lr=(s_args["lr"] /
        #s_args["activated"]))
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=(s_args["lr"] / s_args["activated"])
        )

# ------------------------------------------------------------------------------
class Server_model_cifar(nn.Module):
    def __init__(self):
        super(Server_model_cifar, self).__init__()
        self.fc1 = nn.Linear(in_features=6 * 6 * 64, out_features=384)
        #self.fc1 = nn.Linear(in_features=4096, out_features=384)
        self.fc2 = nn.Linear(in_features=384, out_features=192)
        self.olayer = nn.Linear(in_features=192, out_features=10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.olayer(x), dim=1)
        
        return x

# ------------------------------------------------------------------------------