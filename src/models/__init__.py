# -----------------------------------------------------------------------------
from dataclasses import dataclass
import os, importlib
import torch.nn as nn
from torch.optim import Adam
from typing import Callable, Dict, Tuple, Type, Union

from models import aux_models

# -----------------------------------------------------------------------------
# name: str ->
#       client_server_pair: tuple of (client_model_cls, server_model_cls)
# note that the components are not instances of nn.Module, but subclasses or
# callable constructors of it
CLIENT_SERVER_MODEL_REGISTRY: Dict[
    str, Tuple[Callable, Union[Callable, None]]
] = dict()

# -----------------------------------------------------------------------------
def register_client_server_pair(name, client, server=None):
    if name in CLIENT_SERVER_MODEL_REGISTRY:
        raise ValueError(
            'Cannot register duplicate client-server pair {}'.format(name)
        )
    if not isinstance(client, Callable):
        raise ValueError(
            'Client model must be a callable or nn.Module constructor'
        )
    if server is not None and not isinstance(server, Callable):
        raise ValueError(
            'Server model must be a callable or nn.Module constructor'
        )
    CLIENT_SERVER_MODEL_REGISTRY[name] = (client, server)

# -----------------------------------------------------------------------------
@dataclass
class ModelPackage():
    client: Callable
    server: Callable
    auxiliary: Callable

# -----------------------------------------------------------------------------
def model_package(model_name: str, aux_model_name: str) -> ModelPackage:
    assert model_name in CLIENT_SERVER_MODEL_REGISTRY.keys(), \
        f"Model {model_name} not found in registry!"
    assert aux_model_name in aux_models.AUXILIARY_MODEL_REGISTRY.keys(), \
        f"Auxiliary model {aux_model_name} not round in registry!"

    model_pack = ModelPackage(
        CLIENT_SERVER_MODEL_REGISTRY[model_name][0],
        CLIENT_SERVER_MODEL_REGISTRY[model_name][1],
        aux_models.AUXILIARY_MODEL_REGISTRY[aux_model_name]
    )
    return model_pack

# ------------------------------------------------------------------------------
class Client():
    def __init__(self,
        id, train_loader, client, auxiliary, cfg, device='cpu'
    ):
        self.model = client 
        self.auxiliary_model = auxiliary
        self.criterion = nn.NLLLoss().to(device)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=c_args["lr"])       
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr)       
        #self.auxiliary_criterion = nn.NLLLoss().to(device)
        #self.auxiliary_optimizer = optim.SGD(
        #    self.auxiliary_model.parameters(), lr=c_args["lr"]
        #)
        
        self.train_loader = train_loader
        self.epochs = cfg.epoch
        self.dataset_size = len(self.train_loader) * cfg.batch_size 

# -----------------------------------------------------------------------------
class Server():
    def __init__(self,server, cfg, device='cpu'):
        #if c_args['dataset'] == "cifar":
        self.model = server
        # elif c_args['dataset'] == "femnist":
        #     self.model = model.Server_model_femnist()
        self.criterion = nn.NLLLoss().to(device)
        self.alignLoss = nn.MSELoss().to(device)

        # If we optimize the server once for every client, we want to
        # divide the learning rate by the number of active clients
        self.optimizer = Adam(
            self.model.parameters(),
            #lr=(cfg.lr / num_clients)
            lr=cfg.lr
        )

# -----------------------------------------------------------------------------
# Automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file[0].isalpha():
        module = file[:file.find('.py')]
        importlib.import_module('models.' + module)

# -----------------------------------------------------------------------------
# Automatically import any Python files in the models/aux_models directory
for file in os.listdir(os.path.join(os.path.dirname(__file__), 'aux_models')):
    if file.endswith('.py') and file[0].isalpha():
        module = file[:file.find('.py')]
        importlib.import_module('models.aux_models.' + module)

# ------------------------------------------------------------------------------