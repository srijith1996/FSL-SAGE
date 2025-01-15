# -----------------------------------------------------------------------------
from dataclasses import dataclass
import os, importlib
import torch
import torch.nn as nn
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
def config_optimizer(params, cfg):
    if cfg.name == 'adam':
        optim = torch.optim.Adam
    elif cfg.name == 'sgd':
        optim = torch.optim.SGD
    else:
        raise Exception(f"Optimizer {cfg.name} is not configured.")

    return optim(params, **cfg.options)

# ------------------------------------------------------------------------------
def config_lr_scheduler(optimizer, cfg):
    if cfg is None: return None
    if cfg.name == 'multistep_lr':
        sched = torch.optim.lr_scheduler.MultiStepLR
    elif cfg.name == 'step_lr':
        sched = torch.optim.lr_scheduler.StepLR
    else:
        raise Exception(f"LR Scheduler {cfg.name} is not configured")

    return sched(optimizer, **cfg.options)

# ------------------------------------------------------------------------------
class Client():
    def __init__(self,
        id, train_loader, client, cfg_model, device='cpu'
    ):
        self.id = id
        self.model = client 
        self.criterion = nn.NLLLoss().to(device)

        self.optimizer = config_optimizer(
            self.model.parameters(), cfg_model.optimizer
        )
        self.optimizer_options = cfg_model.optimizer

        self.lr_scheduler = config_lr_scheduler(
            self.optimizer, cfg_model.lr_scheduler
        ) if "lr_scheduler" in cfg_model and cfg_model.lr_scheduler else None
        self.lr_scheduler_options = cfg_model.lr_scheduler \
            if "lr_scheduler" in cfg_model else None

        self.train_loader = train_loader
        self.epochs = cfg_model.epoch
        self.dataset_size = len(self.train_loader) * cfg_model.batch_size

    def init_auxiliary(self, auxiliary, cfg_aux):
        self.auxiliary_model = auxiliary

        # optimizer and lr schedules for the auxiliary model
        self.aux_optimizer = config_optimizer(
                self.auxiliary_model.parameters, cfg_aux.optimizer
            )
        self.auxiliary_model.set_optimizer_lr_scheduler(
            self.aux_optimizer, config_lr_scheduler(
                self.aux_optimizer, cfg_aux.lr_scheduler
            ) if 'lr_scheduler' in cfg_aux else None
        )

# -----------------------------------------------------------------------------
class Server():
    def __init__(self,server, cfg, device='cpu'):
        self.model = server
        self.criterion = nn.NLLLoss().to(device)
        self.alignLoss = nn.MSELoss().to(device)

        self.optimizer = config_optimizer(
            self.model.parameters(), cfg.optimizer
        )
        self.optimizer_options = cfg.optimizer

        self.lr_scheduler = config_lr_scheduler(
            self.optimizer, cfg.lr_scheduler
        ) if "lr_scheduler" in cfg and cfg.lr_scheduler else None

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