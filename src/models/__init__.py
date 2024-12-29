# -----------------------------------------------------------------------------
from dataclasses import dataclass
import os, importlib
import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple, Any, Union

from models import aux_models
from omegaconf import open_dict
from utils import opt_utils

# -----------------------------------------------------------------------------
# name: str ->
#       client_server_pair: tuple of (client_model_cls, server_model_cls)
# note that the components are not instances of nn.Module, but subclasses or
# callable constructors of it
CLIENT_SERVER_MODEL_REGISTRY: Dict[
    str, Tuple[Callable, Union[Callable, None]]
] = dict()

# -----------------------------------------------------------------------------
def register_client_server_pair(
    name, client, server=None, client_to_server_params_fn=None
):
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
    CLIENT_SERVER_MODEL_REGISTRY[name] = (
        client, server, client_to_server_params_fn
    )

# -----------------------------------------------------------------------------
@dataclass
class ModelPackage():
    client: Callable
    server: Callable
    auxiliary: Callable
    client_to_server_params: Callable

# -----------------------------------------------------------------------------
def model_package(model_name: str, aux_model_name: str) -> ModelPackage:
    assert model_name in CLIENT_SERVER_MODEL_REGISTRY.keys(), \
        f"Model {model_name} not found in registry!"
    assert aux_model_name in aux_models.AUXILIARY_MODEL_REGISTRY.keys(), \
        f"Auxiliary model {aux_model_name} not round in registry!"

    model_pack = ModelPackage(
        CLIENT_SERVER_MODEL_REGISTRY[model_name][0],
        CLIENT_SERVER_MODEL_REGISTRY[model_name][1],
        aux_models.AUXILIARY_MODEL_REGISTRY[aux_model_name],
        CLIENT_SERVER_MODEL_REGISTRY[model_name][2],
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
    id      : int
    model   : Union[nn.Module, Callable]
    optimizer: Any
    dataset_size: int
    train_loader: torch.utils.data.DataLoader
    epochs  : int
    auxiliary_model : aux_models.AuxiliaryModel

    def __init__(self,
        id, train_loader, client, cfg_model, device='cpu'
    ):
        self.id = id
        self.model = client 

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
                self.auxiliary_model.parameters(), cfg_aux.optimizer
            )
        self.auxiliary_model.set_optimizer_lr_scheduler(
            self.aux_optimizer, config_lr_scheduler(
                self.aux_optimizer, cfg_aux.lr_scheduler
            ) if 'lr_scheduler' in cfg_aux else None
        )

# -----------------------------------------------------------------------------
# source: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
class MaskingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(MaskingCrossEntropyLoss, self).__init__()
        #self.cel = nn.CrossEntropyLoss(
        #    weight=weight, label_smoothing=smoothing, reduction='none',
        #    ignore_index=-1
        #)
        self.smoothing = smoothing


    def forward(self, pred, target, mask=None):

        _batch, _len = pred.shape[:2]
        logprobs = torch.nn.functional.log_softmax(pred.view(-1, pred.size(-1)), dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.view(-1).unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        loss = loss.view(_batch, _len)
 
        if mask is None:
            mask = torch.ones(
                loss.shape, dtype=loss.dtype, device=loss.device
            )

        loss = loss * mask 
        return loss.sum() / (mask.sum() + 0.0001)

# -----------------------------------------------------------------------------
class Server():
    model: nn.Module
    criterion : Callable
    optimizer : Any
    alignment_loss: Callable

    def __init__(self,
        server, cfg, problem_type='image_classification', device='cpu'
    ):
        self.model = server
        self.alignment_loss = nn.MSELoss().to(device)

        self.optimizer = config_optimizer(
            self.model.parameters(), cfg.optimizer
        )
        self.optimizer_options = cfg.optimizer

        self.lr_scheduler = config_lr_scheduler(
            self.optimizer, cfg.lr_scheduler
        ) if "lr_scheduler" in cfg and cfg.lr_scheduler else None

        if problem_type != 'image_classification':
            if 'label_smooth' not in cfg:
                with open_dict(cfg): cfg['label_smooth'] = 0.0
            self.criterion = MaskingCrossEntropyLoss(
                smoothing=cfg.label_smooth).to(device)
        else:
            self.criterion = nn.NLLLoss().to(device)

    def forward(self, pred, target, mask=None):

        loss = self.cel(pred, target)
        if mask is None:
            mask = torch.ones(
                loss.shape, dtype=loss.dtype, device=loss.device
            )

        loss = loss * mask 
        return loss.sum() / (mask.sum() + 0.0001)

# -----------------------------------------------------------------------------
class Server():
    model: nn.Module
    criterion : Callable
    optimizer : Any
    alignment_loss: Callable

    def __init__(self, server, cfg, device='cpu'):
        self.model = server

        if 'label_smooth' not in cfg:
            with open_dict(cfg): cfg['label_smooth'] = 0.0
        self.criterion = MaskingCrossEntropyLoss(smoothing=cfg.label_smooth)
        self.alignment_loss = nn.MSELoss().to(device)

        if 'optimizer' in cfg:
            with open_dict(cfg): cfg.optimizer['lr'] = cfg.lr
            self.optimizer = opt_utils.create_adam_optimizer_from_args(
                self.model.parameters(), cfg.optimizer,
                grouped_parameters=None
            )
        else:
            self.optimizer = Adam(
                self.model.parameters(),
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