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
# https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/optimizer.py
def create_grouped_parameters(model, no_decay_bias): # args):
    if not no_decay_bias:
        optimizer_grouped_parameters = [{
            "params": [p for n, p in model.named_parameters()],
        }]
    else:
        no_decay = ["bias", "layer_norm.weight"]

        optimizer_grouped_parameters = [{
            "params": [p for n, p in model.named_parameters() \
                if not any(nd in n for nd in no_decay)],
        }, {
            "params": [p for n, p in model.named_parameters() \
                if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0,
        }]
    return optimizer_grouped_parameters

# ------------------------------------------------------------------------------
def config_optimizer(model, cfg, no_decay_bias=False):

    params = create_grouped_parameters(model, no_decay_bias)
    if cfg.name == 'adam':
        optim = torch.optim.Adam
    elif cfg.name == 'sgd':
        optim = torch.optim.SGD
    elif cfg.name == 'adamw':
        optim = opt_utils.AdamW
    else:
        raise Exception(f"Optimizer {cfg.name} is not configured.")

    return optim(params, **cfg.options)

# ------------------------------------------------------------------------------
def config_lr_scheduler(optimizer, cfg, **kwargs):
    if cfg is None: return None
    if cfg.name == 'multistep_lr':
        sched = torch.optim.lr_scheduler.MultiStepLR
    elif cfg.name == 'step_lr':
        sched = torch.optim.lr_scheduler.StepLR

    elif cfg.name == 'lambda_lr':
        assert 'max_steps' in kwargs.keys()
        sched = torch.optim.lr_scheduler.LambdaLR
        def lr_lambda(current_step):
            if current_step < cfg.warmup_steps:
                return float(current_step) / float(max(1, cfg.warmup_steps))
            return max(0.0, float(kwargs['max_steps'] - current_step) /
                       float(max(1, kwargs['max_steps'] - cfg.warmup_steps)))

        return sched(optimizer, lr_lambda, last_epoch=-1)
    else:
        raise Exception(f"LR Scheduler {cfg.name} is not configured")

    return sched(optimizer, **cfg.options)

# ------------------------------------------------------------------------------
max_steps = -1
class Client():
    id      : int
    model   : Union[nn.Module, Callable]
    optimizer: Any
    dataset_size: int
    train_loader: torch.utils.data.DataLoader
    epochs  : int
    auxiliary_model : aux_models.AuxiliaryModel

    def __init__(self,
        id, train_loader, client, cfg_model, rounds, device='cpu'
    ):
        self.id = id
        self.model = client 

        self.optimizer = config_optimizer(
            self.model, cfg_model.optimizer,
            no_decay_bias=cfg_model.no_decay_bias \
                if 'no_decay_bias' in cfg_model.keys() else False
        )
        self.optimizer_options = cfg_model.optimizer
        self.optimizer_no_decay_bias = cfg_model.no_decay_bias \
            if 'no_decay_bias' in cfg_model.keys() else False

        self.lr_scheduler = config_lr_scheduler(
            self.optimizer, cfg_model.lr_scheduler,
            max_steps=(len(train_loader) * cfg_model.epoch *  rounds)
        ) if "lr_scheduler" in cfg_model and cfg_model.lr_scheduler \
            else None
        self.lr_scheduler_options = cfg_model.lr_scheduler \
            if "lr_scheduler" in cfg_model else None
        global max_steps
        max_steps = (len(train_loader) * cfg_model.epoch *  rounds)
        self.max_steps = max_steps

        self.train_loader = train_loader
        self.epochs = cfg_model.epoch
        self.dataset_size = len(self.train_loader) * cfg_model.batch_size

    def init_auxiliary(self, auxiliary, cfg_aux):
        self.auxiliary_model = auxiliary

        # optimizer and lr schedules for the auxiliary model
        self.aux_optimizer = config_optimizer(
            self.auxiliary_model, cfg_aux.optimizer,
            no_decay_bias=cfg_aux.no_decay_bias \
                if 'no_decay_bias' in cfg_aux.keys() else False
        )
        global max_steps
        self.aux_lr_scheduler = config_lr_scheduler(
                self.aux_optimizer, cfg_aux.lr_scheduler,
                max_steps=max_steps
        ) if 'lr_scheduler' in cfg_aux else None

        self.auxiliary_model.set_optimizer_lr_scheduler(
            self.aux_optimizer, self.aux_lr_scheduler
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
        logprobs = torch.nn.functional.log_softmax(
            pred.view(-1, pred.size(-1)), dim=-1
        )
        nll_loss = -logprobs.gather(
            dim=-1, index=target.view(-1).unsqueeze(1)
        )
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
            self.model, cfg.optimizer,
            no_decay_bias=cfg.no_decay_bias \
                if 'no_decay_bias' in cfg.keys() else False
        )
        self.optimizer_options = cfg.optimizer

        global max_steps
        self.lr_scheduler = config_lr_scheduler(
            self.optimizer, cfg.lr_scheduler,
            max_steps=max_steps
        ) if "lr_scheduler" in cfg and cfg.lr_scheduler else None

        if problem_type != 'image_classification':
            if 'label_smooth' not in cfg:
                with open_dict(cfg): cfg['label_smooth'] = 0.0
            self.criterion = MaskingCrossEntropyLoss(
                smoothing=cfg.label_smooth).to(device)
        else:
            self.criterion = nn.NLLLoss().to(device)

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