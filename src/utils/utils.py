#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.7
#import random
import logging
import os, pathlib
import yaml
import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import List, Any
import numpy as np

import torch
import torch.nn as nn

def calculate_load(model):        
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size)  / 1024**2    # MB
    size_all_mb = (param_size + buffer_size)   # B
    return size_all_mb


def show_utils(args):
    '''
        Print system setup profile.
    '''

    logging.info('*'*80); print('SYSTEM CONFIGS'); print('*'*80)
    logging.info(f"  \\__ Algorithm:     {args.algorithm}")
    logging.info(f"  \\__ Dataset:       {args.dataset.name}")
    logging.info(f"  \\__ Distribution:  {args.dataset.distribution}")
    logging.info(f"  \\__ Model:         {args.model}")
    logging.info(f"  \\__ Save:          {args.save_path}")


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))

def compute_model_size(*models):
    sizes = [0.0 for _ in models]
    counts = [0 for _ in models]
    for i, m in enumerate(models):
        for p in m.parameters():
            counts[i] += p.nelement()
            sizes[i] += p.nelement() * p.element_size()
        for p in m.buffers():
            counts[i] += p.nelement()
            sizes[i] += p.nelement() * p.element_size()
        sizes[i] = sizes[i] / (1024 ** 2)

    return sizes, counts

def process_counts(counts):
    str_list = []
    for i, c in enumerate(counts):
        if c // int(1e12) > 0:
            str_list.append(f'{c / 1e12:.1f}T')
        elif c // int(1e9) > 0:
            str_list.append(f'{c / 1e9:.1f}B')
        elif c // int(1e6) > 0:
            str_list.append(f'{c / 1e6:.1f}M')
        elif c // int(1e3) > 0:
            str_list.append(f'{c / 1e3:.1f}K')
        else:
            str_list.append(f'{c:d}')
    return str_list

def sha256_of_yaml(yaml_dict):
    """Calculates the SHA-256 hash of a YAML dictionary."""

    # Serialize the dictionary to a YAML string
    yaml_string = yaml.dump(yaml_dict, sort_keys=True)

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256(yaml_string.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hex_digest = hash_object.hexdigest()

    return hex_digest

def create_save_dir(cfg):
    ## setup saving paths
    client_info = f"{cfg.dataset.name}-{cfg.dataset.distribution}"
    train_info = f"R{cfg.rounds}m{cfg.num_clients}E{cfg.model.client.epoch}"
    train_info += f"B{cfg.model.client.batch_size}"
    if 'fsl' in cfg.algorithm.name:
        train_info += f"q{cfg.algorithm.server_update_interval}"
    if cfg.algorithm.name == 'fsl_sage':
        train_info += f"l{cfg.algorithm.align_interval}"
    if cfg.dataset.distribution == 'noniid_dirichlet':
        train_info += f'-alp{cfg.dataset.alpha:.2e}'

    train_info += f"-seed{cfg.seed}"

    timestamp = datetime.now().strftime(r'%y%m%d-%H%M%S')
    dir_name = os.path.join(cfg.algorithm.name, cfg.model.name, client_info, train_info)
    exp_dir = os.path.join(dir_name, timestamp)
    if cfg.save:
        os.makedirs(os.path.join(cfg.save_dir_prefix, dir_name), exist_ok=True)
        p = os.path.join(cfg.save_dir_prefix, exp_dir)
        if os.path.exists(p): p = os.path.join(p, sha256_of_yaml(cfg))
        cfg.save_path = p

        pathlib.Path(p).mkdir(exist_ok=False)
        logging.info(f"\033[1;36m[NOTICE] Directory '{p}' build.\033[0m")

        # add a plot path in case required
        cfg['plot_path'] = os.path.join(cfg['save_path'], 'plots')
        os.makedirs(cfg['plot_path'], exist_ok=True)

@dataclass
class CheckpointData():
    round: int
    metric_name: str
    metric_val: float
    client_model: nn.Module
    server_model: nn.Module
    auxiliary_models: List[nn.Module]
    client_optimizers: List[Any]
    server_optimizer: Any
    auxiliary_optimizers: List[Any]

class Checkpointer():
    best_metric: float
    metric_minimize: bool

    def __init__(self, metric_name, metric_obj='max', save_dir='../saves'):
        '''Initializes a checkpointer object to save or load checkpoints'''

        self.metric_name = metric_name

        if metric_obj == 'max':
            self.best_metric = 0.0
        elif metric_obj == 'min':
            self.best_metric = np.Inf
        else:
            raise Exception(
                f"expected `metric_obj` to be one of `min`, `max`. Got {metric_obj}"
            )

        self.metric_minimize = (metric_obj == 'min')

        assert os.path.exists(save_dir), \
            f"Path to save checkpoints {save_dir}, doesn't exist"
        self.save_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

    def is_improvement(self, metric_val):
        if (self.metric_minimize and metric_val < self.best_metric) or (
            (not self.metric_minimize) and metric_val > self.best_metric
        ):
            return True
        else:
            return False
        
    def __save_checkpoint(self, chkpt: CheckpointData, type):
        chkpt_file = os.path.join(self.save_dir, f'{type}.pt')
        torch.save(chkpt, chkpt_file)

    def __load_checkpoint(self, type):
        chkpt_file = os.path.join(self.save_dir, f'{type}.pt')
        chkpt = torch.load(chkpt_file, weights_only=True)
        assert isinstance(chkpt, CheckpointData), \
            f"Checkpoint file @ {chkpt_file} doesn't load a `CheckpointData` object"
        return chkpt

    def save(self, round, server, clients, metrics_dict):
        '''Save current checkpoint and best checkpoint by comparing with past
        runs.
        '''
        curr_chkpt = CheckpointData(
            round = round,
            metric_name = self.metric_name,
            metric_val = metrics_dict[self.metric_name],
            client_model = clients[0].model.state_dict(),
            server_model = server.model.state_dict(),
            auxiliary_models = [
                c.auxiliary_model.state_dict() for c in clients
            ],
            client_optimizers=[
                c.optimizer.state_dict() for c in clients
            ],
            server_optimizer=server.optimizer.state_dict(),
            auxiliary_optimizers=[
                c.auxiliary_model.optimizer.state_dict() for c in clients
            ]
        )

        # save current
        self.__save_checkpoint(curr_chkpt, 'latest')

        # check and save best
        if self.is_improvement(metrics_dict[self.metric_name]):
            self.__save_checkpoint(curr_chkpt, 'best')
            self.best_metric = metrics_dict[self.metric_name]

    def load(self, clients, server, type='latest'):
        '''Load the checkpoint type specified by `type` and return the round,
        client, axuiliary and server models, and the client, server and server
        optimizers'''

        chkpt = self.__load_checkpoint(type)

        [c.model.load_state_dict(chkpt.client_model) for c in clients]
        [c.model.optimizer.load_state_dict(opt) for c, opt in
         zip(clients, chkpt.client_optimizers)]

        [c.auxiliary_model.load_state_dict(am) for c, am in
         zip(clients, chkpt.auxiliary_models)]
        [c.auxiliary_model.optimizer.load_state_dict(opt) for c, opt in
         zip(clients, chkpt.auxiliary_optimizers)]

        server.model.load_state_dict(chkpt.server_model)
        server.optimizer.load_state_dict(chkpt.server_optimizer)

        return chkpt.round, server, clients, {chkpt.metric_name: chkpt.metric_val}
def count_trainable_params(model: nn.Module):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_params

def __reduce_dict(module_dict, group_name):
    sub_dict = {}
    for k, v in module_dict.items():
        gn, rem = k.split('.', 1)
        if gn == group_name:
            sub_dict[rem] = v

    return sub_dict

def get_nested_dict(module_dict):
    params = {}
    for name, param in module_dict.items():
        if '.' not in name:
            params[name] = param
            continue

        group_name, _ = name.split('.', 1)
        if group_name in params.keys():
            continue
        params[group_name] = get_nested_dict(
            __reduce_dict(module_dict, group_name)
        )

    return params

def recursive_map(fn, nested_dict):
    res_dict = {}
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            res_dict[k] = recursive_map(fn, v)
        else:
            res_dict[k] = fn(v)
    return res_dict

OKGREEN = '\033[92m'
BOLD = '\033[1m'
ENDC = '\033[0m'

def pretty(d, color_cond_dict, indent=0, sep='|-- '):
    for (key, value), color_cond in zip(d.items(), color_cond_dict.values()):
        if isinstance(value, dict):
            logging.info(sep * indent + BOLD + str(key) + ENDC)
            pretty(value, color_cond, indent+1)
        else:
            if color_cond: print(OKGREEN, end='')
            logging.info(sep * indent + str(key) + '\t' + str(value))
            if color_cond: print(ENDC, end='')

def print_model(model):
    model_dict = get_nested_dict(dict(model.named_parameters()))
    pretty(
        recursive_map(lambda x: list(x.shape), model_dict),
        recursive_map(lambda x: x.requires_grad, model_dict) 
    )
