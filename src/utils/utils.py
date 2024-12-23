#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.7
#import random
import logging
import os, pathlib
from datetime import datetime
import torch

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
    for i, m in enumerate(models):
        for p in m.parameters():
            sizes[i] += p.nelement() * p.element_size()
        for p in m.buffers():
            sizes[i] += p.nelement() * p.element_size()
        sizes[i] = sizes[i] / (1024 ** 2)

    return sizes

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
        cfg.save_path = p

        pathlib.Path(p).mkdir(exist_ok=False)
        logging.info(f"\033[1;36m[NOTICE] Directory '{p}' build.\033[0m")

        # add a plot path in case required
        cfg['plot_path'] = os.path.join(cfg['save_path'], 'plots')
        os.makedirs(cfg['plot_path'], exist_ok=True)
