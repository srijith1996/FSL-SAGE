# -----------------------------------------------------------------------------
# Main script to call configured algorithms on configured dataset, model and
# configured hyperparameters.  Hydra is used for configurations and running
# experiments
# -----------------------------------------------------------------------------
import os, logging
import random
import numpy as np

import hydra
from omegaconf import DictConfig, open_dict
from functools import partial
from typing import List, Tuple, Callable
import torch
from torch.utils.data import DataLoader

from models import model_package, Client, Server
import datasets as dss
from utils import utils, logs
from algos import run_fl_algorithm

# -----------------------------------------------------------------------------
global_torch_device = None

# config torch backends and device 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
def get_dataloaders(cfg) -> Tuple[List[DataLoader], DataLoader]:

    trainSet, testSet = dss.get_dataset(cfg.dataset) 
    client_train_set, _ = dss.depart_dataset(
        cfg.num_clients, trainSet, testSet, cfg.dataset
    )

    dataset_size_list = [
        int(client_train_set[i]['num']) for i in range(cfg.num_clients)
    ]
    total = sum(dataset_size_list)
    logging.info("Dataset size for each client:")
    logging.info(dataset_size_list)
    logging.info(f"Total size: {total} samples.")

    if cfg.agg_factor == 'auto':
        cfg.agg_factor = [i / total for i in dataset_size_list]
    else:
        assert cfg.num_clients == len(cfg.agg_factor), \
            "# aggregation weights (agg_factor) is not equal to # clients."
    logging.info(f"Aggregation Factor: {cfg.agg_factor}")

    trainloaders = []
    for i in range(cfg.num_clients):
        train_set = client_train_set[i]["idxs"]
        trainloaders.append(
            DataLoader(
                dss.DatasetSplit(trainSet, train_set),
                batch_size=cfg.model.client.batch_size,
                shuffle=True, pin_memory=False
            )
        )
    
    testloader = DataLoader(
        testSet, batch_size=cfg.model.client.batch_size, shuffle=False,
        pin_memory=False
    )

    return trainloaders, testloader
    
# -----------------------------------------------------------------------------
def model_constructors(cfg) -> Callable:

    model_pack = model_package(cfg.model.name, cfg.model.auxiliary.name)

    __safe_copy_dict = lambda x : {} if x is None else x
    coptions = __safe_copy_dict(cfg.model.client.options)
    soptions = __safe_copy_dict(cfg.model.server.options)
    aoptions = __safe_copy_dict(cfg.model.auxiliary.options)

    client_constructor = partial(model_pack.client, **coptions)
    server_constructor = partial(model_pack.server, **soptions) \
        if model_pack.server is not None else (lambda: None)
    auxiliary_constructor = partial(
        model_pack.auxiliary, **aoptions
    ) if model_pack.auxiliary is not None else (lambda: None)

    return client_constructor, server_constructor, auxiliary_constructor

# -----------------------------------------------------------------------------
def setup_server_and_clients(
    cfg: DictConfig
) -> Tuple[Server, List[Client], DataLoader]:

    client_loaders, test_loader = get_dataloaders(cfg)

    client_constructor, server_constructor, auxiliary_constructor = \
        model_constructors(cfg)
        
    server = Server(
        server_constructor(), cfg.model.server, device=global_torch_device
    )
    server.model.to(global_torch_device)

    client_list = [
        Client(i, client_loaders[i],
            client_constructor(),
            auxiliary_constructor(server=server, device=global_torch_device),
            cfg.model.client, device=global_torch_device
        )
        for i in range(cfg.num_clients)
    ]

    for c in client_list:
        c.model.load_state_dict(client_list[0].model.state_dict())
        c.model.to(global_torch_device)

        if c.auxiliary_model is not None:
            c.auxiliary_model.load_state_dict(
                client_list[0].auxiliary_model.state_dict()
            )
            c.auxiliary_model.to(global_torch_device)

    return server, client_list, test_loader

# -----------------------------------------------------------------------------
@hydra.main(
    config_path='hydra_config', config_name='config', version_base='1.3'
)
def main(cfg: DictConfig):

    # setup logging
    with open_dict(cfg):
        if cfg.save:
            utils.create_save_dir(cfg)
            log_file = os.path.join(cfg.save_path, "output.log")
        logs.configure_logging(log_file if cfg.save else None)
        if cfg.save: logs.log_hparams(cfg)

        # fix random seed
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        global global_torch_device
        global_torch_device = torch.device(cfg.device)

    # configure and run FL algorithm
    server, client_list, test_loader = setup_server_and_clients(cfg)
    run_fl_algorithm(
        cfg, server, client_list, test_loader, global_torch_device
    )

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------