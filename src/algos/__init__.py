# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Dict, List
import os, importlib
import logging, copy
from dataclasses import dataclass
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import torch
import torch.nn as nn

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from models import Server, Client
from utils.utils import calculate_load, Checkpointer

# ------------------------------------------------------------------------------
def aggregate_models(model_list, weights, device='cpu'):
    '''Aggregate Models
    
    Aggregate a given list of models using weights supplied in `weights`.

    Params
    ------
    model_list - List of pytorch models
    weights - List of weights used in averaging
    device - Device to use for computations

    Returns
    -------
    aggregated - Torch model containing the aggregated weights.
    '''

    assert len(weights) == len(model_list),\
        "Length of model_list and weights is different."

    assert sum(weights) == 1, "Sum of weights should be 1"

    aggregated = copy.deepcopy(model_list[0])
    aggregated_weights = aggregated.state_dict()

    for key in aggregated_weights:
        aggregated_weights[key] = model_list[0].state_dict()[key] * weights[0]

    for i in range(1, len(model_list)):
        for key in aggregated_weights:
            aggregated_weights[key] += model_list[i].state_dict()[key] * weights[i]

    aggregated.to(device)
    aggregated.load_state_dict(aggregated_weights)

    # Won't add to the comm load here because I'll add it for when algs load
    # in the agg model
    return aggregated

# ------------------------------------------------------------------------------
class FLAlgorithm(ABC):
    aggregated_client   : nn.Module
    comm_load           : float = 0.0
    loss                : float = np.inf
    acc                 : float = 0.0

    def __init__(self,
        cfg: DictConfig, server: Server, clients: List[Client],
        test_loader: DataLoader, agg_factor: List[float],
        client_lr: float, server_lr: float,
        device: str = 'cpu', use_64bit: bool = False
    ):
        self.cfg = cfg
        self.server = server
        self.clients = clients
        self.test_loader = test_loader
        self.criterion = server.criterion

        self.client_lr = client_lr
        self.server_lr = server_lr

        self.agg_factor = agg_factor
        self.device = device
        self.use_64bit = use_64bit

    @abstractmethod
    def full_model(self):
        pass
    
    @abstractmethod
    def client_step(self, x, y):
        pass

    def aggregate_clients(self):
        self.aggregated_client = aggregate_models(
            [c.model for c in self.clients], self.agg_factor, self.device
        )
        agg_weights = self.aggregated_client.state_dict()

        for c in self.clients:
            c.model.load_state_dict(agg_weights)
            self.comm_load += 2 * calculate_load(self.aggregated_client)

    def aggregate(self):
        return self.aggregate_clients()

    def evaluate(self):
        test_correct = 0
        test_loss = []
        with torch.no_grad():
            for x, y in tqdm(
                self.test_loader, desc="Batch", unit='batch', leave=False
            ):
                x = x.to(self.device).double() if self.use_64bit \
                    else x.to(self.device).float()
                y = y.to(self.device).long()
                out = self.full_model(x)
                batch_loss = self.criterion(out, y)
                test_loss.append(batch_loss.item())
                _, predicted = torch.max(out.data, 1)
                test_correct += predicted.eq(y.view_as(predicted)).sum().item()
            loss = sum(test_loss) / len(test_loss)
            acc =  test_correct / len(self.test_loader.dataset)

        return acc, loss

# ------------------------------------------------------------------------------
# maps alg_name: str -> instance of FLAlgorithm
ALGORITHM_REGISTRY: Dict[str, FLAlgorithm] = dict()

# ------------------------------------------------------------------------------
def register_algorithm(name):
    """Decorator to register a new alogrithm"""
    def register_alg_cls(cls):
        if name in ALGORITHM_REGISTRY:
            raise ValueError(
                'Cannot register duplicate algorithm {}'.format(name)
            )
        if not issubclass(cls, FLAlgorithm):
            raise ValueError(
                'Model {} must extend {}'.format(name, cls.__name__)
            )
        ALGORITHM_REGISTRY[name] = cls
        return cls
    return register_alg_cls

# -----------------------------------------------------------------------------
# Automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file[0].isalpha():
        module = file[:file.find('.py')]
        importlib.import_module('algos.' + module)

# ------------------------------------------------------------------------------
@dataclass
class FLResults():
    server      : Server   
    client_list : List[Client]
    loss        : List[float]
    accuracy    : List[float]
    comm_load   : List[float]

# ------------------------------------------------------------------------------
def run_fl_algorithm(
    cfg: DictConfig,
    server: Server,
    clients: List[Client],
    test_loader: DataLoader,
    checkpointer: Checkpointer,
    torch_device
) -> FLResults:

    # get algorithm
    alg = ALGORITHM_REGISTRY[cfg.algorithm.name](
        cfg.algorithm, server, clients, test_loader,
        cfg.agg_factor, cfg.model.client.lr, cfg.model.server.lr,
        device=torch_device, use_64bit=cfg.use_64bit
    )

    comm_load = []
    loss = []
    acc = []
    
    # main loop
    with logging_redirect_tqdm():
        for t in tqdm(
            range(cfg.rounds), unit="rd", desc="Round", leave=False,
            colour='green'
        ):
            for i in tqdm(
                range(cfg.num_clients), unit="cl", desc="Client", leave=False
            ):
                for j in tqdm(
                    range(clients[i].epochs), unit="ep", desc="Local epoch",
                    leave=False
                ):
                    for k, (x, y) in enumerate(
                        tqdm(clients[i].train_loader, unit="batch",
                        desc="Local batch", leave=False)
                    ):
                        x = x.to(torch_device).double() \
                            if cfg.use_64bit else x.to(torch_device).float()
                        y = y.to(torch_device).long()
                        alg.client_step((t, i, j, k), x, y)

            alg.aggregate()
            comm_load.append(alg.comm_load)

            acc_, loss_ = alg.evaluate()
            acc.append(acc_)
            loss.append(loss_)

            logging.info(f' > Round {t}, testing loss: {loss_:.2f}, testing acc: {100. * acc_:.2f}%')

            # save checkpoints
            if t % cfg.checkpoint_interval == 0:
                checkpointer.save(
                    t, alg.server, alg.clients, {'accuracy': acc_}
                )

    return FLResults(alg.server, alg.clients, loss, acc, comm_load)

# ------------------------------------------------------------------------------
