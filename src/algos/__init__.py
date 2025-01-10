# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Dict, List
import os, importlib
import logging, copy
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from models import Server, Client
from utils.utils import calculate_load
from utils import metrics as met_utils

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
    def client_step(self, x, y, *args):
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
    comm_load   : List[float]
    metric_lists : Dict[str, List]

# ------------------------------------------------------------------------------
def prepare_batch(data, torch_device, task, use_64bit=False):
    for i, d in enumerate(data):
        data[i] = d.to(torch_device)

    x, y = data[0], data[1]
    if torch.is_floating_point(x):
        x = x.double() if use_64bit else x.float()
    y = y.long() if task == 'classification' else y
    args = list(data)[2:] if len(data) > 2 else []

    return x, y, args

# ------------------------------------------------------------------------------
def evaluate(
    model, test_loader, metric_fns=met_utils.METRIC_FUNCTION_DICT,
    problem_type='image_classification', use_64bit=False, device='cpu'
):
    metrics = {}
    for v in metric_fns.values(): v.reset()
    with torch.no_grad():
        for data in test_loader:
            x, y, args = prepare_batch(
                data, device, problem_type, use_64bit
            )
            out = model(x)
            for v in metric_fns.values():
                v.update(out, y)

        for k, v in metric_fns.items(): metrics[k] = v.average()
    return metrics

# ------------------------------------------------------------------------------
def run_fl_algorithm(
    cfg: DictConfig,
    server: Server,
    clients: List[Client],
    test_loader: DataLoader,
    torch_device
) -> FLResults:

    # get algorithm
    if cfg.algorithm.name == 'fed_avg':
        kwargs = {'optimizer_options': cfg.model.client.optimizer}
    else:
        kwargs = {}

    alg = ALGORITHM_REGISTRY[cfg.algorithm.name](
        cfg.algorithm, server, clients, test_loader,
        cfg.agg_factor, cfg.model.client.lr, cfg.model.server.lr,
        device=torch_device, use_64bit=cfg.use_64bit, **kwargs
    )

    comm_load = []
    metric_lists = {k: [] for k in cfg.metric_names}
    metric_fns = {
        k: met_utils.METRIC_FUNCTION_DICT[k]
        for k in cfg.metric_names if k != 'loss'
    }
    if 'loss' in cfg.metric_names:
        metric_fns['loss'] = met_utils.LossMetric(server.criterion)

    # main loop
    with logging_redirect_tqdm():
        for t in tqdm(
            range(cfg.rounds), unit="rd", desc="Round", leave=False,
            colour='green'
        ):
            # first set models to train mode; then take client steps
            server.model.train()
            for i in tqdm(
                range(cfg.num_clients), unit="cl", desc="Client", leave=False
            ):
                clients[i].model.train()
                for j in tqdm(
                    range(clients[i].epochs), unit="ep", desc="Local epoch",
                    leave=False
                ):
                    for k, data in enumerate(
                        tqdm(clients[i].train_loader, unit="batch",
                        desc="Local batch", leave=False)
                    ):
                        x, y, args = prepare_batch(
                            data, torch_device, cfg.dataset.problem_type, cfg.use_64bit
                        )
                        alg.client_step((t, i, j, k), x, y, *args)

            alg.aggregate()
            comm_load.append(alg.comm_load)

            # set models to eval mode and evaluate
            server.model.eval()
            for i in range(cfg.num_clients): clients[i].model.eval()

            metrics = evaluate(
                alg.full_model, test_loader, metric_fns=metric_fns,
                problem_type=cfg.dataset.problem_type, use_64bit=cfg.use_64bit,
                device=torch_device
            )

            for k in metric_lists.keys():
                metric_lists[k].append(metrics[k])

            print_str = f' > Round {t}, Test '
            for i, (k, v) in enumerate(metrics.items()):
                if i == len(metrics.keys()) - 1:
                    print_str += f'{k}: {v:.4f}.'
                else:
                    print_str += f'{k}: {v:.4f}, '

            logging.info(print_str)

    return FLResults(alg.server, alg.clients, comm_load, metric_lists)

# ------------------------------------------------------------------------------
