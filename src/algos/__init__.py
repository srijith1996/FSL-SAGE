# ------------------------------------------------------------------------------
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Callable
import os, importlib
import logging, copy
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from dataclasses import dataclass
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import torch
import torch.nn as nn

from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from models import Server, Client
from utils.utils import calculate_load, Checkpointer
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
        device: str = 'cpu', use_64bit: bool = False
    ):
        self.cfg = cfg
        self.server = server
        self.clients = clients
        self.test_loader = test_loader
        self.criterion = server.criterion
        self.server_updated = True          # default for FL and basic SL algs

        self.agg_factor = agg_factor
        self.device = device
        self.use_64bit = use_64bit

        # init aggregated client
        self.aggregated_client = copy.deepcopy(clients[0].model)

    def full_model(self, x):
        cl_out = self.client_model()(x)
        if self.server_model() is None:
            return cl_out
        return self.server_model()(*cl_out) if isinstance(cl_out, tuple) \
            else self.server_model()(cl_out)

    @abstractmethod
    def client_model(self):
        pass

    @abstractmethod
    def server_model(self):
        pass
    
    @abstractmethod
    def client_step(self, x, y, *args):
        pass

    def special_models_train_mode(self, t):
        pass

    def special_models_eval_mode(self):
        pass

    def train_mode(self, t):
        for c in self.clients: c.model.train()
        self.server.model.train()

        # the condition is required because the aggregate models are not defined
        # at the zero'th round
        if t > 0: self.aggregated_client.train()
        self.special_models_train_mode(t)

    def eval_mode(self):
        for c in self.clients: c.model.eval()
        self.server.model.eval()
        self.aggregated_client.eval()
        self.special_models_eval_mode()

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

# ------------------------------------------------------------------------------
def get_auxlist_memory_consumption(aux_list):
    size = 0        # in bytes
    for el in aux_list:
        if isinstance(el, torch.Tensor):
            size += el.numel() * el.element_size()
        elif isinstance(el, torch.Size):
            size += len(el) * 8
        elif isinstance(el, (list, tuple)):
            size += get_auxlist_memory_consumption(el)
        elif el is None:
            size += 0
        else:
            print(f"Warning dont know how to compute size of {type(el)}")
            size += sys.getsizeof(el)
    return size

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
    train_metrics : Dict[str, List]
    comm_load   : List[float]
    metric_lists : Dict[str, List]

# ------------------------------------------------------------------------------
def prepare_batch(data, torch_device, task, use_64bit=False):
    for i, d in enumerate(data):
        data[i] = d.to(torch_device)

    x, y = data[0], data[1]
    if torch.is_floating_point(x):
        x = x.double() if use_64bit else x.float()
    y = y.long() if task == 'image_classification' else y
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
        for data in tqdm(
            test_loader, desc="Test Batch", unit='batch', leave=False
        ):
            x, y, args = prepare_batch(
                data, device, problem_type, use_64bit
            )
            out = model(x)
            out = out[0] if isinstance(out, tuple) else out
            for v in metric_fns.values():
                v.update(out, y) if len(args) == 0 else v.update(out, y, *args)

        for k, v in metric_fns.items(): metrics[k] = v.average()
    return metrics

# ------------------------------------------------------------------------------
def __print_aggregate_metrics(client_metrics_dict, prefix='tr.'):
    agg_metrics_dict = {
        k: [np.mean(v)] for k, v in client_metrics_dict[0].items()
    }
    for met_dict in client_metrics_dict[1:]:
        [agg_metrics_dict[k].append(np.mean(v)) for k, v in met_dict.items()]

    print_str = ''
    for i, (k, v) in enumerate(agg_metrics_dict.items()):
        print_str += f'{prefix} {k}: {np.mean(v):.2f}' if i == 0 \
            else f', {prefix} {k}: {np.mean(v):.2f}'

    return print_str

# ------------------------------------------------------------------------------
def take_lr_step(obj):

    if obj.lr_scheduler:
        obj.lr_scheduler.step()
        lr = obj.lr_scheduler.get_last_lr()[0]
    else:
        lr = obj.optimizer.param_groups[0]['lr']
    return lr

# ------------------------------------------------------------------------------
def log_and_step_lr_per_round(alg, alg_name):
    lr_dict = {}
    log_dict = {}
    if alg_name not in ['fed_avg', 'sl_multi_server']:
        lr_dict['server_lr'] = take_lr_step(alg.server)
        log_dict[f'Server/server_lr'] = lr_dict['server_lr']

    return lr_dict, log_dict

# ------------------------------------------------------------------------------
def log_and_step_lr_per_client(i, alg, alg_name):

    # for client model
    lr_dict = {"client_lr" : take_lr_step(alg.clients[i])}
    log_dict = {}
    log_dict[f'Clients/client_{i}/cl_model_lr'] = lr_dict['client_lr']

    # for server model(s) is multi_server
    if alg_name != 'fed_avg':
        if alg_name == 'sl_multi_server':
            lr_dict[f'server_{i}_lr'] = \
                take_lr_step(alg.servers[i])
            log_dict[f'Server/server_{i}/server_lr'] = \
                lr_dict[f'server_{i}_lr']

    # for auxiliary models. For fsl_sage, the optimization happens within the
    # align() method.
    if alg_name == 'cse_fsl':
        lr_dict['aux_lr'] = take_lr_step(alg.clients[i].auxiliary_model)

    # log auxiliary model learning rate for fsl algorithms
    if alg_name == 'cse_fsl' or \
        alg_name =='fsl_sage':
        log_dict.update({
            f'Clients/client_{i}/aux_model/aux_model_lr': \
                alg.clients[i].auxiliary_model.optimizer.param_groups[0]['lr']
        })

    return lr_dict, log_dict

# ------------------------------------------------------------------------------
def run_fl_algorithm(
    cfg: DictConfig,
    server: Server,
    clients: List[Client],
    test_loader: DataLoader,
    checkpointer: Checkpointer,
    torch_device,
    logger_fn: Callable
) -> FLResults:

    # get algorithm
    if cfg.algorithm.name == 'fed_avg':
        kwargs = {'optimizer_options': cfg.model.client.optimizer}
    else:
        kwargs = {}

    alg = ALGORITHM_REGISTRY[cfg.algorithm.name](
        cfg.algorithm, server, clients, test_loader,
        cfg.agg_factor, 
        device=torch_device, use_64bit=cfg.use_64bit
    )

    with open_dict(cfg):
        if cfg.comm_threshold_mb is None:
            cfg.comm_threshold_mb = np.inf

    comm_load = []
    metric_lists = {k: [] for k in cfg.dataset.problem.metric_names}
    metric_fns = {
        k: met_utils.METRIC_FUNCTION_DICT[k]
        for k in cfg.dataset.problem.metric_names if k != 'loss'
    }
    if 'loss' in cfg.dataset.problem.metric_names:
        metric_fns['loss'] = met_utils.LossMetric(server.criterion)

    train_metrics = [{} for _ in range(cfg.num_clients)]
    
    # main loop
    with logging_redirect_tqdm():
        for t in tqdm(
            range(cfg.rounds), unit="rd", desc="Round", leave=False,
            colour='green'
        ):
            log_dict = {}

            # set all models to train mode and train
            alg.train_mode(t)
            with tqdm(
                range(cfg.num_clients), unit="cl", desc="Client", leave=False,
                colour='blue'
            ) as pbar:
                for i in pbar:
                    tr_mets = {}
                    for j in tqdm(
                        range(clients[i].epochs), unit="ep", desc="Local epoch",
                        leave=False
                    ):
                        with tqdm(
                            clients[i].train_loader, unit="batch",
                            desc="Local batch", leave=False
                        ) as pbar_local:
                            for k, data in enumerate(pbar_local):
                                x, y, args = prepare_batch(
                                    data, torch_device, cfg.use_64bit
                                )

                                # take client step
                                tr_metrics = alg.client_step(
                                    (t, i, j, k), x, y, *args
                                )

                                pbar_local.set_postfix(**tr_metrics)
                                if j == 0 and k == 0:
                                    tr_mets = {
                                        k: [v] for k, v in tr_metrics.items()
                                    }
                                else:
                                    [tr_mets[k].append(v) for k, v in
                                    tr_metrics.items()]

                    # compute mean of metrics
                    tr_mets = {k: np.mean(v) for k, v in tr_mets.items()}
                    if t == 0:
                        train_metrics[i] = {
                            k: [v] for k, v in tr_mets.items()
                        }
                    else:
                        [train_metrics[i][k].append(v) for k, v in
                        tr_mets.items()]
                    log_dict.update({
                        f'Clients/client_{i}/{k}': v for k, v in tr_mets.items()
                    })

                    # adjust learning rate based on algorithm
                    lr_dict, log_dict_ = log_and_step_lr_per_client(
                        i, alg, cfg.algorithm.name
                    )
                    log_dict.update(log_dict_)
                    pbar.set_postfix(**tr_mets, **lr_dict)

            # set models to eval mode and evaluate
            alg.aggregate()
            comm_load.append(alg.comm_load)

            tr_str = __print_aggregate_metrics(train_metrics)

            # set models to eval mode and evaluate
            alg.eval_mode()
            metrics = evaluate(
                alg.full_model, test_loader, metric_fns=metric_fns,
                problem_type=cfg.dataset.problem.name, 
                use_64bit=cfg.use_64bit, device=torch_device
            )

            for k in metric_lists.keys():
                metric_lists[k].append(metrics[k])

            log_dict.update(
                {f'Test/{k}': v for k, v in metric_lists.items()}
            )

            # adjust learning rates for server models in single server runs
            # i.e., sl_single_server, cse_fsl and fsl_sage
            _, log_dict_ = log_and_step_lr_per_round(
                alg, cfg.algorithm.name
            )
            log_dict.update(log_dict_)

            # log to wandb, console and log file
            print_str = f' > Round {t}, Test '
            for i, (k, v) in enumerate(metrics.items()):
                if i == len(metrics.keys()) - 1:
                    print_str += f'{k}: {v:.2g}'
                else:
                    print_str += f'{k}: {v:.2g}, '
            logging.info(print_str)
            logger_fn(log_dict, step=t)

            # save checkpoints
            if cfg.save and t % cfg.checkpoint_interval == 0:
                checkpointer.save(
                    t, alg.server, alg.clients,
                    {'accuracy': metrics['accuracy']}
                )

            # stop if communication load exceeds threshold
            if alg.comm_load / (1024**2) >= cfg.comm_threshold_mb:
                logging.info(
                    f"Communication budget reached/exceeded @ {t:d} rounds!"
                )
                break

    return FLResults(
        alg.server, alg.clients, train_metrics, comm_load, metric_lists
    )

# ------------------------------------------------------------------------------
