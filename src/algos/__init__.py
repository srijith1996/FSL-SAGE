# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Dict, List, Callable
import time
import os, importlib
import logging, copy
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

        self.agg_factor = agg_factor
        self.device = device
        self.use_64bit = use_64bit

    @abstractmethod
    def full_model(self):
        pass
    
    @abstractmethod
    def client_step(self, x, y):
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
        ret_dict = dict()
        t0 = time.time()
        self.aggregated_client = aggregate_models(
            [c.model for c in self.clients], self.agg_factor, self.device
        )
        agg_weights = self.aggregated_client.state_dict()

        for c in self.clients:
            c.model.load_state_dict(agg_weights)
            self.comm_load += 2 * calculate_load(self.aggregated_client)

        ret_dict['client_agg_compute_time'] = time.time() - t0
        return ret_dict

    def aggregate(self):
        return self.aggregate_clients()

    def evaluate(self):
        test_correct = 0
        test_loss = []
        self.eval_mode()
        with torch.no_grad():
            for x, y in tqdm(
                self.test_loader, desc="Test Batch", unit='batch', leave=False
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
    server              : Server
    client_list         : List[Client]
    loss                : List[float]
    accuracy            : List[float]
    train_metrics       : List[Dict[str, List]]
    aggregation_metrics : Dict[str, List]
    comm_load           : List[float]
    avg_compute_times   : Dict[str, float]

# ------------------------------------------------------------------------------
def __aggregate_metrics_dict(
    client_metrics_dict, reduction=np.mean, keys='', excl_keys=None
):
    '''Given a list of dicts, each dict comprising of metric key-value pairs,
    where each value is a list of metrics values, one for each round of FL,
    compute the reduction of the values specified by `reduction` for keys
    containing the substring `keys` and not containing the substring
    `excl_keys`, and create a new dictionary that transforms a list of key ->
    value mappings to a dictionary of key -> value-list.
    '''
    def key_condition(k, inc_k, exc_k):
        if exc_k:
            return (inc_k in k and exc_k not in k)
        else:
            return (inc_k in k)

    agg_metrics_dict = {
        k: [reduction(v)] for k, v in client_metrics_dict[0].items()
        if key_condition(k, keys, excl_keys)
    }
    for met_dict in client_metrics_dict[1:]:
        [agg_metrics_dict[k].append(reduction(v)) for k, v in met_dict.items()
         if key_condition(k, keys, excl_keys)]

    return agg_metrics_dict

# ------------------------------------------------------------------------------
def __print_aggregate_metrics(
    client_metrics_dict, aggregation_metrics, prefix='tr.'
):
    agg_metrics_dict = __aggregate_metrics_dict(
        client_metrics_dict, reduction=lambda x: x[-1]
    )
    print_str = ''
    for i, (k, v) in enumerate(agg_metrics_dict.items()):
        print_str += f'avg {prefix} {k}: {np.mean(v):.2f}' if i == 0 \
            else f', avg {prefix} {k}: {np.mean(v):.2f}'

    for k, v in aggregation_metrics.items():
        print_str += f', {prefix} {k}: {v[-1]:.2f}'

    return print_str

# ------------------------------------------------------------------------------
def __add_aggregated_compute_times(client_metrics_dict, aggregation_metrics):
    # reduce and pack model_compute_times
    agg_metrics_dict = __aggregate_metrics_dict(
        client_metrics_dict, reduction=np.sum, keys='model_compute_time'
    )
    # update keys with agg_compute_times
    agg_metrics_dict.update({k: np.sum(v) for k, v in aggregation_metrics.items()})
    new_compute_times = {k: np.mean(v) for k, v in agg_metrics_dict.items()}
    return new_compute_times

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
    lr_dict = {f"client_{i}_lr" : take_lr_step(alg.clients[i])}
    log_dict = {}
    log_dict[f'Clients/client_{i}/cl_model_lr'] = lr_dict[f'client_{i}_lr']

    # for server model(s) is multi_server
    if alg_name != 'fed_avg':
        if alg_name == 'sl_multi_server':
            lr_dict[f'server_{i}_lr'] = \
                take_lr_step(alg.servers[i])
            log_dict[f'Server/server_{i}/server_lr'] = \
                lr_dict[f'server_{i}_lr']

    # for auxiliary models. For fsl_sage, the optimization happens within the
    # align() method.
    if alg_name == 'cse_fsl' or alg_name == 'fsl_sage':
        lr_dict['aux_lr'] = take_lr_step(alg.clients[i].auxiliary_model)

    # log auxiliary model learning rate for fsl algorithms
    if alg_name == 'cse_fsl' or alg_name =='fsl_sage':
        log_dict.update({
            f'Clients/client_{i}/aux_model/aux_model_lr': \
                alg.clients[i].auxiliary_model.optimizer.param_groups[0]['lr']
        })

    return lr_dict, log_dict

# ------------------------------------------------------------------------------
def _run_fl_algorithm(
    cfg:DictConfig,
    server: Server,
    clients: List[Client],
    test_loader: DataLoader,
    checkpointer: Checkpointer,
    torch_device,
    logger_fn: Callable,
    test_loss=None,
    test_acc=None,
    train_metrics=None,
    aggregation_metrics=None,
    comm_load=None,
) -> FLResults:

    # get algorithm
    alg = ALGORITHM_REGISTRY[cfg.algorithm.name](
        cfg.algorithm, server, clients, test_loader,
        cfg.agg_factor, 
        device=torch_device, use_64bit=cfg.use_64bit
    )

    with open_dict(cfg):
        if cfg.comm_threshold_mb is None:
            cfg.comm_threshold_mb = np.inf

    if comm_load is None: comm_load = []
    if test_loss is None: test_loss = []
    if test_acc is None: test_acc = []
    if train_metrics is None:
        train_metrics = [{} for _ in range(cfg.num_clients)]
    if aggregation_metrics is None:
        aggregation_metrics = {}
    
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
                            for k, (x, y) in enumerate(pbar_local):
                                x = x.to(torch_device).double() \
                                    if cfg.use_64bit else x.to(torch_device).float()
                                y = y.to(torch_device).long()

                                tr_metrics = alg.client_step(
                                    (t, i, j, k), x, y
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
                    tr_mets = {k: np.mean(v) if 'model_compute_time' not in k
                               else np.sum(v) for k, v in tr_mets.items()}
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

            # aggregate required models
            agg_metrics = alg.aggregate()
            comm_load.append(alg.comm_load)
            if t == 0:
                for k, v in agg_metrics.items():
                    aggregation_metrics[k] = [v]
            else:
                [aggregation_metrics[k].append(v) for k, v in
                agg_metrics.items()]

            tr_str = __print_aggregate_metrics(
                train_metrics, aggregation_metrics, prefix='tr'
            )

            # set models to eval mode and evaluate
            alg.eval_mode()
            acc_, loss_ = alg.evaluate()
            test_acc.append(acc_)
            test_loss.append(loss_)
            log_dict.update({
                'Test/accuracy': acc_,
                'Test/loss': loss_,
                'Test/load': comm_load
            })

            # adjust learning rates for server models in single server runs
            # i.e., sl_single_server, cse_fsl and fsl_sage
            _, log_dict_ = log_and_step_lr_per_round(
                alg, cfg.algorithm.name
            )
            log_dict.update(log_dict_)

            logging.info(
                f' > Round {t}, ' + tr_str +
                f', ts. loss: {loss_:.2f}, ts. acc: {100. * acc_:.2f}%' +
                f', comm: {(alg.comm_load / (1024**3)):.2f} GiB.',
            )
            logger_fn(log_dict, step=t)

            # save checkpoints
            if cfg.save and t % cfg.checkpoint_interval == 0:
                checkpointer.save(
                    t, alg.server, alg.clients, {'accuracy': acc_}
                )

            # stop if communication load exceeds threshold
            if alg.comm_load / (1024**2) >= cfg.comm_threshold_mb:
                logging.info(f"Communication budget reached/exceeded @ {t:d} rounds!")
                break

    avg_compute_times = __add_aggregated_compute_times(
        train_metrics, aggregation_metrics
    )

    return FLResults(
        alg.server, alg.clients, test_loss, test_acc, train_metrics,
        aggregation_metrics, comm_load, avg_compute_times
    )

# ------------------------------------------------------------------------------
def run_fl_algorithm(
    cfg:DictConfig,
    server: Server,
    clients: List[Client],
    test_loader: DataLoader,
    checkpointer: Checkpointer,
    torch_device,
    logger_fn: Callable,
    warm_start=False
):

    if cfg.algorithm.name == 'fsl_sage' and warm_start:
        ws_cfg = copy.deepcopy(cfg)
        with open_dict(ws_cfg):
            ws_cfg.rounds = 1
            ws_cfg.algorithm.name = 'cse_fsl'
            if 'server_update_interval' not in ws_cfg.algorithm.keys():
                ws_cfg.algorithm.server_update_interval = 5

        logging.info("Warm-starting auxiliary model with CSE-FSL")
        results = _run_fl_algorithm(
            ws_cfg, server, clients, test_loader, checkpointer, torch_device,
            logger_fn
        )
        server = results.server
        clients = results.client_list
        test_loss = results.loss
        test_acc = results.accuracy
        train_metrics = results.train_metrics
        aggregation_metrics = results.aggregation_metrics
        comm_load = results.comm_load
    else:
        test_loss = None
        test_acc = None
        train_metrics = None
        aggregation_metrics = None
        comm_load = None

    return _run_fl_algorithm(
        cfg, server, clients, test_loader, checkpointer, torch_device,
        logger_fn, test_loss, test_acc, train_metrics, aggregation_metrics,
        comm_load
    )

# ------------------------------------------------------------------------------
