# -----------------------------------------------------------------------------
# Main script to call configured algorithms on configured dataset, model and
# configured hyperparameters.  Hydra is used for configurations and running
# experiments
# -----------------------------------------------------------------------------
import os, logging, json
import random
import numpy as np
import wandb
from datetime import datetime

import hydra
from omegaconf import DictConfig, open_dict
from functools import partial
from typing import List, Tuple, Callable
import torch
from torch.utils.data import DataLoader
import loralib

from models import model_package, Client, Server
import datasets as dss
from utils import utils, logs
from algos import run_fl_algorithm
from omegaconf import OmegaConf

# -----------------------------------------------------------------------------
global_torch_device = None

# config torch backends and device 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
def get_dataloaders(cfg) -> Tuple[List[DataLoader], DataLoader]:

    trainSet, testSet = dss.get_dataset(
        cfg.dataset, batch_size=cfg.model.client.batch_size
    )
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
                batch_size=cfg.model.client.batch_size, shuffle=True,
                num_workers=cfg.num_workers, pin_memory=True
            )
        )
    
    testloader = DataLoader(
        testSet, batch_size=cfg.model.client.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    return trainloaders, testloader
    
# -----------------------------------------------------------------------------
def __safe_copy_dict_key(d, key):
    if key in d:
        return d[key] if d[key] is not None else {}
    else:
        return {}

# -----------------------------------------------------------------------------
def check_and_mark_lora_trainable(cfg, server, clients):
    options = __safe_copy_dict_key(cfg.model, 'options')
    if (not "lora_attn_dim" in options) or options['lora_attn_dim'] <= 0:
        return

    loralib.mark_only_lora_as_trainable(server.model)
    for c in clients: loralib.mark_only_lora_as_trainable(c.model)

    logging.info("After marking LoRA only as trainable:-")
    tr_p, all_p = utils.count_trainable_params(clients[0].model)
    logging.info(
        f"--> Client: # trainable params: {tr_p}, " + 
        f"Total # params: {all_p}, " + 
        f"Fraction trainable: {(tr_p/all_p * 100.0):.2f}%"
    )
    tr_p, all_p = utils.count_trainable_params(server.model)
    logging.info(
        f"--> Server: # trainable params: {tr_p}, " +
        f"Total # params: {all_p}, " +
        f"Fraction trainable: {(tr_p/all_p * 100.0):.2f}%"
    )
        
# -----------------------------------------------------------------------------
def model_constructors(cfg) -> Callable:

    model_pack = model_package(cfg.model.name, cfg.model.auxiliary.name)

    coptions = __safe_copy_dict_key(cfg.model.client, 'options')
    soptions = __safe_copy_dict_key(cfg.model.server, 'options')
    common_options = __safe_copy_dict_key(cfg.model, 'options')
    aoptions = __safe_copy_dict_key(cfg.model.auxiliary, 'options')

    client_constructor = partial(
        model_pack.client, **coptions, **common_options
    )
    server_constructor = partial(
        model_pack.server, **soptions, **common_options
    ) if model_pack.server is not None else (lambda: None)
    auxiliary_constructor = partial(
        model_pack.auxiliary, **aoptions, **common_options
    ) if model_pack.auxiliary is not None else (lambda: None)

    return client_constructor, server_constructor, auxiliary_constructor, \
        model_pack.client_to_server_params

# -----------------------------------------------------------------------------
def setup_server_and_clients(
    cfg: DictConfig
) -> Tuple[Server, List[Client], DataLoader]:

    client_loaders, test_loader = get_dataloaders(cfg)

    client_constructor, server_constructor, auxiliary_constructor, \
        client_to_server_params = model_constructors(cfg)
        
    # init client objects
    client_list = [
        Client(i, client_loaders[i],
            client_constructor(),
            cfg.model.client, cfg.rounds, device=global_torch_device
        )
        for i in range(cfg.num_clients)
    ]

    client_server_params = client_to_server_params(client_list[0].model) \
        if client_to_server_params is not None else dict()

    # init server object 
    server = Server(
        server_constructor(**client_server_params), cfg.model.server,
        device=global_torch_device, problem_type=cfg.dataset.problem.name
    )
    server.model.to(global_torch_device)

    # initialize auxiliary models
    for c in client_list:
        if auxiliary_constructor:
            c.init_auxiliary(
                auxiliary_constructor(
                    server=server, device=global_torch_device,
                    **client_server_params
                ),
                cfg.model.auxiliary
            )

    # load pretrained_weights
    def __safe_load_weights(model, *args, **kwargs):
        if not 'pretrained_weights_file' in cfg.model:
            return

        saved_state_dict = torch.load(
            cfg.model.pretrained_weights_file, weights_only=True
        )

        if hasattr(model, 'load_weight') and callable(model.load_weight):
            model.load_weight(saved_state_dict, *args, **kwargs)
        else:
            model.load_state_dict(saved_state_dict, strict=False)

        logging.info(f"Successfully loaded pretrained weights: {cfg.model.pretrained_weights_file}")

    __safe_load_weights(client_list[0].model)
    client_server_params = client_to_server_params(client_list[0].model) \
        if client_to_server_params is not None else dict()

    __safe_load_weights(server.model, **client_server_params)

    # Copy first client model's and aux model's weights to all clients
    for c in client_list:
        c.model.load_state_dict(client_list[0].model.state_dict())
        c.model.to(global_torch_device)

        if c.auxiliary_model is not None:
            c.auxiliary_model.load_state_dict(
                client_list[0].auxiliary_model.state_dict()
            )
            c.auxiliary_model.to(global_torch_device)

    # compute sizes of different models
    mod_sizes, param_counts = utils.compute_model_size(
        client_list[0].model, server.model, client_list[0].auxiliary_model
    )
    param_counts = utils.process_counts(param_counts)
    logging.info(
        f"Model size : Client = {mod_sizes[0]:.3f}MiB, " +
        f"Server = {mod_sizes[1]:.3f}MiB, " +
        f"Auxiliary = {mod_sizes[2]:.3f}MiB."
    )
    logging.info(
        f"# parameters : Client = {param_counts[0]}, " +
        f"Server = {param_counts[1]}, " +
        f"Auxiliary = {param_counts[2]}."
    )
        
    check_and_mark_lora_trainable(cfg, server, client_list)

    if False:
        utils.print_model(server.model)
        utils.print_model(client_list[0].model)
        client_list[0].auxiliary_model

    return server, client_list, test_loader

# -----------------------------------------------------------------------------
@hydra.main(
    config_path='hydra_config', config_name='config', version_base='1.3'
)
def main(cfg: DictConfig):

    #print("##" * 10 + " CFG FILE IS: " + "##" * 10)
    #print(OmegaConf.to_yaml(cfg))

    # wandb setup
    if cfg.save:
        wandb.login()
        wandb_run_name = f'{cfg.algorithm.name}_{cfg.model.name}_{cfg.dataset.name}'
        if cfg.dataset.distribution in ['iid', 'noniid']:
            wandb_run_name += f'_{cfg.dataset.distribution}'
        else:
            wandb_run_name += f'_alp_{cfg.dataset.alpha}'

        wandb_run_name += datetime.now().strftime(r'_%y%m%d_%H%M%S')

        wandb.init(
            project="fsl-sage", group=cfg.dataset.name,
            name=wandb_run_name,
            config={k: v for k, v in cfg.items() if not isinstance(v, (list,tuple))}
        )

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
    checkpointer = utils.Checkpointer(
        'accuracy', metric_obj='max', save_dir=cfg.save_path
    ) if cfg.save else None

    def __dummy_fn(*args, **kwargs):
        pass

    results = run_fl_algorithm(
        cfg, server, client_list, test_loader, checkpointer,
        global_torch_device,
        logger_fn=wandb.log if cfg.save else __dummy_fn
    )

    # end wandb run
    wandb.finish()

    # save all results
    train_metrics = {}
    for met in results.train_metrics:
        train_metrics.update(met)

    save_res = results.metric_lists
    save_res['comm_load'] = results.comm_load
    save_res.update(train_metrics)

    # save models and results
    if cfg.save:
        file_name = os.path.join(cfg.save_path, 'results.json')
        with open(file_name, 'w') as outf:
            json.dump(save_res, outf)
            logging.info(f"[NOTICE] Saved results to '{file_name}'.")
        
        metrics_file = os.path.join(cfg.save_path, 'metrics.pt')
        torch.save(
            [*list(results.metric_lists.values()), results.comm_load],
            metrics_file
        )

        # save trained models
        model_save_path = os.path.join(cfg.save_path, 'models')
        os.makedirs(model_save_path, exist_ok=True)
        utils.save_model(
            results.client_model,
            os.path.join(model_save_path, 'client_model.pt')
        )
        if results.server_model is not None:
            utils.save_model(
                results.server_model,
                os.path.join(model_save_path, 'server_model.pt')
            )
        #for i, c in enumerate(results.client_list):
        #    utils.save_model(
        #        c.auxiliary_model,
        #        os.path.join(model_save_path, f'auxiliary_model_{i}.pt')
        #    )

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------