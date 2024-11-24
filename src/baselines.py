# -----------------------------------------------------------------------------
import os, sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import utils, options, logs
from trains import client, algs
from trains import server as serv
import logging

# -----------------------------------------------------------------------------
# TODO: Move these to command line args
DEBUG = True
USE_64BIT = False
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if USE_64BIT: torch.set_default_dtype(torch.float64)
if DEBUG: torch.set_printoptions(sci_mode=True)

# -----------------------------------------------------------------------------
def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)   

# -----------------------------------------------------------------------------
def get_dataset(u_args, s_args, c_args):
    ## process dataset
    trainSet, testSet = utils.get_dataset(s_args, u_args) 
    client_train_set, client_test_set = utils.depart_dataset(
        u_args, s_args, trainSet, testSet
    )

    trainLoader_list = []
    for i in range(s_args["activated"]):
        train_set = client_train_set[i]["idxs"]
        trainLoader_list.append(
            DataLoader(
                utils.DatasetSplit(trainSet, train_set),
                batch_size=c_args['batch_size'],
                shuffle=True, pin_memory=False
            )
        )
    
    testLoader = DataLoader(
        testSet, batch_size=c_args['batch_size'], shuffle=False,
        pin_memory=False
    )
    return trainLoader_list, testLoader, client_train_set
    
# -----------------------------------------------------------------------------
def train_alg(alg_name, save_path, u_args, s_args, c_args):

    trainLoader_list, testLoader, client_train_set = get_dataset(
        u_args, s_args, c_args)
    criterion = nn.NLLLoss().to(DEVICE)

    # Calculate the weights for dataset size
    dataset_size_list = [
        int(client_train_set[i]['num']) for i in range(s_args["activated"])
    ]
    total = sum(dataset_size_list)
    factor = [i / total for i in dataset_size_list]
    logging.info(f"Aggregation Factor: {factor}")

    # for saving models and info
    if alg_name == 'fed_avg':

        model = nn.Sequential(
            client.Client_model_cifar(), serv.Server_model_cifar()
        )
        init_all(model, torch.nn.init.normal_, mean=0., std=0.05)

        # train
        aggregated_client, test_loss, acc, comm_load_list = algs.fed_avg(
            s_args['round'], model, criterion, trainLoader_list, testLoader,
            factor, 1e-3, use_64bit=False, device=DEVICE
        )

        # save model
        utils.save_model(
            aggregated_client, os.path.join(save_path, 'agg_client.pt')
        )

    else:

        server = serv.Server(
            serv.Server_model_cifar(), s_args, device=DEVICE
        )
        client_copy_list = []
    
        for i in range(s_args["activated"]):   
            client_copy_list.append(client.Client(
                i, trainLoader_list[i],
                client.Client_model_cifar(),
                None, c_args, device=DEVICE
            ))

        # Initial client & server model
        init_all(
            client_copy_list[0].model, torch.nn.init.normal_, mean=0., std=0.05
        )
        init_all(server.model, torch.nn.init.normal_, mean=0., std=0.05) 

        for i in range(s_args["activated"]):
            client_copy_list[i].model.load_state_dict(
                client_copy_list[0].model.state_dict()
            )
            client_copy_list[i].model.to(DEVICE)
        server.model.to(DEVICE)    
    
        if alg_name == 'sl_single_server':
            client_copy_list, aggregated_client, server, test_loss, acc, comm_load_list = \
            algs.sl_single_server(
                s_args['round'], client_copy_list, server, trainLoader_list,
                testLoader, factor, 1e-3, 1e-3, device=DEVICE
            )

        elif alg_name == 'sl_multi_server':
            client_copy_list, aggregated_client, server, test_loss, acc, comm_load_list = \
            algs.sl_multi_server(
                s_args['round'], client_copy_list, server, trainLoader_list,
                testLoader, factor, 1e-3, 1e-3, device=DEVICE
            )
        else:
            raise Exception(f"Unknown algorithm name: {alg_name}")

        utils.save_model(
            aggregated_client, os.path.join(save_path, 'agg_client.pt')
        )
        utils.save_model(server, os.path.join(save_path, 'server.pt'))

    save_dict = {'test_loss': test_loss,
                 'test_acc' : acc,
                 'comm_load': comm_load_list
                }
    filename = os.path.join(save_path, 'test_metrics.json')
    with open(filename, 'w') as outf:
            json.dump(save_dict, outf)
            logging.info(f"[NOTICE] Saved results to '{filename}'.")
    
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    args = options.args_parser('sl_single_server')    #---------todo
    u_args, s_args, c_args = options.group_args(args, create_dir=False) #---------todo

    train_elms = ["fed_avg", "sl_single_server", "sl_multi_server"]

    for te in train_elms:
        args.method = te
        save_path = f'../saves/baselines/{args.method}'
        os.makedirs(save_path, exist_ok=True)

        logs.configure_logging(os.path.join(save_path, "output.log"))
        logs.log_hparams(u_args, c_args, s_args, settings_dir=save_path)

        train_alg(args.method, save_path, u_args, s_args, c_args)

# -----------------------------------------------------------------------------