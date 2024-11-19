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

alg_name = sys.argv[1]
SAVE_PATH = f'../saves/baselines/{sys.argv[1]}'
os.makedirs(SAVE_PATH, exist_ok=True)

# -----------------------------------------------------------------------------
def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)   

# -----------------------------------------------------------------------------
def calculate_load(model):        
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2    # MB
#     size_all_mb = (param_size + buffer_size)   # B
    return size_all_mb

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
    return trainLoader_list, testLoader
    
# -----------------------------------------------------------------------------
def train_alg(alg_name, u_args, s_args, c_args):

    trainLoader_list, testLoader = get_dataset(u_args, s_args, c_args)
    criterion = nn.NLLLoss().to(DEVICE)

    # Calculate the weights for dataset size
    dataset_size_list = [
        client_copy_list[i].dataset_size for i in range(s_args["activated"])
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
        aggregated_client, test_loss, acc = algs.fed_avg(
            s_args['round'], c_args['batch_round'], model, criterion,
            trainLoader_list, testLoader, factor, use_64bit=False, device=DEVICE
        )

        # save model
        utils.save_model(
            aggregated_client, os.path.join(SAVE_PATH, 'agg_client.pt')
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
        init_all(
            client_copy_list[0].auxiliary_model, torch.nn.init.normal_, mean=0.,
            std=0.05
        )
        init_all(server.model, torch.nn.init.normal_, mean=0., std=0.05) 

        for i in range(s_args["activated"]):
            client_copy_list[i].model.load_state_dict(
                client_copy_list[0].model.state_dict()
            )
            client_copy_list[i].model.to(DEVICE)
        server.model.to(DEVICE)    
    
        if alg_name == 'sl_single_server':
            client_copy_list, aggregated_client, server, test_loss, acc = \
            algs.sl_single_server(
                s_args['round'], c_args['batch_round'], client_copy_list,
                server, trainLoader_list, testLoader, factor, device=DEVICE
            )

        elif alg_name == 'sl_multi_server':
            client_copy_list, aggregated_client, server, test_loss, acc = \
            algs.sl_multi_server(
                s_args['round'], c_args['batch_round'], client_copy_list,
                server, trainLoader_list, testLoader, factor, device=DEVICE
            )

        utils.save_model(
            aggregated_client, os.path.join(SAVE_PATH, 'agg_client.pt')
        )
        utils.save_model(server, os.path.join(SAVE_PATH, 'server.pt'))

    save_dict = {'test_loss': test_loss,
                 'test_acc' : acc}
    filename = os.path.join(SAVE_PATH, 'test_metrics.json')
    with open(filename, 'w') as outf:
            json.dump(save_dict, outf)
            logging.info(f"[NOTICE] Saved results to '{filename}'.")
    
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    args = options.args_parser('fed_avg')    #---------todo
    u_args, s_args, c_args = options.group_args(args) #---------todo

    logs.configure_logging(os.path.join(SAVE_PATH, "output.log"))
    logs.log_hparams(u_args, c_args, s_args)

    train_alg(args.method, u_args, s_args, c_args)

# -----------------------------------------------------------------------------