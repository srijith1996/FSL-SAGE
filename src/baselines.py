# -----------------------------------------------------------------------------
import os, sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import utils, options, logs
from trains import client, algs, resnet
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
def train_alg(alg_name, u_args, s_args, c_args):

    save, save_path = u_args['save'], u_args['save_path']
    trainLoader_list, testLoader, client_train_set = get_dataset(
        u_args, s_args, c_args)
    criterion = nn.NLLLoss().to(DEVICE)

    # Calculate the weights for dataset size
    dataset_size_list = [
        int(client_train_set[i]['num']) for i in range(s_args["activated"])
    ]
    print(dataset_size_list)
    total = sum(dataset_size_list)
    factor = [i / total for i in dataset_size_list]
    logging.info(f"Aggregation Factor: {factor}")

    # for saving models and info
    if alg_name == 'fed_avg':

        if u_args['model'] == 'simple_conv':
            model = nn.Sequential(
                client.Client_model_cifar(), serv.Server_model_cifar()
            )
            init_all(model, torch.nn.init.normal_, mean=0., std=0.05)

        elif u_args['model'] == 'resnet18':
            model = resnet.resnet18(num_classes=10)

        else:
            logging.error(f"Model {u_args['model']} is not implemented.")

        # compute sizes of different models
        mod_sizes = utils.compute_model_size(model)
        logging.info(f"Model size = {mod_sizes[0]:.3f}MiB.")

        # train
        aggregated_client, test_loss, acc, comm_load_list = algs.fed_avg(
            s_args['round'], model, criterion, trainLoader_list, testLoader,
            factor, 1e-3, use_64bit=False, device=DEVICE
        )

        # save model
        if save:
            utils.save_model(
                aggregated_client, os.path.join(save_path, 'agg_client.pt')
            )

    else:

        client_copy_list = []
        if u_args['model'] == 'simple_conv':
            server = serv.Server(
                serv.Server_model_cifar(), s_args, device=DEVICE
            )
    
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

        elif u_args['model'] == 'resnet18':
            cmodels, smodel = resnet.resnet18_sl(num_clients=s_args['activated'], num_classes=10)
            server = serv.Server(smodel, s_args, device=DEVICE)
            for i in range(s_args['activated']):
                client_copy_list.append(client.Client(
                    i, trainLoader_list[i], cmodels[i],
                    None, c_args, device=DEVICE
                ))

        else:
            logging.error(f"Model {u_args['model']} is not implemented.")

        # compute sizes of different models
        mod_sizes = utils.compute_model_size(
            client_copy_list[0].model, server.model
        )
        logging.info(f"Model size - Client = {mod_sizes[0]:.3f}MiB, Server = {mod_sizes[1]:.3f}MiB.")

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

        if save:
            utils.save_model(
                aggregated_client, os.path.join(save_path, 'agg_client.pt')
            )
            utils.save_model(server, os.path.join(save_path, 'server.pt'))

    if save:
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

    args = options.args_parser()    #---------todo
    u_args, s_args, c_args = options.group_args(args, create_dir=True) #---------todo

    #train_elms = ["fed_avg", "sl_single_server", "sl_multi_server"]

    #for te in train_elms:
    #args.method = te
    #save_path = f'../saves/baselines/{args.method}'
    #os.makedirs(save_path, exist_ok=True)

    if u_args['save']:
        logs.configure_logging(os.path.join(u_args['save_path'], "output.log"))
        logs.log_hparams(u_args, c_args, s_args, settings_dir=u_args['save_path'])
    else:
        u_args['save_path'] = None

    train_alg(args.method, u_args, s_args, c_args)

# -----------------------------------------------------------------------------