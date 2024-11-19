"""
Add local loss, no need to transmit the gradient
client transmit smashed data not every batch data
server part: model 0 batch 0 - model 0 batch 4 - model 1 batch 0 - model 1 batch 4  ...
"""

import os
import time
import math
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import copy
from torch.utils.data import DataLoader
from utils import options, utils
from trains import client, model, aux_models
from trains import server as serv

DEBUG = False
USE_64BIT = False
WARM_START_EPOCHS = 0
AGGREGATE_AUXILIARY_MODELS = False

if USE_64BIT: torch.set_default_dtype(torch.float64)

print("Using GPU: ", torch.cuda.is_available())
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#use_cuda = True if torch.cuda.is_available() else False
 
if DEBUG: torch.set_printoptions(sci_mode=True)
    
def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)   

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

if __name__ == '__main__':    
    ## get system configs
    args = options.args_parser('FSL-Approx')    #---------todo
    u_args, s_args, c_args = options.group_args(args) #---------todo
    utils.show_utils(u_args) #---------todo
    
    seed = u_args['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    assert s_args["activated"] <= s_args["client"], \
        f"# activated clients {s_args['activated']} is greater than # clients {s_args['client']}"

    ## process dataset
    trainSet, testSet = utils.get_dataset(s_args, u_args) 
    client_train_set, client_test_set = utils.depart_dataset(u_args, s_args, trainSet, testSet)

    trainLoader_list = []
    for i in range(s_args["activated"]):
        train_set = client_train_set[i]["idxs"]
        trainLoader_list.append(
            DataLoader(
                utils.DatasetSplit(trainSet, train_set),
                batch_size=c_args['batch_size'],
                shuffle=True, pin_memory=False)
        )
    
    testLoader = DataLoader(testSet, batch_size=c_args['batch_size'], shuffle=False, pin_memory=False)
    
    
    # Define the server, and the list of client copies
    server = serv.Server(serv.Server_model_cifar(), s_args, device=DEVICE)
    client_copy_list = []
    
    for i in range(s_args["activated"]):   
        client_copy_list.append(client.Client(
            i, trainLoader_list[i],
            client.Client_model_cifar(),
            aux_models.VanillaFSL(
                6 * 6 * 64, 10, server, device=DEVICE, n_hidden=None,
                align_iters=5000, align_step=3e-3
            ),
            c_args, device=DEVICE
        ))
    
    # Initial client & server model
    init_all(client_copy_list[0].model, torch.nn.init.normal_, mean=0., std=0.05) 
    init_all(client_copy_list[0].auxiliary_model, torch.nn.init.normal_, mean=0., std=0.05) 
    init_all(server.model, torch.nn.init.normal_, mean=0., std=0.05) 
    #init_all(client_copy_list[0].model, torch.nn.init.kaiming_normal_) 
    #init_all(client_copy_list[0].auxiliary_model, torch.nn.init.normal_, mean=0., std=1.0) 
    #init_all(server.model, torch.nn.init.kaiming_normal_) 
    
        
    for i in range(s_args["activated"]):
        client_copy_list[i].model.load_state_dict(client_copy_list[0].model.state_dict())
        client_copy_list[i].model.to(DEVICE)
        client_copy_list[i].auxiliary_model.load_state_dict(client_copy_list[0].auxiliary_model.state_dict())
        client_copy_list[i].auxiliary_model.to(DEVICE)
    server.model.to(DEVICE)    
    
        
    # # Calculate the weights for dataset size
    dataset_size_list = [client_copy_list[i].dataset_size for i in range(s_args["activated"])]
    total = sum(dataset_size_list)
    factor = [i / total for i in dataset_size_list]
    print("Aggregation Factor: ", factor)


    r = 0  # current communication round
    l = c_args['align'] # the number of training steps to take before the alignment step 
    print("seed is " + str(seed))
    print("l is " + str(l))
    acc_list = []
    loss_list = []
    comm_load_list = []
    start = time.time()
    comm_load = 0

    assert c_args['batch_size'] <= total  // s_args["activated"], \
        f"Chosen batch_size per client ({c_args['batch_size']}) is larger than the dataset size per client ({total // s_args['activated']})."

    # TODO: Right now the code assumes activated clients = total number of clients.
    # May need to change this later

    # it_list = [iter(tl) for tl in trainLoader_list]
    it_list = []
    num_resets = [0 for _ in range(s_args['activated'])]

    # WARM START
    print("----------------------------- WARM START USING SL -----------------------------------")
    print(f"Configured epochs = {WARM_START_EPOCHS}")

    client_optim_ws = [optim.Adam(c.model.parameters(), lr=1e-3) for c in client_copy_list]
    server_optim_ws = optim.Adam(server.model.parameters(), lr=1e-3)

    for i in range(s_args['activated']):
        it_list.append(iter(trainLoader_list[i]))

    for r in range(WARM_START_EPOCHS):
        for i in range(s_args["activated"]):
            for k, (samples, labels) in enumerate(trainLoader_list[i]):

                # client feedforward
                if USE_64BIT:
                    samples, labels = samples.to(DEVICE).double(), labels.to(DEVICE).long()
                else:
                    samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                client_optim_ws[i].zero_grad()
                server_optim_ws.zero_grad()

                # pass smashed data through full model 
                splitting_output = client_copy_list[i].model(samples)
                output = server.model(splitting_output) 
                loss = server.criterion(output, labels)
                loss.backward()
                server_optim_ws.step()
                client_optim_ws[i].step()

        # aggregate client model
        aggregated_client = copy.deepcopy(client_copy_list[0].model)
        aggregated_client_weights = aggregated_client.state_dict()

        for key in aggregated_client_weights:
            aggregated_client_weights[key] = client_copy_list[0].model.state_dict()[key] * factor[0]

        for i in range(1, s_args["activated"]):
            for key in aggregated_client_weights:
                aggregated_client_weights[key] += client_copy_list[i].model.state_dict()[key] * factor[i]

        # Update client model weights and auxiliary weights
        for i in range(s_args["activated"]):
            client_copy_list[i].model.load_state_dict(aggregated_client_weights)
            
        # Inference
        aggregated_client.to(DEVICE)
        aggregated_client.load_state_dict(aggregated_client_weights)
        test_correct = 0
        test_loss = []
        with torch.no_grad():
            for samples, labels in testLoader:
                if USE_64BIT:
                    samples, labels = samples.to(DEVICE).double(), labels.to(DEVICE).long()
                else:
                    samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                splitting_output = aggregated_client(samples)
                output = server.model(splitting_output)
                batch_loss = server.criterion(output, labels)
                test_loss.append(batch_loss.item())
                _, predicted = torch.max(output.data, 1)
                test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
            loss = sum(test_loss) / len(test_loss)
            print('WS Round {}, testing loss: {:.2f}, testing acc: {:.2f}%'
                    .format(r, loss, 100. * test_correct / len(testLoader.dataset)))
    print("-------------------------------------------------------------------------------------")

    batch_max_round = total // s_args["activated"] // c_args["batch_size"]
    set_mark = False
    dbg_saved_aux_params = [[] for _ in range(s_args['activated'])]
    for r in range(s_args["round"]):
        it_list = []

        for i in range(s_args['activated']):
            it_list.append(iter(trainLoader_list[i]))

        # round_make = u_args["batch_round"]
        # assert False, f"Rounds are {batch_max_round // round_make}"
        for k in range(batch_max_round // u_args["batch_round"]):
            for i in range(s_args["activated"]):
                for b in range(u_args["batch_round"]):
                    # if (r * u_args["batch_round"] + k) == (num_resets[i] + 1) * len(it_list[i]):
                    #     num_resets[i] += 1
                    #     it_list[i] = iter(trainLoader_list[i])

                    # sample dataset
                    #batch_count[i] += 1
                    samples, labels = next(it_list[i])
                    if USE_64BIT:
                        samples, labels = samples.to(DEVICE).double(), labels.to(DEVICE).long()
                    else:
                        samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                    
                    # client feedforward
                    client_copy_list[i].optimizer.zero_grad()
                    splitting_output = client_copy_list[i].model(samples)
                    local_smashed_data = splitting_output.clone().detach().requires_grad_(True)
                    smashed_data = splitting_output.clone().detach().requires_grad_(True)

                    # client backpropagation and update client-side model weights
                    client_copy_list[i].auxiliary_model.optimizer.zero_grad()
                    out = client_copy_list[i].auxiliary_model(local_smashed_data, labels) # TODO CHECK
                    out.backward()
                    client_copy_list[i].auxiliary_model.optimizer.step()

                    splitting_output.backward(local_smashed_data.grad)
                
                    client_copy_list[i].optimizer.step()

                    if b == u_args["batch_round"] - 1:
                        server.optimizer.zero_grad()
                        output = server.model(smashed_data)
                        s_loss = server.criterion(output, labels)
                        s_loss.backward()
                        server.optimizer.step()

        # Model Aggregation (weighted)
        aggregated_client = copy.deepcopy(client_copy_list[0].model)
        aggregated_client_weights = aggregated_client.state_dict()

        for key in aggregated_client_weights:
            aggregated_client_weights[key] = client_copy_list[0].model.state_dict()[key] * factor[0]

        if AGGREGATE_AUXILIARY_MODELS:
            aggregated_client_auxiliary = copy.deepcopy(client_copy_list[0].auxiliary_model)
            aggregated_client_weights_auxiliary = aggregated_client_auxiliary.state_dict()

            for key in aggregated_client_weights_auxiliary:
                aggregated_client_weights_auxiliary[key] = client_copy_list[0].auxiliary_model.state_dict()[key] * factor[0]

        for i in range(1, s_args["activated"]):
            for key in aggregated_client_weights:
                aggregated_client_weights[key] += client_copy_list[i].model.state_dict()[key] * factor[i]
            if AGGREGATE_AUXILIARY_MODELS:
                for key in aggregated_client_weights_auxiliary:
                    aggregated_client_weights_auxiliary[key] += client_copy_list[i].auxiliary_model.state_dict()[key] * factor[i]

        # Update client model weights and auxiliary weights
        for i in range(s_args["activated"]):
            client_copy_list[i].model.load_state_dict(aggregated_client_weights)
            comm_load += 2 * calculate_load(client_copy_list[i].model)
            if AGGREGATE_AUXILIARY_MODELS:
                client_copy_list[i].auxiliary_model.load_state_dict(aggregated_client_weights_auxiliary)
                comm_load += 2 * calculate_load(client_copy_list[i].auxiliary_model)

        # Inference
        aggregated_client.to(DEVICE)
        aggregated_client.load_state_dict(aggregated_client_weights)
        test_correct = 0
        test_loss = []
        with torch.no_grad():
            for samples, labels in testLoader:
                if USE_64BIT:
                    samples, labels = samples.to(DEVICE).double(), labels.to(DEVICE).long()
                else:
                    samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                splitting_output = aggregated_client(samples)
                output = server.model(splitting_output)
                batch_loss = server.criterion(output, labels)
                test_loss.append(batch_loss.item())
                _, predicted = torch.max(output.data, 1)
                test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
            loss = sum(test_loss) / len(test_loss)
            print(' > R {:2d}, for the weighted aggregated final model, testing loss: {:.2f}, testing acc: {:.2f}%  ({:5d}/{})'
                    .format(r, loss, 100. * test_correct / len(testLoader.dataset), test_correct, len(testLoader.dataset)))
                
            with open("fsl-vanilla.txt", "a") as f:
                print('\nRound {}, for the weighted aggregated final model, testing loss: {:.2f}, testing acc: {:.2f}%  ({}/{})'
                    .format(r, loss, 100. * test_correct / len(testLoader.dataset), test_correct, len(testLoader.dataset)),
                    file=f)

        acc_list.append(test_correct / len(testLoader.dataset))
        loss_list.append(loss)
        comm_load_list.append(comm_load)


    print('The total running time for all rounds is ', round(time.time() - start, 2), 'seconds')
    print("Testing accuracy:", acc_list)
    print("Testing loss:", loss_list)
    
    # Save reults to .json files.
    results = {'test_loss': loss_list, 'test_acc' : acc_list,
               'comm_load' : comm_load_list, 'step': s_args['t_round']}

    if u_args['save']:
        file_name = os.path.join(u_args['save_path'], 'results.json')
        with open(file_name, 'w') as outf:
            json.dump(results, outf)
            print(f"\033[1;36m[NOTICE] Saved results to '{file_name}'.\033[0m")
        
        metrics_file = os.path.join(u_args['save_path'], 'metrics.pt')
        torch.save([acc_list, loss_list, comm_load_list], metrics_file)
