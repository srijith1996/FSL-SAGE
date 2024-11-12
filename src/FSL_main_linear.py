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
from trains import model_linear

print("Whether we are using GPU: ", torch.cuda.is_available())
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#use_cuda = True if torch.cuda.is_available() else False
 
# for debugging
#torch.set_printoptions(sci_mode=True)
    
class Client():
    def __init__(self, id, train_loader, c_args):
        if c_args['dataset'] == "cifar":
            self.model = model_linear.Client_model_cifar() 
            self.auxiliary_model = model_linear.Auxiliary_model_cifar(bias=True)
        else:
            raise Exception("Only CIFAR is supported for now")
        # elif c_args['dataset'] == "femnist":
        #     self.model = model.Client_model_femnist() 
        #     self.auxiliary_model = model.Auxiliary_model_femnist()
        self.criterion = nn.NLLLoss().to(DEVICE)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=c_args["lr"])       
        self.optimizer = optim.Adam(self.model.parameters(), lr=c_args["lr"])       
        self.auxiliary_criterion = nn.NLLLoss().to(DEVICE)
        self.auxiliary_optimizer = optim.SGD(self.auxiliary_model.parameters(), lr=c_args["lr"])
        
        self.train_loader = train_loader
        self.epochs = c_args['epoch']
        self.dataset_size = len(self.train_loader) * c_args["batch_size"] 

class Server():
    def __init__(self, c_args, s_args):
        if c_args['dataset'] == "cifar":
            self.model = model_linear.Server_model_cifar()
        else:
            raise Exception("Only CIFAR is supported for now")
        # elif c_args['dataset'] == "femnist":
        #     self.model = model.Server_model_femnist()
        self.criterion = nn.NLLLoss().to(DEVICE)
        self.alignLoss = nn.MSELoss().to(DEVICE)

        # Srijith: If we optimize the server once for every client, we want to
        # divide the learning rate by the number of active clients
        #self.optimizer = optim.SGD(self.model.parameters(), lr=(s_args["lr"] / s_args["activated"]))
        self.optimizer = optim.Adam(self.model.parameters(), lr=(s_args["lr"] / s_args["activated"]))

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
        trainLoader_list.append(DataLoader(utils.DatasetSplit(trainSet, train_set), batch_size=c_args['batch_size'], shuffle=True, pin_memory=False))
    
    
    testLoader = DataLoader(testSet, batch_size=c_args['batch_size'], shuffle=False, pin_memory=False)
    
    
    # Define the server, and the list of client copies
    server = Server(c_args, s_args)
    client_copy_list = []
    
    for i in range(s_args["activated"]):   
        client_copy_list.append(Client(i, trainLoader_list[i], c_args))
    
    # Initial client model
    # Initial server model
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
    batch_max_round = total // c_args["batch_size"] // s_args["activated"]

    assert c_args['batch_size'] <= batch_max_round, \
        f"Chosen batch_size per client ({c_args['batch_size']}) is larger than the dataset size per client ({batch_max_round})."

    # TODO: Right now the code assumes activated clients = total number of clients.
    # May need to change this later
    # These save the server and client gradients need for alignment
    # Note that server must also store grads w.r.t. to client data
    save_server_grads = [ torch.empty(0, 2304).to(DEVICE) for _ in range(s_args["activated"]) ]
    save_smashed_input = [ torch.empty(0, 2304).to(DEVICE) for _ in range(s_args["activated"]) ]
    save_labels = [ torch.empty(0).to(DEVICE) for _ in range(s_args['activated']) ]

    def print_grad_mse(true_grad, approx_grad, r, i, k, pre=''):
        assert approx_grad.shape == true_grad.shape
        with torch.no_grad():
            #print("True grad: ", true_grad)
            #print("Approx grad: ", approx_grad)
            nmse_grad = torch.nn.functional.mse_loss(true_grad, approx_grad, reduction='sum') / torch.sum(true_grad**2)
            print(f"{pre} <round {r:2d}, client {i:2d}, batch index {k:2d}> NMSE: {nmse_grad:0.3e}")
        
    client_i = 0
    start_index = 0
    batch_round = u_args['batch_round']
    max_batch = batch_round
    it_list = [iter(tl) for tl in trainLoader_list]
    num_resets = [0 for _ in range(s_args['activated'])]
    #batch_count = [0 for _ in range(s_args['activated'])]

    # WARM START
    print("----------------------------- WARM START USING SL -----------------------------------")
    WARM_START_EPOCHS = 1
    print(f"Configured epochs = {WARM_START_EPOCHS}")

    for r in range(WARM_START_EPOCHS):
        for i in range(s_args["activated"]):
            for k, (samples, labels) in enumerate(trainLoader_list[i]):

                # client feedforward
                samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                client_copy_list[i].optimizer.zero_grad()
                server.optimizer.zero_grad()

                # pass smashed data through full model 
                splitting_output = client_copy_list[i].model(samples)
                output = server.model(splitting_output) 
                loss = server.criterion(output, labels)
                loss.backward()
                server.optimizer.step()
                client_copy_list[i].optimizer.step()

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

    set_mark = False
    for r in range(s_args["round"]):
        for i in range(s_args["activated"]):
            for k in range(u_args["batch_round"]):

                # check if data iterator has finished iterating current cycle
                if (r * u_args["batch_round"] + k) == (num_resets[i] + 1) * len(it_list[i]):
                    num_resets[i] += 1
                    it_list[i] = iter(trainLoader_list[i])

                # sample dataset
                #batch_count[i] += 1
                samples, labels = next(it_list[i])
                samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                
                # client feedforward
                client_copy_list[i].optimizer.zero_grad()
                splitting_output = client_copy_list[i].model(samples)
                local_smashed_data = splitting_output.clone().detach().requires_grad_(True)

                # contact server and perform alignment at every l^th round and first local iteration
                if r % l == 0 and k == 0:

                    print(" ---------------------------------------- ALIGNMENT ----------------------------------------")
                    print(f"Aligning at client {i} .... ", end='')
                    smashed_data = splitting_output.clone().detach().requires_grad_(True)
                    comm_load += smashed_data.numel() * 4   # float32 = 4 bytes

                    # pass smashed data through server
                    server.optimizer.zero_grad()
                    output = server.model(smashed_data) 
                    loss = server.criterion(output, labels)
                    loss.backward()
                    server.optimizer.step()

                    # save smashed data to memory
                    save_smashed_input[i] = torch.cat(
                        (save_smashed_input[i], local_smashed_data.clone().detach()), dim = 0)
                    save_labels[i] = torch.cat((save_labels[i], labels), dim=0).long()

                    # recompute gradients of smashed data
                    server.optimizer.zero_grad()
                    all_ins = save_smashed_input[i].clone().detach().requires_grad_(True)
                    all_out = server.model(all_ins)
                    all_loss = server.criterion(all_out, save_labels[i])
                    all_loss.backward()
                    save_server_grads[i] = all_ins.grad.clone().detach()
                    server.optimizer.zero_grad()

                    # Debug the newly computed grad approximation
                    if True:
                        with torch.no_grad():
                            set_mark = True
                            approx_grad = client_copy_list[i].auxiliary_model(local_smashed_data).clone().detach()
                            true_grad = save_server_grads[i][-c_args["batch_size"]:].clone().detach()
                            print_grad_mse(true_grad, approx_grad, r, i, k, "[Before alignment] ")
               
                    # perform alignment for current client
                    client_copy_list[i].auxiliary_model.align(save_smashed_input[i], save_server_grads[i])
                    print(".... done.")

                # client backpropagation and update client-side model weights
                client_copy_list[i].auxiliary_optimizer.zero_grad()
                client_grad_approx = client_copy_list[i].auxiliary_model(local_smashed_data) 

                # Debug the newly computed grad approximation
                if True:
                    if set_mark and r % l == 0:
                        set_mark = False
                        approx_grad = client_grad_approx.clone().detach()
                        true_grad = save_server_grads[i][-c_args["batch_size"]:].clone().detach()
                        print_grad_mse(true_grad, approx_grad, r, i, k, "[After alignment] ")
                        print(" -------------------------------------------------------------------------------------------")
                
                # for debugging ********
                if True:
                    l_sm_d = local_smashed_data.clone().detach().requires_grad_(True)
                    out = server.criterion(server.model(l_sm_d), labels)
                    out.backward()
                    print_grad_mse(l_sm_d.grad, client_grad_approx, r, i, k)

                splitting_output.backward(client_grad_approx)
                client_copy_list[i].optimizer.step()

                # Test on training data (debugging)
                with torch.no_grad():
                    tr_loss = []
                    tr_correct = 0
                    for samples, labels in trainLoader_list[i]:
                        samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                        output = server.model(client_copy_list[i].model(samples))
                        batch_loss = server.criterion(output, labels)
                        tr_loss.append(batch_loss.item())
                        _, predicted = torch.max(output.data, 1)
                        tr_correct += predicted.eq(labels.view_as(predicted)).sum().item()
                    tr_loss = sum(tr_loss) / len(tr_loss)
                    print(f'\t\t[R{r:2d} C{i:2d} k{k:2d}] tr. loss: {tr_loss:.2f}, tr acc: {100. * tr_correct / len(trainLoader_list[i].dataset):.2f}%')


        # Model Aggregation (weighted)
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
            comm_load += 2 * calculate_load(client_copy_list[i].model)
            comm_load += 2 * calculate_load(client_copy_list[i].auxiliary_model)
            
        # Inference
        aggregated_client.to(DEVICE)
        aggregated_client.load_state_dict(aggregated_client_weights)
        test_correct = 0
        test_loss = []
        with torch.no_grad():
            for samples, labels in testLoader:
                samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                splitting_output = aggregated_client(samples)
                output = server.model(splitting_output)
                batch_loss = server.criterion(output, labels)
                test_loss.append(batch_loss.item())
                _, predicted = torch.max(output.data, 1)
                test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
            loss = sum(test_loss) / len(test_loss)
            print(
                'Round {}, for the weighted aggregated final model, testing loss: {:.2f}, testing acc: {:.2f}%  ({}/{})'
                    .format(r, loss, 100. * test_correct / len(testLoader.dataset), test_correct, len(testLoader.dataset)))

        acc_list.append(test_correct / len(testLoader.dataset))
        loss_list.append(loss)
        comm_load_list.append(comm_load)

    # while r < s_args["round"]:
    #     #print("in 1st loop: ", batch_max_round, max_batch)
    #     while max_batch <= batch_max_round and client_i < s_args["activated"]: 
    #         client_batch_index = start_index

    #         #print("in 2nd loop: ", client_batch_index, batch_max_round, max_batch)
    #         while client_batch_index < batch_max_round and client_batch_index < max_batch: 
    #             # For a given client, iterate through the samples it needs to train on
    #             # TODO: Store last samples on device for training.
    #             samples, labels = next(it_list[client_i])
    #             client_copy_list[client_i].optimizer.zero_grad()
    #             samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
    #             
    #             # client feedforward
    #             splitting_output = client_copy_list[client_i].model(samples)
    #             #torch.set_printoptions(sci_mode=True)
    #             local_smashed_data = splitting_output.clone().detach().requires_grad_(True)

    #             # (r + 1) is to shift 0th, 1st, 2nd rounds to 1st, 2nd, 3rd, ...
    #             # (r + 1) + 1 is to get the last grads before alignment
    #             #print("in 3rd loop: ", client_batch_index, max_batch-1, r, (((r + 1) + 1) % l))
    #             if client_batch_index == max_batch - 1 and (r + 1) % l == 0:

    #                 # Pass smashed data through the server model to update the server model
    #                 smashed_data = splitting_output.clone().detach().requires_grad_(True)
    #                 comm_load += smashed_data.numel() * 4   # float32 = 4 bytes

    #                 output = server.model(smashed_data) 
    #                 loss = server.criterion(output, labels)
    #                 loss.backward()
    #                 server.optimizer.step()
    #                 server.optimizer.zero_grad()

    #                 #client_batch_index += 1

    #                 # ---------------------------------------------------------------------------------------------------------------------
    #                 # Srijith: Perhaps here we need to recompute and save gradients everytime the server-side model changes
    #                 save_smashed_input[client_i] = torch.cat((save_smashed_input[client_i], local_smashed_data.clone().detach()), dim = 0)
    #                 save_labels[client_i] = torch.cat((save_labels[client_i], labels), dim=0).long()
    #                 #print(save_labels)

    #                 # Srijith: Run all previous smashed inputs through the server-side model
    #                 #print(save_smashed_input[client_i])
    #                 #start_time_serv = time.time()
    #                 all_ins = save_smashed_input[client_i].clone().detach().requires_grad_(True)
    #                 all_out = server.model(all_ins)
    #                 all_loss = server.criterion(all_out, save_labels[client_i])
    #                 all_loss.backward()
    #                 save_server_grads[client_i] = all_ins.grad.clone().detach()
    #                 server.optimizer.zero_grad()
    #                 #print(f"  -- [Client {client_i:2d}, Round {r:2d}, Batch {client_batch_index:2d}] Time to recompute server grads: {time.time() - start_time_serv:.2e}s")

    #                 # Debug the newly computed grad approximation
    #                 if True:
    #                     with torch.no_grad():
    #                         print(" ________________________________________________________ ALIGNMENT ______________________________________________________________")
    #                         approx_grad = client_copy_list[client_i].auxiliary_model(local_smashed_data).clone().detach()
    #                         true_grad = save_server_grads[client_i][-c_args["batch_size"]:].clone().detach()
    #                         print_grad_mse(true_grad, approx_grad, client_i, "[Before alignment] ")
    #             
    #                 # ------------------------------------------------------------------------------------------------
    #                 lambda_reg = 1e-3  # Regularization parameter, adjust as needed
    #                 X = save_smashed_input[client_i] # Martix of (N, D) size
    #                 Y = save_server_grads[client_i]  # Matrix of (N, C) size (here C = D since grad same dimension)

    #                 #print("X = ", X, "\nshape = ", X.shape)
    #                 #print("Y = ", Y, "\nshape = ", Y.shape)

    #                 # Srijith: Compute (X^T X + N λ I) where N is the number of data points in X and Y
    #                 #start_time_align = time.time()
    #                 XtX = X.T @ X
    #                 XtX = XtX + X.shape[0] * lambda_reg * torch.eye(XtX.shape[0]).to(DEVICE)

    #                 # Srijith: Calculate W = Y^T . X . (X^T X + NλI)^(-1) using torch.solve.
    #                 # Srijith: The left=False solves for WA = B rather than AW = B, which is what we want
    #                 W = torch.linalg.solve(XtX, Y.T @ X, left=False)
    #                 #print(W)

    #                 # ------------------------------------------------------------------------------------------------
    #                 #client_copy_list[client_i].auxiliary_model.weight.data = \
    #                 #    nn.Parameter(W).to(DEVICE)
    #                 client_copy_list[client_i].auxiliary_model.set_weight(W)
    #                 #print(f"  -- [Client {client_i:2d}, Round {r:2d}, Batch {client_batch_index:2d}] Time to align aux model: {time.time() - start_time_align:.2e}s")

    #                 # ---------------------------------------------------------------------------------------------------------------------

    #             # client backpropagation and update client-side model weights
    #             client_copy_list[client_i].auxiliary_optimizer.zero_grad()
    #             client_grad_approx = client_copy_list[client_i].auxiliary_model(local_smashed_data) 

    #             splitting_output.backward(client_grad_approx)
    #             client_copy_list[client_i].optimizer.step()

    #             # Debug the newly computed grad approximation
    #             if True:
    #                 if client_batch_index == max_batch - 1 and (r + 1) % l == 0:
    #                     approx_grad = client_grad_approx.clone().detach()
    #                     true_grad = save_server_grads[client_i][-c_args["batch_size"]:].clone().detach()
    #                     print_grad_mse(true_grad, approx_grad, client_i, "[After alignment] ")
    #                     print(" _________________________________________________________________________________________________________________________________")
    #             
    #             client_batch_index += 1

    #         
    #         if client_i == s_args["activated"] - 1:
    #             client_i = 0
    #             if max_batch + batch_round > batch_max_round:        # reached end of data
    #                 start_index = 0
    #                 max_batch = batch_round
    #             else:
    #                 start_index += batch_round
    #                 max_batch += batch_round
    #             break
    #         else:
    #             client_i += 1

    #     # ===========================================================================================
    #     # Model Aggregation (weighted)
    #     # ===========================================================================================

    #     # Initial the aggregated model and its weights
    #     aggregated_client = copy.deepcopy(client_copy_list[0].model)
    #     aggregated_client_weights = aggregated_client.state_dict()

    #     for key in aggregated_client_weights:
    #         aggregated_client_weights[key] = client_copy_list[0].model.state_dict()[key] * factor[0]

    #     for i in range(1, s_args["activated"]):
    #         for key in aggregated_client_weights:
    #             aggregated_client_weights[key] += client_copy_list[i].model.state_dict()[key] * factor[i]

    #     # Update client model weights and auxiliary weights
    #     for i in range(s_args["activated"]):
    #         client_copy_list[i].model.load_state_dict(aggregated_client_weights)
    #         comm_load += 2 * calculate_load(client_copy_list[i].model)
    #         comm_load += 2 * calculate_load(client_copy_list[i].auxiliary_model)
    #         
    #     # ===========================================================================================
    #     # Inference
    #     # ===========================================================================================
    #     aggregated_client.to(DEVICE)
    #     aggregated_client.load_state_dict(aggregated_client_weights)
    #     test_correct = 0
    #     test_loss = []
    #     for samples, labels in testLoader:
    #         samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
    #         splitting_output = aggregated_client(samples)
    #         output = server.model(splitting_output)
    #         batch_loss = server.criterion(output, labels)
    #         test_loss.append(batch_loss.item())
    #         _, predicted = torch.max(output.data, 1)
    #         test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
    #     loss = sum(test_loss) / len(test_loss)
    #     print(
    #         'Round {}, for the weighted aggregated final model, testing loss: {:.2f}, testing acc: {:.2f}%  ({}/{})'
    #             .format(r, loss, 100. * test_correct / len(testLoader.dataset), test_correct, len(testLoader.dataset)))

    #     acc_list.append(test_correct / len(testLoader.dataset))
    #     loss_list.append(loss)
    #     comm_load_list.append(comm_load)
    #     r += 1

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
