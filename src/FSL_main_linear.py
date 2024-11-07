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
 
    
class Client():
    def __init__(self, id, train_loader, c_args):
        if c_args['dataset'] == "cifar":
            self.model = model_linear.Client_model_cifar() 
            self.auxiliary_model = model_linear.Auxiliary_model_cifar()
        else:
            raise Exception("Only CIFAR is supported for now")
        # elif c_args['dataset'] == "femnist":
        #     self.model = model.Client_model_femnist() 
        #     self.auxiliary_model = model.Auxiliary_model_femnist()
        self.criterion = nn.NLLLoss().to(DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=c_args["lr"])       
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
        self.optimizer = optim.SGD(self.model.parameters(), lr=(s_args["lr"] / s_args["activated"]))

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
    
    # TODO: Right now the code assumes activated clients = total number of clients.
    # May need to change this later
    # These save the server and client gradients need for alignment
    # Note that server must also store grads w.r.t. to client data
    save_server_grads = [ torch.empty(0, 2304).to(DEVICE) for _ in range(s_args["activated"]) ]
    save_smashed_input = [ torch.empty(0, 2304).to(DEVICE) for _ in range(s_args["activated"]) ]
    save_labels = [ torch.empty(0).to(DEVICE) for _ in range(s_args['activated']) ]

    while r < s_args["round"]:
        # r + 1 is b/c indexing by 0
        # so 1st, 2nd, 3rd, 4th instead of 0th, 1st, 2nd, 3rd, ...
        # Alignment then happens every lth term instead of every (l-1)th and 0th term
        #if r != 0 and (r + 1) % l == 0:
        #    # TODO: Store the previous gradients (somewhere)
        #    # TODO: Calculate the analytic solution for solving the linear model   
        #    for client_i in range(s_args['activated']):
        #        # Calculate the analytical solution for W
        #        # The formula below is the analytical solution of linear regression 
        #        # with regularization term to handle when X is nonintvertible
        #        lambda_reg = 1e-5  # Regularization parameter, adjust as needed
        #        X = save_smashed_input[client_i] # Martix of (N, D) size
        #        Y = save_server_grads[client_i]  # Matrix of (N, C) size (here C = D since grad same dimension)

        #        # ------------------------------------------------------------------------------------------------
        #        # Srijith: Compute (X^T X + N λ I) where N is the number of data points in X and Y
        #        XtX = X.T @ X
        #        XtX = XtX + X.shape[0] * lambda_reg * torch.eye(XtX.shape[0]).to(DEVICE)

        #        # Srijith: Calculate W = Y^T . X . (X^T X + NλI)^(-1) using torch.solve.
        #        # Srijith: The left=False solves for WA = B rather than AW = B, which is what we want
        #        W = torch.linalg.solve(XtX, Y.T @ X, left=False)

        #        # Compute (X^T X + λI)
        #        #XtX = X.T @ X
        #        #I = torch.eye(XtX.shape[0]).to(DEVICE)  # Identity matrix with shape (D, D)
        #        #XtX_inv = torch.inverse(XtX + lambda_reg * I)  # Inverse of (X^T X + λI)

        #        # Calculate W = (X^T X + λI)^(-1) X^T Y
        #        #W = XtX_inv @ X.T @ Y
        #        # ------------------------------------------------------------------------------------------------

        #        client_copy_list[client_i].auxiliary_model.weight = \
        #            nn.Parameter(W).to(DEVICE)
        #     # Store the accumulated gradient per round
        #    # Note that this must be stored seperately because we need the gradients per client in order to tune the aux model, but 
        #    # we need the acc gradient to train the server.
        #    #save_server_grads = [ torch.empty(0, 2304).to(DEVICE) for _ in range(s_args["activated"]) ]
        #    #save_smashed_input = [ torch.empty(0, 2304).to(DEVICE) for _ in range(s_args["activated"]) ]

        #    # Complete the forward pass on the server-side for all the acc gradients
        #    server.optimizer.step()
        #    server.optimizer.zero_grad()

        it_list = []
        for i in range(s_args["activated"]):
            it_list.append(iter(trainLoader_list[i]))
        batch_round = u_args['batch_round']
        max_batch = batch_round
        client_i = 0
        start_index = 0

        # This is a bit complicated, but basically:
        #   batch_round - the number of local rounds a client will train on its dataset, per 
        #       client cycle. Each local round it trains on a batch of local data, hence batch_round
        #   batch_max_round - the maximum possible number of batch rounds a client can take. 
        #       This serves as a limit to the total num of batch-rounds a client will do per 
        #       global round (i think). 
        #   start_index - the index of the batch the client will start on. Increases by 
        #        batch_round every time all clients have been cycled through.
        #   max_batch - start_index + batch_round. This is used to keep track of when to stop
        #       training for a given client.
        #   client_i - index of the client currently undergoing training
        #   client_batch_index - current index of the batch, for the given client training.
        #       Used to keep track of what batch the client is currently on (NOTE THAT THIS
        #       IS CULMATIVE ACROSS CLIENT CYCLES / LOCAL ROUNDS!)
        # So the loop below will cycle through local batch_rounds for each client, and will stop
        # as soon as equal to or more than batch_max_roudns would be executed.
        # Not entirely sure why the second condition is needed in this loop since the client_i
        # always resets to 0 once all clients are passed.
        while max_batch <= batch_max_round and client_i < s_args["activated"]: 
            client_batch_index = start_index
            while client_batch_index < batch_max_round and client_batch_index < max_batch: 
                # For a given client, iterate through the samples it needs to train on
                # TODO: Store last samples on device for training.
                samples, labels = next(it_list[client_i])
                client_copy_list[client_i].optimizer.zero_grad()
                samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                
                # client feedforward
                splitting_output = client_copy_list[client_i].model(samples)
                local_smashed_data = splitting_output.clone().detach().requires_grad_(True)
                smashed_data = splitting_output.clone().detach().requires_grad_(True)

                # (r + 1) is to shift 0th, 1st, 2nd rounds to 1st, 2nd, 3rd, ...
                # (r + 1) + 1 is to get the last grads before alignment
                if client_batch_index == max_batch - 1 and ((r + 1) + 1) % l == 0:
                    comm_load += smashed_data.numel() * 4   # float32 = 4 bytes
                    output = server.model(smashed_data) 
                    loss = server.criterion(output, labels)
                    loss.backward()
                    server.optimizer.step()
                    server.optimizer.zero_grad()
                    client_batch_index += 1

                    # ---------------------------------------------------------------------------------------------------------------------
                    # Srijith: Perhaps here we need to recompute and save gradients everytime the server-side model changes
                    save_smashed_input[client_i] = torch.cat((save_smashed_input[client_i], local_smashed_data.clone().detach()), dim = 0)
                    save_labels[client_i] = torch.cat((save_labels[client_i], labels), dim=0).long()
                    #print(save_labels)

                    # Srijith: Run all previous smashed inputs through the server-side model
                    #print(save_smashed_input[client_i])
                    all_ins = save_smashed_input[client_i].clone().detach().requires_grad_(True)
                    all_out = server.model(all_ins)
                    all_loss = server.criterion(all_out, save_labels[client_i])
                    all_loss.backward()
                    save_server_grads[client_i] = all_ins.grad.clone().detach()
                    server.optimizer.zero_grad()

                    # Calculate the analytical solution for W
                    # The formula below is the analytical solution of linear regression 
                    # with regularization term to handle when X is nonintvertible
                    lambda_reg = 1e-5  # Regularization parameter, adjust as needed
                    X = save_smashed_input[client_i] # Martix of (N, D) size
                    Y = save_server_grads[client_i]  # Matrix of (N, C) size (here C = D since grad same dimension)

                    # ------------------------------------------------------------------------------------------------
                    # Srijith: Compute (X^T X + N λ I) where N is the number of data points in X and Y
                    XtX = X.T @ X
                    XtX = XtX + X.shape[0] * lambda_reg * torch.eye(XtX.shape[0]).to(DEVICE)

                    # Srijith: Calculate W = Y^T . X . (X^T X + NλI)^(-1) using torch.solve.
                    # Srijith: The left=False solves for WA = B rather than AW = B, which is what we want
                    W = torch.linalg.solve(XtX, Y.T @ X, left=False)

                    # Compute (X^T X + λI)
                    #XtX = X.T @ X
                    #I = torch.eye(XtX.shape[0]).to(DEVICE)  # Identity matrix with shape (D, D)
                    #XtX_inv = torch.inverse(XtX + lambda_reg * I)  # Inverse of (X^T X + λI)

                    # Calculate W = (X^T X + λI)^(-1) X^T Y
                    #W = XtX_inv @ X.T @ Y
                    # ------------------------------------------------------------------------------------------------

                    client_copy_list[client_i].auxiliary_model.weight = \
                        nn.Parameter(W).to(DEVICE)
                    #save_server_grads[client_i] = torch.cat((save_server_grads[client_i], smashed_data.grad.clone().detach()), dim = 0)
                    #save_server_grads[client_i] = torch.cat((save_server_grads[client_i], smashed_data.grad.clone().detach()), dim = 0)
                    # ---------------------------------------------------------------------------------------------------------------------

                    #print("------ Debugging saved data -------")
                    #print(f"Size of dataset for alignment: {smashed_data.grad.size()}")
                    #print(client_grad_approx)
                    #print(smashed_data)

                # client backpropagation and update client-side model weights
                # TODO: Sanity check to make sure the gradients here actually correspond to what I think it does
                # (e.g., that local_loss.backward() derives the gradients)
                # gradient = local_smashed_data.grad
                # client calculates the local loss and do the backpropagation and update auxiliary model weights
                client_copy_list[client_i].auxiliary_optimizer.zero_grad()
                client_grad_approx = client_copy_list[client_i].auxiliary_model(local_smashed_data) 

                splitting_output.backward(client_grad_approx)
                client_copy_list[client_i].optimizer.step()
        
                client_batch_index += 1

                    #     save_server_grads[client_i] = torch.cat((save_server_grads[client_i], smashed_data.grad.clone().detach()), dim = 0)
                    #     print(smashed_data.grad.clone().detach().size())
                    #     print(save_server_grads[client_i].size())

            
            if client_i == s_args["activated"] - 1:
                client_i = 0
                start_index += batch_round
                max_batch += batch_round
            else:
                client_i += 1

        # ===========================================================================================
        # Model Aggregation (weighted)
        # ===========================================================================================

        # Initial the aggregated model and its weights
        aggregated_client = copy.deepcopy(client_copy_list[0].model)
        aggregated_client_weights = aggregated_client.state_dict()
        # aggregated_client_auxiliary = copy.deepcopy(client_copy_list[0].auxiliary_model)
        # aggregated_client_weights_auxiliary = aggregated_client_auxiliary.state_dict()

        for key in aggregated_client_weights:
            aggregated_client_weights[key] = client_copy_list[0].model.state_dict()[key] * factor[0]
        # for key in aggregated_client_weights_auxiliary:
        #     aggregated_client_weights_auxiliary[key] = client_copy_list[0].auxiliary_model.state_dict()[key] * factor[0]

        for i in range(1, s_args["activated"]):
            for key in aggregated_client_weights:
                aggregated_client_weights[key] += client_copy_list[i].model.state_dict()[key] * factor[i]
            # for key in aggregated_client_weights_auxiliary:
            #     aggregated_client_weights_auxiliary[key] += client_copy_list[i].auxiliary_model.state_dict()[key] * factor[i]
    

        # Update client model weights and auxiliary weights
        for i in range(s_args["activated"]):
            client_copy_list[i].model.load_state_dict(aggregated_client_weights)
            # client_copy_list[i].auxiliary_model.load_state_dict(aggregated_client_weights_auxiliary)
            comm_load += 2 * calculate_load(client_copy_list[i].model)
            comm_load += 2 * calculate_load(client_copy_list[i].auxiliary_model)

            
        # ===========================================================================================
        # Inference
        # ===========================================================================================
        aggregated_client.to(DEVICE)
        aggregated_client.load_state_dict(aggregated_client_weights)
        test_correct = 0
        test_loss = []
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
        r += 1

    print('The total running time for all rounds is ', round(time.time() - start, 2), 'seconds')
    print("Testing accuracy:", acc_list)
    print("Testing loss:", loss_list)
    
    '''
        Save reults to .json files.
    '''
    results = {'test_loss': loss_list, 'test_acc' : acc_list,
               'comm_load' : comm_load_list, 'step': s_args['t_round']}

    file_name = os.path.join(u_args['save_path'], 'results.json')
    with open(file_name, 'w') as outf:
        json.dump(results, outf)
        print(f"\033[1;36m[NOTICE] Saved results to '{file_name}'.\033[0m")
        
    metrics_file = os.path.join(u_args['save_path'], 'metrics.pt')
    torch.save([acc_list, loss_list, comm_load_list], metrics_file)
