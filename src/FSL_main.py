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
from trains import model

print("Whether we are using GPU: ", torch.cuda.is_available())
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#use_cuda = True if torch.cuda.is_available() else False
 
    
class Client():
    def __init__(self, id, train_loader, c_args):
        if c_args['dataset'] == "cifar":
            self.model = model.Client_model_cifar() 
            self.auxiliary_model = model.Auxiliary_model_cifar()
        elif c_args['dataset'] == "femnist":
            self.model = model.Client_model_femnist() 
            self.auxiliary_model = model.Auxiliary_model_femnist()
        self.criterion = nn.NLLLoss().to(DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=c_args["lr"])       
        self.auxiliary_criterion = nn.NLLLoss().to(DEVICE)
        self.auxiliary_optimizer = optim.SGD(self.auxiliary_model.parameters(), lr=c_args["lr"])
        
        self.train_loader = train_loader
        self.epochs = c_args['epoch']
        self.dataset_size = len(self.train_loader) * c_args["batch_size"] 

class Server():
    def __init__(self, c_args):
        if c_args['dataset'] == "cifar":
            self.model = model.Server_model_cifar()
        elif c_args['dataset'] == "femnist":
            self.model = model.Server_model_femnist()
        self.criterion = nn.NLLLoss().to(DEVICE)
        self.alignLoss = nn.MSELoss().to(DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=c_args["lr"])

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
    args = options.args_parser('CSE_FSL')    #---------todo
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
    server = Server(c_args)
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
    
    while r < s_args["round"]:
        # Send data samples to each client
        it_list = []
        for i in range(s_args["activated"]):
            it_list.append(iter(trainLoader_list[i]))
        batch_round = u_args['batch_round']
        max_batch = batch_round
        client_i = 0
        start_index = 0
        
        # Loop through either all activated clients or the batches distributed
        # (whichever comes first)
        while max_batch <= batch_max_round and client_i < s_args["activated"]: 
            cur_client_index = start_index
            while cur_client_index < batch_max_round and cur_client_index < max_batch: 
                # For a given client, iterate through the samples it needs to train on and send it to device
                samples, labels = next(it_list[client_i])
                client_copy_list[client_i].optimizer.zero_grad()
                samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                
                # client feedforward
                splitting_output = client_copy_list[client_i].model(samples)
                local_smashed_data = splitting_output.clone().detach().requires_grad_(True)
                smashed_data = splitting_output.clone().detach().requires_grad_(True)
                
                # Copies of the smashed data for alignment later
                smashed_clone1 = splitting_output.clone().detach().requires_grad_(True)
                smashed_clone2 = splitting_output.clone().detach().requires_grad_(True)
                    
                
                # client calculates the local loss and do the backpropagation and update auxiliary model weights
                client_copy_list[client_i].auxiliary_optimizer.zero_grad()
                local_output = client_copy_list[client_i].auxiliary_model(local_smashed_data) 
                local_loss = client_copy_list[client_i].auxiliary_criterion(local_output, labels)
                local_loss.backward()  
                client_copy_list[client_i].auxiliary_optimizer.step()

                # client backpropagation and update client-side model weights
                gradient = local_smashed_data.grad
                splitting_output.backward(gradient)
                client_copy_list[client_i].optimizer.step() 
                    
                # server feedforward, calculate loss, backpropagation and update server-side model weights 
                if cur_client_index == max_batch - 1:
                    server.optimizer.zero_grad()
                    comm_load += smashed_data.numel() * 4   # float32 = 4 bytes
                    output = server.model(smashed_data) 
                    loss = server.criterion(output, labels)
                            
                    loss.backward()       
                    server.optimizer.step() 
    
                    # Do alignment every l steps
                    if r % l == 0:  
            
                        # TODO: Check if the loss backprop on gradient diff actually changes the underlying model parameters
                        # this stores a copy of the model parameters
                        # auxClone = copy.deepcopy(client_copy_list[client_i].auxiliary_model)
                        
                        # for param in client_copy_list[client_i].auxiliary_model.parameters():
                        #     param.data = nn.parameter.Parameter(torch.ones_like(param))

                        # Following section adapted slightly from https://github.com/caogang/wgan-gp/blob/ae47a185ed2e938c39cf3eb2f06b32dc1b6a2064/gan_mnist.py#L146,
                        #   under the autograd.grad part
                        
                        # Run the sample through the auxilary model again and get the auxilary gradients
                        client_copy_list[client_i].auxiliary_optimizer.zero_grad()
                        aux_Out = client_copy_list[client_i].auxiliary_model(smashed_clone1) 
                        
                        # Get gradients of auxilary model for the cut layer
                        aux_Gradients = autograd.grad(outputs = aux_Out, inputs = smashed_clone1, 
                            grad_outputs=torch.ones(aux_Out.size()).to(DEVICE), 
                            create_graph=True, retain_graph=True)[0]
                        
                        # Run the sample through the server model to get desired server gradients
                        server.optimizer.zero_grad()
                        server_Out = server.model(smashed_clone2) 
                        
                        # Get gradients of server for the cut layer
                        server_Gradients = autograd.grad(outputs = server_Out, inputs = smashed_clone2, 
                            grad_outputs = torch.ones(server_Out.size()).to(DEVICE))[0]
                        
                        # Define a criterion to minimize the square difference of gradients at the cut layer
                        grad_MSE = torch.nn.MSELoss()

                        # Set up the loss function 
                        grad_Loss = grad_MSE(aux_Gradients, server_Gradients)
                        
                        # Calculate the gradients with respect to model weights 
                        # (In this case, we only need to find these for the auxilary model
                        # since we don't update the server during alignment
                        grad_Loss.backward()
                        
                        # Update auxilary model
                        client_copy_list[client_i].auxiliary_optimizer.step()

                        # Doesn't do anything but zero_grad is to prevent weird bugs that might show up
                        server.optimizer.zero_grad()
                        
                        ## Print out the gradients for our auxilary model (like 2nd order derivatives)
                        ## from https://discuss.pytorch.org/t/how-to-calculate-2nd-derivative-of-a-likelihood-function/15085/3
                        # for name, param in client_copy_list[client_i].auxiliary_model.named_parameters():
                        #     print(name, param.grad)
                        
                        # # Check if the gradients are the same
                        # # TODO: Remove when we figure it out
                        # for p1, p2 in zip(auxClone.parameters(), client_copy_list[client_i].auxiliary_model.parameters()):
                        #     if p1.data.ne(p2.data).sum() > 0:
                        #         print("Models are different!")
                        #         break
                                
                        #print("Models are the same")
                        # if (grad_Loss < 0.03):
                        #     print("... but Aux_grad and serv_grad are the essentially the same as well")

                cur_client_index += 1
            
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
        aggregated_client_auxiliary = copy.deepcopy(client_copy_list[0].auxiliary_model)
        aggregated_client_weights_auxiliary = aggregated_client_auxiliary.state_dict()

        for key in aggregated_client_weights:
            aggregated_client_weights[key] = client_copy_list[0].model.state_dict()[key] * factor[0]
        for key in aggregated_client_weights_auxiliary:
            aggregated_client_weights_auxiliary[key] = client_copy_list[0].auxiliary_model.state_dict()[key] * factor[0]

        for i in range(1, s_args["activated"]):
            for key in aggregated_client_weights:
                aggregated_client_weights[key] += client_copy_list[i].model.state_dict()[key] * factor[i]
            for key in aggregated_client_weights_auxiliary:
                aggregated_client_weights_auxiliary[key] += client_copy_list[i].auxiliary_model.state_dict()[key] * factor[i]
    

        # Update client model weights and auxiliary weights
        for i in range(s_args["activated"]):
            client_copy_list[i].model.load_state_dict(aggregated_client_weights)
            client_copy_list[i].auxiliary_model.load_state_dict(aggregated_client_weights_auxiliary)
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
            '\nRound {}, for the weighted aggregated final model, testing loss: {:.2f}, testing acc: {:.2f}%  ({}/{})'
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
