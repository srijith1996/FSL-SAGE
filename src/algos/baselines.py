# ------------------------------------------------------------------------------
import time
import copy
import torch.nn as nn
import torch
from torch import optim

from algos import register_algorithm, aggregate_models, FLAlgorithm
from models import config_optimizer, config_lr_scheduler

# ------------------------------------------------------------------------------
@register_algorithm("fed_avg")
class FedAvg(FLAlgorithm):

    def __init__(self,
        *args, **kwargs 
    ):
        super(FedAvg, self).__init__(*args, **kwargs)

        # merge client and server models into one
        for c in self.clients:
            c.model = nn.Sequential(c.model, self.server.model)
            c.optimizer = config_optimizer(
                c.model.parameters(), c.optimizer_options
            )
            c.lr_scheduler = config_lr_scheduler(
                c.optimizer, c.lr_scheduler_options
            )

    def full_model(self, x):
        return self.aggregated_client(x)

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it
        t0 = time.time()
        self.clients[i].optimizer.zero_grad()
        out = self.clients[i].model(x)
        loss = self.criterion(out, y)
        t1 = time.time()

        with torch.no_grad():
            train_loss = loss.item()
            _, predicted = torch.max(out.data, 1)
            train_correct = predicted.eq(y.view_as(predicted)).sum().item()

        t2 = time.time()
        loss.backward()
        self.clients[i].optimizer.step()
        t_c = (time.time() - t2 + t1 - t0)
        return {
            'acc' : train_correct / y.size(dim=0),
            'loss': train_loss,
            'client_model_compute_time' : t_c
        }
    
# ------------------------------------------------------------------------------
@register_algorithm("sl_single_server")
class SplitFedv2(FLAlgorithm):

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        t0_c = time.time()
        self.clients[i].optimizer.zero_grad()

        # pass smashed data through full model 
        splitting_output = self.clients[i].model(x)
        t1_c = time.time()

        # Represents the uploaded data
        smashed_data = splitting_output.clone().detach().requires_grad_(True)

        # Comm cost for upload splitting output to server
        self.comm_load += smashed_data.numel() * smashed_data.element_size() 

        t0_s = time.time()
        self.server.optimizer.zero_grad()
        output = self.server.model(smashed_data) 
        loss = self.server.criterion(output, y)
        t1_s = time.time()

        with torch.no_grad():
            train_loss = loss.item()
            _, predicted = torch.max(output.data, 1)
            train_correct = predicted.eq(y.view_as(predicted)).sum().item()

        t2_s = time.time()
        loss.backward()
        self.server.optimizer.step()
        t_s = time.time() - t2_s + t1_s - t0_s

        # Comm cost for downloading grads of smashed data
        self.comm_load += smashed_data.grad.numel() * smashed_data.grad.element_size()

        # Backprop split output with smashed data grad
        t2_c = time.time()
        splitting_output.backward(smashed_data.grad)
        self.clients[i].optimizer.step()
        t_c = time.time() - t2_c + t1_c - t0_c

        return {
            'acc' : train_correct / y.size(dim=0),
            'loss': train_loss,
            'client_model_compute_time' : t_c,
            'server_model_compute_time' : t_s
        }

# ------------------------------------------------------------------------------
@register_algorithm("sl_multi_server")
class SplitFedv1(FLAlgorithm):
    aggregated_server : nn.Module

    def __init__(self, *args, **kwargs):
        super(SplitFedv1, self).__init__(*args, **kwargs)

        # split server model
        self.servers = []
        for c in self.clients:
            self.servers.append(copy.deepcopy(self.server))

    def full_model(self, x):
        return self.aggregated_server(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        if t > 0: self.aggregated_server.train()

    def special_models_eval_mode(self):
        self.aggregated_server.eval()

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        t0_c = time.time()
        self.clients[i].optimizer.zero_grad()

        # pass smashed data through full model 
        splitting_output = self.clients[i].model(x)
        t1_c = time.time()

        # Represents the uploaded data
        smashed_data = splitting_output.clone().detach().requires_grad_(True)

        # Upload the smashed data to the server
        self.comm_load += smashed_data.numel() * smashed_data.element_size() 

        t0_s = time.time()
        self.servers[i].optimizer.zero_grad()
        output = self.servers[i].model(smashed_data) 
        loss = self.criterion(output, y)
        t1_s = time.time()

        with torch.no_grad():
            train_loss = loss.item()
            _, predicted = torch.max(output.data, 1)
            train_correct = predicted.eq(y.view_as(predicted)).sum().item()

        t2_s = time.time()
        loss.backward()
        self.servers[i].optimizer.step()
        t_s = time.time() - t2_s + t1_s - t0_s

        # Download gradients of the smashed data
        self.comm_load += smashed_data.grad.numel() * smashed_data.grad.element_size() 

        # Backprop grads back to splitting_output
        t2_c = time.time()
        splitting_output.backward(smashed_data.grad)
        self.clients[i].optimizer.step()
        t_c = time.time() - t2_c + t1_c - t0_c

        return {
            'acc' : train_correct / y.size(dim=0),
            'loss': train_loss,
            'client_model_compute_time' : t_c,
            'server_model_compute_time' : t_s
        }

    def aggregate(self):
        ret_dict = self.aggregate_clients()

        t0 = time.time()
        self.aggregated_server = aggregate_models(
            [s.model for s in self.servers], self.agg_factor, self.device
        )
        agg_weights = self.aggregated_server.state_dict()

        for s in self.servers:
            s.model.load_state_dict(agg_weights)

        ret_dict['server_agg_compute_time'] = time.time() - t0
        return ret_dict
        
# ------------------------------------------------------------------------------