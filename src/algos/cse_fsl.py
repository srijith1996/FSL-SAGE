# ------------------------------------------------------------------------------
import copy
import torch
from utils.utils import calculate_load
from algos import (
    register_algorithm, aggregate_models, FLAlgorithm,
    get_auxlist_memory_consumption
)
from models.aux_models import AuxiliaryModel

# ------------------------------------------------------------------------------
@register_algorithm("cse_fsl")
class CSEFSL(FLAlgorithm):
    aggregated_auxiliary : AuxiliaryModel

    def __init__(self, *args, **kwargs):
        super(CSEFSL, self).__init__(*args, **kwargs)

        self.server_update_interval = self.cfg.server_update_interval
        self.iters_per_epoch = [len(c.train_loader) for c in self.clients]

        self.aggregated_auxiliary = copy.deepcopy(self.clients[0].auxiliary_model)

    def client_model(self):
        return self.aggregated_client

    def server_model(self):
        return self.server.model

    def special_models_train_mode(self, t):
        if t > 0: self.aggregated_auxiliary.train()
        for c in self.clients: c.auxiliary_model.train()

    def special_models_eval_mode(self):
        self.aggregated_auxiliary.eval()
        for c in self.clients: c.auxiliary_model.eval()

    def client_step(self, rd_cl_ep_it, x, y, *args):

        t, i, j, k = rd_cl_ep_it       # (round, client, epoch, iter)

        self.clients[i].optimizer.zero_grad()
        self.clients[i].auxiliary_model.optimizer.zero_grad()

        # client feedforward
        splitting_outputs = self.clients[i].model(x)
        splitting_output = splitting_outputs[0] \
            if isinstance(splitting_outputs, tuple) else splitting_outputs

        local_smashed_data = splitting_output.clone(
            ).detach().requires_grad_(True)
        smashed_data = splitting_output.clone().detach().requires_grad_(False)

        # client backpropagation and update client-side model weights
        aux_ins_cond = (
            isinstance(splitting_outputs, tuple) and len(splitting_outputs) > 1
        )
        aux_ins = splitting_outputs[1:] if aux_ins_cond else []
        out = self.clients[i].auxiliary_model.forward_inner(
            local_smashed_data, *aux_ins
        )
        out = out[0] if isinstance(out, tuple) else out
        loss = self.criterion(out, y, *args)
        loss.backward()

        ret_dict = dict()
        with torch.no_grad():
            local_loss = loss.item()
            #_, predicted = torch.max(out.data, 1)
            #local_correct = predicted.eq(y.view_as(predicted)).sum().item()
            ret_dict['l_loss'] = local_loss
            #ret_dict['l_acc'] = local_correct / y.size(dim=0)

        self.clients[i].auxiliary_model.optimizer.step()
        splitting_output.backward(local_smashed_data.grad)
        self.clients[i].optimizer.step()

        # server model update
        local_iter = j * self.iters_per_epoch[i] + k
        if local_iter % self.server_update_interval == 0:
            self.comm_load += smashed_data.numel() * \
                smashed_data.element_size()
            if aux_ins_cond:
                self.comm_load += get_auxlist_memory_consumption(
                    splitting_outputs[1:]
                )

            self.server.optimizer.zero_grad()
            out = self.server.model(smashed_data, *aux_ins)
            out = out[0] if isinstance(out, tuple) else out
            s_loss = self.criterion(out, y, *args)

            with torch.no_grad():
                global_loss = s_loss.item()
                #_, predicted = torch.max(out.data, 1)
                #global_correct = predicted.eq(y.view_as(predicted)).sum().item()
                ret_dict['g_loss'] = global_loss
                #ret_dict['g_acc'] = global_correct / y.size(dim=0)

            s_loss.backward()
            self.server.optimizer.step()

            self.server_updated = True      # for lr scheduler step
        else:
            self.server_updated = False     # for lr scheduler step

        return ret_dict

    def aggregate(self):
        self.aggregate_clients()

        self.aggregated_auxiliary = aggregate_models(
            [c.auxiliary_model for c in self.clients],
            self.agg_factor, self.device
        )
        agg_weights = self.aggregated_auxiliary.state_dict()

        for c in self.clients:
            c.auxiliary_model.load_state_dict(agg_weights)
            self.comm_load += 2 * calculate_load(c.auxiliary_model)

# ------------------------------------------------------------------------------