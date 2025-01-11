# ------------------------------------------------------------------------------
import torch
from utils.utils import calculate_load
from algos import register_algorithm, FLAlgorithm

# ------------------------------------------------------------------------------
@register_algorithm("fsl_sage")
class FSLSAGE(FLAlgorithm):

    def __init__(self, *args, **kwargs):
        super(FSLSAGE, self).__init__(*args, **kwargs)

        self.server_update_interval = self.cfg.server_update_interval
        self.align_interval = self.cfg.align_interval
        self.iters_per_epoch = [len(c.train_loader) for c in self.clients]

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def client_step(self, rd_cl_ep_it, x, y):

        t, i, j, k = rd_cl_ep_it       # (round, client, epoch, iter)

        self.clients[i].optimizer.zero_grad()

        # client feedforward
        splitting_output = self.clients[i].model(x)
        local_smashed_data = splitting_output.clone().detach().requires_grad_(True)

        # server model update
        local_iter = j * self.iters_per_epoch[i] + k
        ret_dict = dict()
        if local_iter % self.server_update_interval == 0:
            smashed_data = splitting_output.clone().detach().requires_grad_(True)
            self.comm_load += smashed_data.numel() * smashed_data.element_size()

            self.server.optimizer.zero_grad()
            out = self.server.model(smashed_data)
            s_loss = self.criterion(out, y)

            with torch.no_grad():
                train_loss = s_loss.item()
                _, predicted = torch.max(out.data, 1)
                train_correct = predicted.eq(y.view_as(predicted)).sum().item()
                ret_dict['g_loss'] = train_loss
                ret_dict['g_acc'] = train_correct / y.size(dim=0)

            s_loss.backward()
            self.server.optimizer.step()

            # save smashed data to memory
            self.clients[i].auxiliary_model.add_datapoint(
                splitting_output.clone().detach(), y
            )

        # perform alignment for current client
        if t % self.align_interval == 0 and local_iter == 0:
            self.clients[i].auxiliary_model.refresh_data()
            self.clients[i].auxiliary_model.align()

            # the aligned auxiliary model is sent back to client i
            self.comm_load += calculate_load(self.clients[i].auxiliary_model)

        # client backpropagation and update client-side model weights
        client_grad_approx = self.clients[i].auxiliary_model(local_smashed_data, y) 
        splitting_output.backward(client_grad_approx)
        self.clients[i].optimizer.step()

        return ret_dict

# ------------------------------------------------------------------------------