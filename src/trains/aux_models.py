# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from trains.server import Server

# ------------------------------------------------------------------------------
DEBUG = 1

def debug(str):
    if DEBUG: print(str)

# ------------------------------------------------------------------------------
class AuxiliaryModel(ABC, nn.Module):

    def __init__(self, server: Server, device='cpu'):
        '''Base class for Auxiliary models.
        
        Params
        ------
            server - class containing (model, optimizer, etc.) corresponding to
                the server
            device - torch device to use

        Attributes
        ----------
            data_x - Each entry consists of cut-layer activations saved at each
                alignment round.
            data_y - Dataset of cut-layer gradients returned by the server
        '''
        super(AuxiliaryModel, self).__init__()
        self.data_x = torch.tensor([], device=device)   # dataset cut-layer activations
        self.data_labels = torch.tensor([], device=device) # labels corresponding to the data_x
        self.data_y = torch.tensor([], device=device)   # dataset cut-layer true gradients
        self.server = server

    def assert_io_shape(self, x):
        '''Check if auxiliary model indeed returns gradients of the same
        shape as the input.'''
        out = self.forward(x)
        assert out.shape == x.shape,\
            "Auxiliary model should have input and output of same shape"

    @abstractmethod
    def align(self):
        '''Align the auxiliary model given the current data.  '''
        pass

    def refresh_data(self, all=True):
        '''Refresh the current dataset with a potentially updated server model.
        
        Params
        ------
            all - optional, bool
                If set to true, refresh all the data, otherwise only refresh the
                last point.
        
        '''
        if all:
            self.server.optimizer.zero_grad()
            all_ins = self.data_x.clone().detach().requires_grad_(True)
            all_out = self.server.model(all_ins)
            all_loss = self.server.criterion(all_out, self.data_labels)
            all_loss.backward()
            self.data_y = all_ins.grad.clone().detach()
            self.server.optimizer.zero_grad()
        else:
            #TODO: Implement this part if needed
            raise Exception("Only all=True supported currently")

    def add_datapoint(self, x, label):
        '''Add the given datapoint to the dataset.
        
        Params
        ------
            x     - cut-layer activation of shape (H, W, C),
            label - labels corresponding to x.

        '''
        self.data_x = torch.cat((self.data_x, x), dim=0)
        self.data_labels = torch.cat((self.data_labels, label), dim=0).long()

    def debug_grad_nmse(self, x, labels, pre=''):
        x = x.requires_grad_(True)
        self.server.optimizer.zero_grad()
        self.server.criterion(self.server.model(x), labels).backward()
        true_grad = x.grad
        #debug(f"[DEBUG] True grad: \n{true_grad}")
        with torch.no_grad():
            approx_grad = self.forward(x).clone().detach()
            #debug(f"[DEBUG] Approx grad: \n{approx_grad}")
            assert approx_grad.shape == true_grad.shape,\
                "True and predicted gradients don't match"
            #print("True grad: ", true_grad)
            #print("Approx grad: ", approx_grad)
            nmse_grad = torch.nn.functional.mse_loss(true_grad, approx_grad, reduction='sum') / torch.sum(true_grad**2)
            mse_grad = torch.nn.functional.mse_loss(true_grad, approx_grad)
            print(f"{pre} MSE: {mse_grad:0.3e}, NMSE: {nmse_grad:0.3e}")

        self.server.optimizer.zero_grad()
        
# ------------------------------------------------------------------------------
class LinearAuxiliaryModel(AuxiliaryModel):

    def __init__(self, n_input, server, device='cpu', bias=True):
        super(LinearAuxiliaryModel, self).__init__(server, device)
        self.bias = bias
        self.fc = nn.Linear(
            in_features=n_input, out_features=n_input, bias=bias
        )

    def unc_loss(self):
        X, Y = self.data_x, self.data_y
        if self.bias:
            c = self.fc.bias.data[:, None]
            one = torch.ones((X.shape[0], 1)).to(X.device)
            loss = (1 / (2 * X.shape[0])) * torch.norm(self.fc.weight.data @ X.T
                                       + c @ one.T - Y.T, p='fro')**2
        else:
            loss = (1 / (2 * X.shape[0])) * torch.norm(self.fc.weight.data @ X.T
                                       - Y.T, p='fro')**2
        return loss
            
    def align_loss(self, lambda_reg=1e-3):
        loss = self.unc_loss()
        if self.bias: 
            c = self.fc.bias.data[:, None]
            loss += (lambda_reg / 2) * torch.norm(c)**2
        loss += (lambda_reg / 2) * torch.norm(self.fc.weight.data, p='fro')**2
        return loss

    def align(self, lambda_reg=1e-5):

        X, Y = self.data_x[:-1], self.data_y[:-1] # TODO: TMP
        debug(f"Size of alignment dataset: {X.shape[0]}")
        debug(f"Loss Before aux update: {self.unc_loss()}")
        XtX = X.T @ X + X.shape[0] * lambda_reg * torch.eye(X.shape[1]).to(X.device)
        P = torch.linalg.solve(XtX, X, left=False)
        if self.bias:
            one = torch.ones((X.shape[0], 1)).to(X.device)
            q = P @ X.T @ one
            c = Y.T @ (one - q) / (X.shape[0] * (1 + lambda_reg) - one.T @ q)
            W = (Y.T - c @ one.T) @ P
            c = torch.flatten(c)
            self.fc.bias.data = c
        else:
            W = Y.T @ P
            c = None
        self.fc.weight.data = W
        debug(f"Loss After aux update: {self.unc_loss()}")
        self.debug_grad_nmse(self.data_x[-1].view(1, *self.data_x[-1].shape), self.data_labels[-1].view(1,), pre="On unseen data --")

    def forward(self, x):
        return self.fc(x)

# ------------------------------------------------------------------------------
class NNAuxiliaryModel(AuxiliaryModel):
    def __init__(self,
        n_input, server, device='cpu', n_hidden=None,
        align_iters=5, align_step=1e-3, 
    ):
        super(NNAuxiliaryModel, self).__init__(server, device)
        if n_hidden is None:
            n_hidden = 2 * n_input
        self.fc1 = nn.Linear(
            in_features=n_input, out_features=n_hidden, bias=True
        )
        self.fc2 = nn.Linear(
            in_features=n_hidden, out_features=n_input, bias=True
        )
        self.align_iters = align_iters
        self.align_step = align_step

    def unc_loss(self):
        loss = F.mse_loss(self.forward(self.data_x), self.data_y)
        return loss
            
    def align_loss(self, lambda_reg=1e-3):
        # TODO: Implement regularization if needed
        loss = self.unc_loss()
        return loss

    def align(self, lambda_reg=1e-1):
        debug(f"Loss Before aux update: {self.unc_loss()}")
        optimizer = optim.Adam(self.parameters(), lr=self.align_step)

        for i in range(self.align_iters):
            optimizer.zero_grad()
            loss = self.align_loss(lambda_reg)
            debug(f" --- Iter {i}, Loss {loss}")
            loss.backward()
            optimizer.step()
        debug(f"Loss After aux update: {self.unc_loss()}")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# ------------------------------------------------------------------------------