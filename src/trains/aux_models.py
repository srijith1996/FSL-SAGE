# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
#import matplotlib.pyplot as plt
from tqdm import trange
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from trains.server import Server
import logging

# ------------------------------------------------------------------------------
class AuxiliaryModel(ABC, nn.Module):

    def __init__(self,
        server: Server, device='cpu', max_dataset_size=1000,
        align_batch_size=100
    ):
        '''Base class for Auxiliary models.
        
        Params
        ------
            server - class containing (model, optimizer, etc.) corresponding to
                the server
            device - torch device to use
            max_dataset_size - Maximum number of alignment points to store
            align_batch_size - batch-size for alignment loop

        Attributes
        ----------
            data_x - Each entry consists of cut-layer activations saved at each
                alignment round.
            data_y - Dataset of cut-layer gradients returned by the server
        '''
        super(AuxiliaryModel, self).__init__()

        self.max_data_size = max_dataset_size
        self.data_x = torch.tensor([], device=device)   # dataset cut-layer activations
        self.data_labels = torch.tensor([], device=device) # labels corresponding to the data_x
        self.data_y = torch.tensor([], device=device)   # dataset cut-layer true gradients
        self.data_loader = None
        self.align_batch_size = align_batch_size
        self.server = server

    @abstractmethod
    def align(self):
        '''Align the auxiliary model given the current data.  '''
        pass

    @abstractmethod
    def forward(self, x, label):
        '''Forward pass'''
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
            logging.debug(f"Size of alignment dataset: {self.data_x.shape[0]}")
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

        self.data_loader = DataLoader(
            TensorDataset(self.data_x, self.data_y, self.data_labels),
            shuffle=True, batch_size=self.align_batch_size, pin_memory=False
        )

    def get_align_dataset(self):
        '''Get X data concatenated with the labels'''

        # TODO: may need more generalized version for multi-dimensional x
        return torch.cat((self.data_x, self.data_labels[:, None]), axis=1), self.data_y

    def get_cat_data(self, x, label):
        '''Get X data concatenated with the labels'''

        # TODO: may need more generalized version for multi-dimensional x
        return torch.cat((x, label[:, None]), axis=1)

    def add_datapoint(self, x, label):
        '''Add the given datapoint to the dataset.
        
        Params
        ------
            x     - cut-layer activation of shape (N, x_shape),
            label - labels corresponding to x.

        '''
        self.data_x = torch.cat((self.data_x, x), dim=0)
        self.data_labels = torch.cat((self.data_labels, label), dim=0).long()
        if self.data_x.shape[0] > self.max_data_size:
            self.data_x = self.data_x[-self.max_data_size:]
            self.data_labels = self.data_labels[-self.max_data_size:]

        assert self.data_x.shape[0] <= self.max_data_size,\
            "Error in <add_datapoint>: Dataset size has exceeded limit"

    def debug_grad_nmse(self, x, labels, pre=''):
        x = x.requires_grad_(True)
        self.server.optimizer.zero_grad()
        self.server.criterion(self.server.model(x), labels).backward()
        true_grad = x.grad

        #debug(f"[DEBUG] True grad: \n{true_grad}")
        approx_grad = self.forward(x, labels).clone().detach()
        #debug(f"[DEBUG] Approx grad: \n{approx_grad}")
        assert approx_grad.shape == true_grad.shape,\
                "Shape of True and predicted gradients don't match"
        #logger.debug("True grad: ", true_grad)
        #logger.debug("Approx grad: ", approx_grad)
        nmse_grad = F.mse_loss(true_grad, approx_grad, reduction='sum') / torch.sum(true_grad**2)
        mse_grad = F.mse_loss(true_grad, approx_grad)
        logging.debug(f"{pre} MSE: {mse_grad:0.3e}, NMSE: {nmse_grad:0.3e}")

        self.server.optimizer.zero_grad()
        return mse_grad, nmse_grad
        
# ------------------------------------------------------------------------------
class LinearAuxiliaryModel(AuxiliaryModel):

    def __init__(self, n_input, server, device='cpu', bias=True):
        super(LinearAuxiliaryModel, self).__init__(server, device)
        self.bias = bias
        self.fc = nn.Linear(
            in_features=n_input, out_features=(n_input-1), bias=bias
        )

    def unc_loss(self):
        X, Y = self.get_align_dataset()
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
        #if self.bias: 
        #    c = self.fc.bias.data[:, None]
        #    loss += (lambda_reg / 2) * torch.norm(c, p=1, dim=(-1, -2))**2
        loss += (lambda_reg / 2) * torch.norm(self.fc.weight.data, p=1, dim=(-1, -2))**2
        return loss

    def align(self, lambda_reg=1e-3):

        X, Y = self.get_align_dataset()
        logging.debug(f"Size of alignment dataset: {X.shape[0]}")
        logging.debug(f"Loss Before aux update: {self.unc_loss()}")
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
        logging.debug(f"Loss After aux update: {self.unc_loss()}")

    def forward(self, x, label):
        x = self.get_cat_data(x, label)
        return self.fc(x)

# ------------------------------------------------------------------------------
class NNAuxiliaryModel(AuxiliaryModel):
    def __init__(self,
        n_input, server, device='cpu', n_hidden=None,
        align_epochs=5, align_step=1e-3, 
    ):
        super(NNAuxiliaryModel, self).__init__(server, device)
        if n_hidden is None:
            n_hidden = 2 * n_input
        self.fc1 = nn.Linear(
            in_features=n_input, out_features=n_hidden, bias=True
        )
        self.fc2 = nn.Linear(
            in_features=n_hidden, out_features=(n_input-1), bias=True
        )
        self.align_epochs = align_epochs
        self.optimizer = optim.Adam(self.parameters(), lr=align_step)

    def align_loss(self, x, y):
        # TODO: Implement regularization if needed
        loss = F.mse_loss(self.forward(x), y)
        return loss

    def align(self):
        logging.debug(f"Loss Before aux update: {self.unc_loss()}")

        for i in range(self.align_epochs):
            for x, y in self.data_loader: 
                self.optimizer.zero_grad()
                loss = self.align_loss(x, y)
                logging.debug(f" --- Iter {i}, Loss {loss}")
                loss.backward()
                self.optimizer.step()
        logging.debug(f"Loss After aux update: {self.unc_loss()}")

    def forward(self, x, label):
        x = self.get_cat_data(x, label)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# ------------------------------------------------------------------------------
class GradScalarAuxiliaryModel(AuxiliaryModel):

    def __init__(self,
        server, device='cpu', align_epochs=5
    ):
        super(GradScalarAuxiliaryModel, self).__init__(server, device)
        self.align_epochs = align_epochs

    @abstractmethod
    def forward_inner(self, x):
        pass

    def set_optimizer(self, align_step):
        self.optimizer = optim.Adam(self.parameters(), lr=align_step)

    def align_loss(self, x, y, labels):
        aux_out = self.forward(x, labels)
        loss = F.mse_loss(aux_out, y, reduction='sum')
        return loss

    def align(self):
        bar = trange(self.align_epochs, desc="Alignment", disable=True)
        logging.debug(f"# batches for alignment: {len(self.data_loader)}")
        for i in bar:
            for x, y, labels in self.data_loader:
                self.optimizer.zero_grad()
                loss = self.align_loss(x, y, labels)
                loss.backward()
                self.optimizer.step()
            if i % 10 == 0:
                logging.debug(f" --- Epoch {i:4d}/{self.align_epochs}, Loss {loss:.4e}")
            bar.set_postfix(loss=loss.item())

    def forward(self, x, label):
        x.requires_grad_(True)
        out = self.server.criterion(self.forward_inner(x), label)
        aux_out = torch.autograd.grad(out, x, create_graph=True)[0]
        return aux_out

# ------------------------------------------------------------------------------
class NNGradScalarAuxiliaryModel(GradScalarAuxiliaryModel):
    def __init__(self,
        n_input, n_output, server, device='cpu', n_hidden=None,
        align_epochs=5, align_step=1e-3, 
    ):
        super(NNGradScalarAuxiliaryModel, self).__init__(
            server, device, align_epochs
        )

        if n_hidden is None:
            n_hidden = 2 * n_input
        self.fc1 = nn.Linear(
            in_features=n_input, out_features=n_hidden, bias=True
        )
        self.olayer = nn.Linear(
            in_features=n_hidden, out_features=n_output, bias=True
        )
        self.align_epochs = align_epochs

        self.set_optimizer(align_step)

    def forward_inner(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.olayer(x), dim=1)
        return x

# ------------------------------------------------------------------------------
class LinearGradScalarAuxiliaryModel(GradScalarAuxiliaryModel):
    def __init__(self,
        n_input, n_output, server, device='cpu',
        align_epochs=5, align_step=1e-3, 
    ):
        super(LinearGradScalarAuxiliaryModel, self).__init__(
            server, device, align_epochs
        )

        self.fc = nn.Linear(
            in_features=n_input, out_features=n_output, bias=True
        )
        self.align_epochs = align_epochs

        self.set_optimizer(align_step)

    def forward_inner(self, x):
        x = F.log_softmax(self.fc(x), dim=1)
        return x
    
# ------------------------------------------------------------------------------