# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import os, importlib
from tqdm import trange
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------------------
class AuxiliaryModel(ABC, nn.Module):

    def __init__(self,
        server, device='cpu', max_dataset_size=1000,
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
class GradScalarAuxiliaryModel(AuxiliaryModel):

    def __init__(self,
        server, device='cpu', align_epochs=5, align_batch_size=100,
        max_dataset_size=1000
    ):
        super(GradScalarAuxiliaryModel, self).__init__(
            server, device, max_dataset_size, align_batch_size
        )
        self.align_epochs = align_epochs

    @abstractmethod
    def forward_inner(self, x):
        pass

    def set_optimizer_lr_scheduler(self, optimizer, lr_scheduler=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def align_loss(self, x, y, labels):
        aux_out = self.forward(x, labels)
        loss = F.mse_loss(aux_out, y, reduction='sum')
        return loss

    def align(self):
        bar = trange(self.align_epochs, desc="Alignment", leave=False)
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
            if self.lr_scheduler is not None: self.lr_scheduler.step()

    def forward(self, x, label):
        x.requires_grad_(True)
        out = self.server.criterion(self.forward_inner(x), label)
        aux_out = torch.autograd.grad(out, x, create_graph=True)[0]
        return aux_out

# ------------------------------------------------------------------------------
# name: str -> aux_model: AuxiliaryModel
AUXILIARY_MODEL_REGISTRY: Dict[str, AuxiliaryModel] = dict()

# -----------------------------------------------------------------------------
def register_auxiliary_model(name, disable_check=False):
    """Decorator to register a new model"""
    def register_model_cls(cls):
        if name in AUXILIARY_MODEL_REGISTRY:
            raise ValueError(
                'Cannot register duplicate model {}'.format(name)
            )
        if not disable_check:
            if not issubclass(cls, AuxiliaryModel):
                raise ValueError(
                    'Model {} must extend {}'.format(name, cls.__name__)
                )
        AUXILIARY_MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls

# -----------------------------------------------------------------------------