# ------------------------------------------------------------------------------
import logging
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from models.aux_models import AuxiliaryModel, GradScalarAuxiliaryModel
from models.aux_models import register_auxiliary_model

# ------------------------------------------------------------------------------
@register_auxiliary_model("linear")
class LinearAuxiliaryModel(AuxiliaryModel):

    def __init__(self, n_input, server, device='cpu', bias=True):
        super(LinearAuxiliaryModel, self).__init__(server, device)
        self.bias = bias
        self.fc = nn.Linear(
            in_features=n_input, out_features=(n_input-1), bias=bias
        )

        for m in self.parameters():
            nn.init.normal_(m, mean=0., std=0.05)
            nn.init.normal_(m, mean=0., std=0.05)

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
@register_auxiliary_model("2_layer_nn")
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

        for m in self.parameters():
            nn.init.normal_(m, mean=0., std=0.05)
            nn.init.normal_(m, mean=0., std=0.05)

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
@register_auxiliary_model("2_layer_nn_grad_scalar")
class NNGradScalarAuxiliaryModel(GradScalarAuxiliaryModel):
    def __init__(self,
        n_input, n_output, server, device='cpu', n_hidden=None,
        align_epochs=5, align_step=1e-3, align_batch_size=100,
        max_dataset_size=1000
    ):
        super(NNGradScalarAuxiliaryModel, self).__init__(
            server, device, align_epochs, align_batch_size, max_dataset_size
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

        for m in self.parameters():
            nn.init.normal_(m, mean=0., std=0.05)
            nn.init.normal_(m, mean=0., std=0.05)

    def forward_inner(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.olayer(x), dim=1)
        return x

# ------------------------------------------------------------------------------
@register_auxiliary_model("linear_grad_scalar")
class LinearGradScalarAuxiliaryModel(GradScalarAuxiliaryModel):
    def __init__(self,
        n_input, n_output, server, device='cpu',
        align_epochs=5, align_step=1e-3, align_batch_size=100,
        max_dataset_size=1000
    ):
        super(LinearGradScalarAuxiliaryModel, self).__init__(
            server, device, align_epochs, align_batch_size, max_dataset_size
        )

        self.fc = nn.Linear(
            in_features=n_input, out_features=n_output, bias=True
        )
        self.align_epochs = align_epochs

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0., std=0.05)
            nn.init.normal_(module.bias, mean=0., std=0.05)
            #if module.bias is not None:
            #    nn.init.zeros_(module.bias)

    def forward_inner(self, x):
        x = F.log_softmax(self.fc(x), dim=1)
        return x
    
    #def forward(self, x, label):
    #    x.requires_grad_(True)
    #    x = self.server.criterion(self.forward_inner(x), label)
    #    
    #    return x
 
# ------------------------------------------------------------------------------