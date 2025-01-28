# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Type, Union
import numpy as np
import torch
import logging
from torch.utils.data import Dataset

# ------------------------------------------------------------------------------
class AuxBatchHandler(ABC):

    def __init__(self, aux_inputs_list, max_size):
        self.__check_store_aux_and_types(aux_inputs_list)
        self.max_size = max_size

    def __check_store_aux_and_types(self, aux_inputs_list):
        self.aux_input_types = []
        self.allowed_types = Union[
            tuple, list, np.ndarray, torch.Tensor, torch.Size, None
        ]
        for i, aux in enumerate(aux_inputs_list):
            if not isinstance(aux, self.allowed_types):
                raise Exception(
                    f"Unknown type {type(aux)} for auxiliary input at " +
                    f"position {i}.  Expected one of tuple, list, " +
                    f"torch.Tensor or torch.Size"
                )
            self.aux_input_types.append(type(aux))
        self.aux_inputs_list = aux_inputs_list

    def verify_aux_inputs(self, aux_inputs_list):
        assert len(aux_inputs_list) == len(self.aux_inputs_list), \
            f"Length of provided auxiliary inputs {len(aux_inputs_list)} " + \
            f"doesn't match the initialized length {len(self.aux_inputs_list)}"

        for i, aux_in in enumerate(aux_inputs_list):
            assert type(aux_in) == self.aux_input_types[i], \
                f"type of specified aux input at {i}, {type(aux_in)}, " + \
                f"doesn't match the initialized type {self.aux_input_types[i]}"

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def update(self, aux_list):
        pass

    @abstractmethod
    def collate(self, aux_inputs_list):
        pass

# ------------------------------------------------------------------------------
class RefreshDataset(Dataset):
    inputs                  : torch.Tensor
    labels                  : torch.Tensor
    index                   : int
    full                    : bool
    max_data_size           : int
    aux_batch_handler_class : Type[AuxBatchHandler]
    aux_batch_handler       : AuxBatchHandler

    def __init__(self,
        inputs, labels, max_dataset_size, aux_inputs_list=[],
        aux_batch_handler_class: Type[AuxBatchHandler]=None
    ):

        self.inputs = torch.empty((max_dataset_size, *inputs.shape[1:])
                                  ).to(inputs.device)
        self.labels = torch.empty((max_dataset_size, *labels.shape[1:])
                                  , dtype=int).to(labels.device)
        self.index = 0
        self.full = False

        if len(aux_inputs_list) > 0:
            assert aux_batch_handler_class is not None, \
                f"A RefreshDataset object requires an auxiliary batch handler" \
                    + f" class if the aux_inputs_list is not empty"
            self.aux_batch_handler_class = aux_batch_handler_class
            self.aux_batch_handler = aux_batch_handler_class(
                aux_inputs_list, max_dataset_size
            )
        else:
            self.aux_batch_handler_class = None
            self.aux_batch_handler = None

        self.max_data_size = max_dataset_size

        self.update(inputs, labels, aux_inputs_list)

    def __len__(self):
        return self.max_data_size if self.full else self.index

    def __getitem__(self, idx):
        if idx >= self.max_data_size:
            raise IndexError(
                f"Illegal index {idx} for dataset of size {self.max_data_size}"
            )

        aux_inputs = self.aux_batch_handler[idx] \
            if self.aux_batch_handler is not None else []
        return self.inputs[idx], self.labels[idx], *aux_inputs

    def update(self, inputs, labels, aux_inputs_list=[]):
        with torch.no_grad():
            batch_size = inputs.shape[0]
        
            # Check if the new batch fits without exceeding the buffer
            if self.index + batch_size <= self.max_data_size:
                self.inputs[self.index:(self.index + batch_size)].copy_(inputs)
                self.labels[self.index:(self.index + batch_size)].copy_(labels)
                self.index = (self.index + batch_size) % self.max_data_size
                # the modulo prevents the case where index = max_dataset_size
                if self.index == 0: self.full = True

            # if the current batch is smaller than the max size
            elif batch_size <= self.max_data_size:
                remaining_space = self.max_data_size - self.index
                self.inputs[self.index:].copy_(inputs[:remaining_space])
                self.inputs[:(batch_size - remaining_space)].copy_(inputs[remaining_space:])
                self.labels[self.index:].copy_(labels[:remaining_space])
                self.labels[:(batch_size - remaining_space)].copy_(labels[remaining_space:])
                self.index = (self.index + batch_size) % self.max_data_size
                self.full = True

            # if the current batch is larger than the dataset size
            else:
                self.inputs.copy_(inputs[:self.max_data_size])
                self.labels.copy_(inputs[:self.max_data_size])
                self.index = 0
                self.full = True

            if len(aux_inputs_list) > 0:
                self.aux_batch_handler.update(aux_inputs_list)

    def collate_fn(self, batch):
        inputs, labels, *aux_inputs_list = zip(*batch)

        # convert all tensors to batches
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)

        proc_aux_ins = self.aux_batch_handler_class.collate(aux_inputs_list) \
            if len(aux_inputs_list) > 0 else []

        return (inputs, labels, *proc_aux_ins)

# ------------------------------------------------------------------------------
class AlignDataset(Dataset):
    def __init__(self, refresh_dataset: RefreshDataset, outputs: torch.Tensor):
        self.refresh_dataset = refresh_dataset
        self.outputs = torch.empty_like(refresh_dataset.inputs)
        self.outputs[:outputs.shape[0]].copy_(outputs)

    def __len__(self):
        return len(self.refresh_dataset)

    def __getitem__(self, idx):
        ref_dat = self.refresh_dataset[idx]
        aux_ins = ref_dat[2:] if len(ref_dat) > 2 else []
        return ref_dat[0], self.outputs[idx], ref_dat[1], *aux_ins

    def update_outs(self, outs):
        self.outputs[:outs.shape[0]].copy_(outs)
        return self

    def collate_fn(self, batch):
        inputs, outputs, labels, *aux_inputs_list = zip(*batch)

        # convert all tensors to batches
        inputs = torch.stack(inputs)
        outputs = torch.stack(outputs)
        labels = torch.stack(labels)

        proc_aux_ins = self.refresh_dataset.aux_batch_handler.collate(
            aux_inputs_list
        ) if len(aux_inputs_list) > 0 else []

        return (inputs, outputs, labels, *proc_aux_ins)

# ------------------------------------------------------------------------------