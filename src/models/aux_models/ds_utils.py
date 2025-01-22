# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Type, Union
import numpy as np
import torch
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
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def update(self, aux_list):
        pass

    @abstractmethod
    def maintain_size(self):
        pass

    @abstractmethod
    def collate(self, aux_inputs_list):
        pass

# ------------------------------------------------------------------------------
class RefreshDataset(Dataset):
    def __init__(self,
        inputs, labels, max_dataset_size, aux_inputs_list=[],
        aux_batch_handler_class: Type[AuxBatchHandler]=None
    ):

        self.inputs = inputs
        self.labels = labels
        if len(aux_inputs_list) > 0:
            assert aux_batch_handler_class is not None, \
                f"A RefreshDataset object requires an auxiliary batch handler" \
                    + f"class if the aux_inputs_list is not empty"
            self.aux_batch_handler = aux_batch_handler_class(
                aux_inputs_list, max_dataset_size
            )
        else:
            self.aux_batch_handler = None
        self.max_data_size = max_dataset_size
        self.__maintain_size()

    def __maintain_size(self):
        self.inputs = self.inputs[-self.max_data_size:]
        self.labels = self.labels[-self.max_data_size:]
        if self.aux_batch_handler is not None:
            self.aux_batch_handler.maintain_size()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        aux_inputs = self.aux_batch_handler[idx] \
            if self.aux_batch_handler is not None else []
        return self.inputs[idx], self.labels[idx], *aux_inputs

    def update(self, inputs, labels, aux_inputs_list):
        '''Append an existing batch of data (inputs, labels, auxiliary_inputs)
        
        Note: To simplify things, we will next assume that if aux_inputs_list is
        not empty it contains the auxiliary inputs of the gpt2 server modeis not
        empty it contains the auxiliary inputs expected by the gpt2 server
        model.'''

        self.inputs = torch.cat((self.inputs, inputs), axis=0)
        self.labels = torch.cat((self.labels, labels), axis=0)
        if len(aux_inputs_list) > 0:
            self.aux_batch_handler.verify_aux_inputs(aux_inputs_list)
            self.aux_batch_handler.update(aux_inputs_list)
        self.__maintain_size()

    def collate_fn(self, batch):
        inputs, labels, *aux_inputs_list = zip(*batch)

        # convert all tensors to batches
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)

        proc_aux_ins = self.aux_batch_handler.collate(aux_inputs_list) \
            if len(aux_inputs_list) > 0 else []

        return (inputs, labels, *proc_aux_ins)

# ------------------------------------------------------------------------------
class AlignDataset(Dataset):
    def __init__(self, refresh_dataset: RefreshDataset, outputs: torch.Tensor):
        self.refresh_dataset = refresh_dataset
        self.outputs = outputs

    def __len__(self):
        return len(self.refresh_dataset)

    def __getitem__(self, idx):
        ref_dat = self.refresh_dataset[idx]
        aux_ins = ref_dat[2:] if len(ref_dat) > 2 else []
        return ref_dat[0], self.outputs[idx], ref_dat[1], *aux_ins

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