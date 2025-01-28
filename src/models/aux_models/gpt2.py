# ------------------------------------------------------------------------------
from typing import Union, List
import sys, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gpt2 import *

from models import aux_models
from models.aux_models.ds_utils import AuxBatchHandler

# ------------------------------------------------------------------------------
class GPT2AuxiliaryModel(nn.Module):
    def __init__(self, config):
        super(GPT2AuxiliaryModel, self).__init__()
        self.n_layer = config.n_auxiliary_layer

        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(self.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.config = config

    def forward(self, 
        hidden_states,
        presents,
        input_shape,
        past=None, 
        len_past=None
    ):
        if past is None: past = [None] * len(self.h)

        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(
                hidden_states, layer_past=layer_past, len_past=len_past
            )
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        return hidden_states.view(*output_shape), presents

# ------------------------------------------------------------------------------
@aux_models.register_auxiliary_model('gpt2')
class GPT2LMAuxiliaryModel(aux_models.GradScalarAuxiliaryModel):

    def __init__(self, server, decoder_weights, *args, device='cpu', **kwargs):
        align_epochs = kwargs.pop('align_epochs', None)
        align_step = kwargs.pop('align_step', None)
        align_batch_size = kwargs.pop('align_batch_size', None)
        max_dataset_size = kwargs.pop('max_dataset_size', None)

        config = GPT2Config(*args, **kwargs)
        super(GPT2LMAuxiliaryModel, self).__init__(
            server, device, align_epochs, align_batch_size, max_dataset_size,
            aux_batch_handler_class=GPT2AuxBatchHandler
        )
        self.transformer = GPT2AuxiliaryModel(config)
        self.lm_head = GPT2LMHead(decoder_weights, config)
        self.apply(self._init_weights)

    def forward_inner(self, *args, **kwargs):
        # args : hidden_states, presents, input_shape, past, past_len
        hidden_states, presents = self.transformer(*args, **kwargs)

        # batch, seq, vocab
        lm_logits = self.lm_head(hidden_states)
        return lm_logits, presents

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

# ------------------------------------------------------------------------------
class GPT2AuxBatchHandler(AuxBatchHandler):
    presents    : List[torch.Tensor]
    input_shape : torch.Size
    past        : List[Union[None, torch.Tensor]]
    len_past    : Union[None, int]
    size        : int
    index       : int
    max_size    : int

    def __init__(self, aux_inputs_list, max_size):
        super(GPT2AuxBatchHandler, self).__init__(aux_inputs_list, max_size)

        assert len(aux_inputs_list) == 4, \
            f"Expected 4 aux inputs for GPT2 model, got {len(aux_inputs_list)}"

        presents, _, past, _ = tuple(aux_inputs_list)
        self.presents = [
            torch.empty((p.shape[0], max_size, *p.shape[2:])
                        ).to(presents[0].device)
            for p in presents
        ]
        self.past = past if past[0] is None else [
            torch.empty((p.shape[0], max_size, *p.shape[2:])
                        ).to(past[0].device) for p in past
        ]

        self.max_size = max_size
        self.full = False
        self.index = 0
        self.len_past = aux_inputs_list[3]
        self.update(aux_inputs_list)

    def __update_non_exhaustive(self,
        batch_size, presents, input_shape, past, len_past
    ):
        for i, p in enumerate(presents):
            self.presents[i][:, self.index:(self.index + batch_size), ...].copy_(p)

        if self.past[0] is not None:
            for i, p in enumerate(past):
                self.past[i][:, self.index:(self.index + batch_size), ...].copy_(p)
 
        self.input_shape = torch.Size(
            [self.index + batch_size, *input_shape[1:]]
        )
        self.index = (self.index + batch_size) % self.max_size
        if self.index == 0: self.full = True

    def __update_exhaustive(self,
        batch_size, presents, input_shape, past, len_past
    ):
        remaining_space = self.max_size - self.index

        for i, p in enumerate(presents):
            self.presents[i][:, self.index:, ...].copy_(p[:, :remaining_space, ...])
            self.presents[i][:, :(batch_size-remaining_space), ...].copy_(
                p[:, remaining_space:, ...]
            )

        if self.past[0] is not None:
            for i, p in enumerate(past):
                self.past[i][:, self.index:, ...].copy_(p[:, :remaining_space, ...])
                self.past[i][:, :(batch_size-remaining_space), ...].copy_(
                    p[:, remaining_space:, ...]
                )

        self.input_shape = torch.Size(
            [self.max_size, *self.input_shape[1:]]
        )
        self.index = (self.index + batch_size) % self.max_size
        self.full = True

    def __update_bigger_batch(self,
        presents, input_shape, past, len_past
    ):
        for i, p in enumerate(presents):
            self.presents[i].copy_(p[:, :self.max_size, ...])

        if self.past[0] is not None:
            for i, p in enumerate(past):
                self.past[i].copy_(p[:, :self.max_size, ...])

        self.input_shape = torch.Size([
            self.max_size, *list(input_shape[1:])
        ])
        self.index = 0
        self.full = True

    def update(self, aux_list):
        presents, input_shape, past, len_past = tuple(aux_list)
        batch_size = presents[0].shape[1]

        if self.index + batch_size <= self.max_size:
            self.__update_non_exhaustive(
                batch_size, presents, input_shape, past, len_past
            )

        # if the current batch is smaller than the max size
        elif batch_size <= self.max_size:
            self.__update_exhaustive(
                batch_size, presents, input_shape, past, len_past
            )

        # if the current batch is larger than the dataset size
        else:
            self.__update_bigger_batch(
                presents, input_shape, past, len_past
            )

    def __len__(self):
        return self.max_size if self.full else self.index

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise IndexError(
                f"Illegal index {idx} for dataset of size {self.__len__()}"
            )

        pres = [p[:, idx, ...] for p in self.presents]
        in_shp = torch.Size([1, *self.input_shape[1:]])
        past = self.past if self.past[0] is None else \
            [p[:, idx, ...] for p in self.past]
        len_past = self.len_past
        return pres, in_shp, past, len_past

    @staticmethod
    def collate(aux_inputs_list):
        presents_list, input_shape_list, past_list, len_past_list = \
            tuple(aux_inputs_list)

        assert len(presents_list) == len(input_shape_list)
        assert len(presents_list) == len(past_list)
        assert len(presents_list) == len(len_past_list)

        #logging.info(f"# of collated items: {len(presents_list)}")
        #logging.info(f"# of one presents list: {len(presents_list[0])}")
        #logging.info(f"Shape of individual present: {presents_list[0][0].shape}")

        stack_kv_list = lambda x: list(torch.unbind(torch.stack(
            [torch.stack(p, dim=0) for p in x], dim=2
        )))

        # list of (stack_dim, 2, 1, ., ., .) with `batch` entries
        presents = stack_kv_list(presents_list)
        #logging.info(f"Length of collated presents: {len(presents)}")
        #logging.info(f"Shape of collated presents: {presents[0].shape}")

        input_shape = torch.Size(
            [len(input_shape_list), *input_shape_list[0][1:]]
        )

        pasts = stack_kv_list(past_list) if past_list[0][0] is not None else \
            past_list[0]
        len_past = len_past_list[0]

        return presents, input_shape, pasts, len_past

# ------------------------------------------------------------------------------