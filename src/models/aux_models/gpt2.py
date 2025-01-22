# ------------------------------------------------------------------------------
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
    def __init__(self, aux_inputs_list, max_size):
        super(GPT2AuxBatchHandler, self).__init__(aux_inputs_list, max_size)

        assert len(aux_inputs_list) == 4, \
            f"Expected 4 aux inputs for GPT2 model, got {len(aux_inputs_list)}"
        self.presents, self.input_shape, self.past, self.len_past = \
            tuple(aux_inputs_list)

    def __update_kv_list(self, orig_list, list):
        return [torch.cat((p, in_p), dim=1) for p, in_p in zip(orig_list, list)]

    def __update_presents(self, presents_batch):
        '''Presents consists of a list of (key, value) tensors stacked along the
        0th axis for each attention block.'''
        self.presents = self.__update_kv_list(self.presents, presents_batch)

    def __update_input_shape(self, input_shape):
        '''Input shape records the shape of the inputs that generated the
        current set of outputs.  We need to simply increment the batch
        dimension, dim=0, of our recorded input_shape with the size of the
        incoming batch. '''
        assert self.input_shape[1:] == input_shape[1:], \
            f"Dimensions of input shape other than the batch dimension, " +\
                f"dim=0, should match"
        new_batch_size = self.input_shape[0] + input_shape[0]
        self.input_shape = torch.Size([new_batch_size, *self.input_shape[1:]])

    def __update_past(self, past):
        assert isinstance(past, list)
        if past[0] is not None and self.past[0] is not None:
            self.past = self.__update_kv_list(self.past, past)

    def __update_len_past(self, len_past):
        assert len_past == self.len_past, \
            f"I don't know how to update a different value, {len_past}," + \
                f"for len_past which is currently {self.len_past}"

    def __getitem__(self, idx):
        pres = [p[:, idx, ...] for p in self.presents]
        in_shp = torch.Size([1, *self.input_shape[1:]])
        past = self.past if self.past[0] is None else \
            [p[:, idx, ...] for p in self.past]
        len_past = self.len_past
        return pres, in_shp, past, len_past

    def update(self, aux_list):
        present, input_shape, past, len_past = tuple(aux_list)
        self.__update_presents(present)
        self.__update_input_shape(input_shape)
        self.__update_past(past)
        self.__update_len_past(len_past)

    def maintain_size(self):
        if self.presents[0].shape[1] <= self.max_size:
            #logging.info(f"Skipping, current presents len {self.presents[0].shape[1]}")
            return # no need to truncate

        #logging.info("About to truncate dataset.  Init sizes----------")
        #logging.info(f"Presents: {self.presents[0].shape[1]}")
        #logging.info(f"Input shape: {self.input_shape}")
        #logging.info(f"Past: {self.past}")
        #logging.info(f"Past len: {self.len_past}")

        self.presents = [p[:, -self.max_size:, ...] for p in self.presents]
        new_batch_size = min(self.max_size, self.input_shape[0])
        self.input_shape = torch.Size([new_batch_size, *self.input_shape[1:]])
        if self.past[0] is not None:
            self.past = [p[:, -self.max_size:, ...] for p in self.past]

        #logging.info("Maintained sizes at ----------")
        #logging.info(f"Presents: {self.presents[0].shape[1]}")
        #logging.info(f"Input shape: {self.input_shape}")

    def collate(self, aux_inputs_list):
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