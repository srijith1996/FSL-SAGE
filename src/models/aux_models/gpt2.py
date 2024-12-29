# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gpt2 import *

from models import aux_models

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
            server, device, align_epochs, align_batch_size, max_dataset_size
        )
        self.transformer = GPT2AuxiliaryModel(config)
        self.lm_head = GPT2LMHead(decoder_weights, config)
        self.apply(self._init_weights)
        self.set_optimizer(align_step)

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