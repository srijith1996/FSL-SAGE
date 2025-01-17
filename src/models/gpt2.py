#  -----------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for
#  license information.
#  -----------------------------------------------------------------------------
import math
import copy
from dataclasses import dataclass
import torch
from torch import nn
import loralib as lora

from models import register_client_server_pair

#  -----------------------------------------------------------------------------
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    ))

#  -----------------------------------------------------------------------------
def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(
        x * 0.7978845608 * (1.0 + 0.044715 * x * x)
    ))

#  -----------------------------------------------------------------------------
def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert
        repo (identical to OpenAI GPT).  Also see
        https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
    ))

#  -----------------------------------------------------------------------------
def swish(x):
    return x * torch.sigmoid(x)

#  -----------------------------------------------------------------------------
def _gelu_python(x):
    """Original Implementation of the gelu activation function in Google Bert
        repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives
    slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 *
            torch.pow(x, 3))))
    This is now written in C in torch.nn.functional. Also see
    https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

#  -----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the
        square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

#  -----------------------------------------------------------------------------
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

#  -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF
        # implem]
        
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(
                1, 1, n_ctx, n_ctx)
        )
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = lora.MergedLinear(
            nx, n_state * 3, 
            r=config.lora_attn_dim, 
            lora_alpha=config.lora_attn_alpha, 
            lora_dropout=config.lora_dropout, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=True,
            merge_weights=False
        )
        self.c_proj = Conv1D(n_state, nx)

        self.config = config
    
    def _attn(self, q, k, v, len_kv=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        # q : (batch, head, q_seq_length, head_features)
        # k : (batch, head, head_features, kv_seq_length)
        # w : (batch, head, q_seq_length, kv_seq_length)
        # v : (batch, head, kv_seq_length, head_features)
        if len_kv is not None:
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk =  _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10)

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            # (batch, head, head_features, seq_length)
            return x.permute(0, 2, 3, 1).contiguous()
        else:
            # (batch, head, seq_length, head_features)
            return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, x, history=None, layer_past=None, len_past=None):
        hidden_states = x

        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        #_input_msk = None

        len_kv = None

        if layer_past is not None:
            # key : (batch, head, head_features, seq_length)
            # value : (batch, head, seq_length, head_features)
            # layer_past, key : (batch, head, seq_length, head_features)
            if len_past is None:
                # transpose back cf below
                past_key, past_value = \
                    layer_past[0].transpose(-2, -1), layer_past[1]
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = torch.arange(
                    0, key.shape[0], dtype=torch.long, device=key.device
                )

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch,:,len_past,:] = key.squeeze(-1)
                past_value[_batch,:,len_past,:] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        # transpose to have same shapes for stacking
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value, len_kv = len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

#  -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

#  -----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, len_past=None):
        a, present = self.attn(
            self.ln_1(x), layer_past=layer_past, len_past=len_past
        )
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

#  -----------------------------------------------------------------------------
class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits

#  -----------------------------------------------------------------------------
@dataclass
class GPT2Config(object):
    vocab_size: int=50257
    n_positions: int=1024
    n_ctx: int=1024
    n_embd: int=768
    #n_layer=12
    n_client_layer: int=3
    n_server_layer: int=9
    n_auxiliary_layer: int=2
    n_head: int=12
    layer_norm_epsilon: float=1e-5
    initializer_range: float=0.02
    lora_attn_dim: int=0
    lora_attn_alpha: float=128
    lora_dropout: float=0.0
    lora_r_dropout: float=0.0
    fix_dropout: float=0.0

#  -----------------------------------------------------------------------------
class GPT2ClientModel(nn.Module):
    def __init__(self, config):
        super(GPT2ClientModel, self).__init__()
        self.n_layer = config.n_client_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(self.n_layer)]
        )
        #self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.config = config

    def forward(
        self, 
        input_ids, 
        position_ids=None, 
        token_type_ids=None, 
        past=None, 
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length, 
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1) #.long()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)     
        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds

        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(
                hidden_states, layer_past = layer_past, len_past=len_past
            )
            presents.append(present)

        return hidden_states, presents, input_shape, past, len_past

#  -----------------------------------------------------------------------------
class GPT2LMClientModel(nn.Module):
    def __init__(self, *args, **kwargs):
        config = GPT2Config(*args, **kwargs) 
        super(GPT2LMClientModel, self).__init__()
        self.transformer = GPT2ClientModel(config)
        self.apply(self._init_weights)

    def forward(
        self, 
        input_ids, 
        past=None, 
        len_past=None, 
    ):
        return self.transformer(input_ids, past=past, len_past=len_past)
           
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"
            
            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        
        for n, p in self.transformer.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        self.transformer.load_state_dict(state_dict, strict=False)

#  -----------------------------------------------------------------------------
class GPT2ServerModel(nn.Module):
    def __init__(self, config):
        super(GPT2ServerModel, self).__init__()
        self.n_layer = config.n_server_layer
        #self.n_embd = config.n_embd
        #self.n_vocab = config.vocab_size

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

#  -----------------------------------------------------------------------------
class GPT2LMServerModel(nn.Module):
    def __init__(self, decoder_weights, *args, **kwargs):
        config = GPT2Config(*args, **kwargs)
        super(GPT2LMServerModel, self).__init__()
        self.transformer = GPT2ServerModel(config)
        self.lm_head = GPT2LMHead(decoder_weights, config)
        self.apply(self._init_weights)

    def set_tied(self, decoder_weights):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(decoder_weights)

    def forward(self, *args, **kwargs):
        # args : hidden_states, presents, input_shape, past, past_len
        #_batch, _len = args[2]
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

    def load_weight(self, state_dict, decoder_weights):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"
            
            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        
        for n, p in self.transformer.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        self.transformer.load_state_dict(state_dict, strict=False)
        self.set_tied(decoder_weights)

#  -----------------------------------------------------------------------------
def gpt2_client_to_server_params(client_model_obj):
    return {'decoder_weights' : client_model_obj.transformer.wte.weight}

register_client_server_pair(
    'gpt2', GPT2LMClientModel,
    server=GPT2LMServerModel,
    client_to_server_params_fn=gpt2_client_to_server_params
)
#  -----------------------------------------------------------------------------