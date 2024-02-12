import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from gla import GatedLinearAttention

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class GLAConfig(PretrainedConfig):
    model_type = "gla"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        d_model=2048,
        n_head=8,
        n_layer=24,
        use_gk=True,
        use_gv=False,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        context_length=2048,
        vocab_size=50432,
        tie_word_embeddings=False,
        load_from_llama=False,
        **kwargs,
    ):
    
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.context_length = context_length
        self.use_cache = use_cache

        self.use_gk = use_gk
        self.use_gv = use_gv
        self.load_from_llama = load_from_llama

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class GLABlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model, eps=1e-5)
        
        self.attn = GatedLinearAttention(config)
        self.ln_2 = RMSNorm(config.d_model, eps=1e-5)

        mlp_ratio = 4
        multiple_of = 256
        mlp_hidden = int(config.d_model * mlp_ratio * 2 / 3)
        mlp_hidden = multiple_of * ((mlp_hidden + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(config.d_model, mlp_hidden, bias=False)
        self.w2 = nn.Linear(mlp_hidden, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, mlp_hidden, bias=False)
        self.mlp = lambda x: self.w2(F.silu(self.w1(x)) * self.w3(x))


    def forward(self, x, hidden_states=None):
        # attention/rnn
        x_att, new_hidden_states = self.attn(self.ln_1(x), hidden_states)
        x = x + x_att
        # ffn
        x_mlp = self.mlp(self.ln_2(x))
        x = x + x_mlp
        return x, new_hidden_states
    
class GLAPreTrainedModel(PreTrainedModel):
    config_class = GLAConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["GLABlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class GLAModel(GLAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.h = nn.ModuleList([GLABlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model, eps=1e-5)

    def forward( self,
        input_ids: torch.LongTensor = None,
        hidden_states: torch.Tensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if hidden_states is None:
            hidden_states = [None] * self.config.n_layer

        new_hidden_states = []
        x = self.wte(input_ids)
        for block, hidden_state in zip(self.h, hidden_states):
            x, last_context_state = block(x, hidden_state)
            new_hidden_states.append(last_context_state)

        x = self.ln_f(x)            
        
        # the hidden states now means the recurrent hidden states
        return BaseModelOutputWithPast(
            last_hidden_state=x,
            hidden_states=new_hidden_states,
        )

class GLAForCausalLM(GLAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.transformer = GLAModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.n_layer = config.n_layer

        self.apply(self._init_weights)
        self._post_init()

    def _init_weights(self, module):
        """general init strategy"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
            if hasattr(module, "bias") and  module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)
    
    def _post_init(self):
        """custom init strategy"""
        for name, module in self.named_modules():
            if hasattr(module, "post_init"):
                # print(name)
                module.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        assert return_dict is True
        
        xmr_output = self.transformer(input_ids, hidden_states)
        logits = self.lm_head(xmr_output.last_hidden_state)
        new_hidden_states = xmr_output.hidden_states
        
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-1)
        else:
            loss = None

        # output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=new_hidden_states,
        )
