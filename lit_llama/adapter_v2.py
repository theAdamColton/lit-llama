"""Implementation of the paper:

LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model
https://arxiv.org/abs/2304.15010

Some small changes:
    Allows the option to finetune wte embeddings.

    Allows the option to finetune the lm_head. This should probably be used in
    conjunction with finetuning embeddings.
"""
# mypy: ignore-errors
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
import lit_llama.adapter as llama_adapter
from lit_llama.model import build_rope_cache, apply_rope, RMSNorm, MLP
from lit_llama.utils import find_multiple


@dataclass
class LLaMAConfig(llama_adapter.LLaMAConfig):
    add_bias_and_scale: bool = True
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 1
    train_wte: bool = True
    train_lm_head: bool = True


def with_s_b(x, module, scale, bias):
    """
    Forward with scale and bias

    In the LLaMA-Adapter 2 paper they write this equation this way, but it is
    different from this in their implementation.
    """
    if scale is not None and bias is not None:
        return scale(module(x + bias) )
    else:
        return module(x)


class CausalSelfAttention(nn.Module):
    """A modification of `lit_llama.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(self, config: LLaMAConfig, block_idx: int) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)


        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)


        if config.add_bias_and_scale:
            self.c_attn_bias = nn.Parameter(torch.zeros(config.n_embd))
            self.c_attn_scale = nn.Parameter(torch.ones(3 * config.n_embd))
            self.c_proj_bias = nn.Parameter(torch.zeros(config.n_embd))
            self.c_proj_scale = nn.Parameter(torch.ones(config.n_embd))
        else:
            self.c_attn_bias = None
            self.c_attn_scale = None
            self.c_proj_bias = None
            self.c_proj_scale = None

        
        if block_idx >= config.adapter_start_layer:
            # adapter higher level embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = nn.Parameter(torch.zeros(1))

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.block_idx = block_idx
        self.adapter_prompt_length = config.adapter_prompt_length
        self.adapter_start_layer = config.adapter_start_layer
        self.rope_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = with_s_b(x,
                           self.c_attn,
                           self.c_attn_scale,
                           self.c_attn_bias).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head, 
                dtype=x.dtype,
                device=x.device,
            )

        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        if self.block_idx >= self.adapter_start_layer:
            prefix = self.adapter_wte.weight.reshape(1, self.adapter_prompt_length, self.n_embd)

            aT = prefix.size(1)
            _, ak, av = self.c_attn(prefix).split(self.n_embd, dim=2)
            ak = ak.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2)
            av = av.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2)

            amask = torch.ones(q.shape[-2], ak.shape[-2], dtype=torch.bool, device=x.device)
            ay = F.scaled_dot_product_attention(q, ak, av, attn_mask=amask, dropout_p=0.0, is_causal=False)
            y = y + self.gating_factor * ay



        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = with_s_b(y, self.c_proj, self.c_proj_scale, self.c_proj_bias)

        return y


class MLP(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

        if config.add_bias_and_scale:
            self.c_fc1_bias = nn.Parameter(torch.zeros(config.n_embd))
            self.c_fc2_bias = nn.Parameter(torch.zeros(config.n_embd))
            self.c_proj_bias = nn.Parameter(torch.zeros(n_hidden))
            self.c_fc1_scale = nn.Parameter(torch.ones(n_hidden))
            self.c_fc2_scale = nn.Parameter(torch.ones(n_hidden))
            self.c_proj_scale = nn.Parameter(torch.ones(config.n_embd))
        else:
            self.c_fc1_bias = None
            self.c_fc2_bias = None
            self.c_proj_bias =None
            self.c_fc1_scale =None
            self.c_fc2_scale =None
            self.c_proj_scale =None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(with_s_b(x, self.c_fc1, self.c_fc1_scale, self.c_fc1_bias)) * with_s_b(x, self.c_fc2, self.c_fc2_scale, self.c_fc2_bias)
        x = with_s_b(x, self.c_proj, self.c_proj_scale, self.c_proj_bias)
        return x


class Block(nn.Module):
    """The implementation is identical to `lit_llama.model.Block` with the exception that
    we replace the attention layer where adaption is implemented,
    """

    def __init__(self, config: LLaMAConfig, block_idx: int) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, block_idx)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


class LLaMA(llama_adapter.LLaMA):
    """The implementation is identical to `lit_llama.model.LLaMA` with the exception that
    the `Block` saves the layer index and passes it down to the attention layer."""

    def __init__(self, config: LLaMAConfig) -> None:
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )

    @classmethod
    def from_name(cls, name: str):
        return cls(LLaMAConfig.from_name(name))


def is_param_trainable(name: str, config: LLaMAConfig) -> bool:
    """
    compares param_name with string snippets of all of the names of the trainable parameters of LLaMA V2
    """
    return "adapter_wte" in name \
                                or "gating_factor" in name \
                                or "_bias" in name \
                                or "_scale" in name \
                                or "lm_head" in name and config.train_lm_head


def mark_only_adapter_as_trainable(model: LLaMA) -> None:
    """Sets `requires_grad=False` for all non-adapter and all non weight/bias
    weights, and on lm_head if that setting has been set."""
    for name, param in model.named_parameters():
        param.requires_grad = is_param_trainable(name, model.config)

def adapter_state_from_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    return {name: param for name, param in state_dict.items() if is_param_trainable(name, state_dict["config"]) }
