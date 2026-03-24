import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def module_fn(
    x,
    q_proj_weight,
    q_proj_bias,
    k_proj_weight,
    k_proj_bias,
    v_proj_weight,
    v_proj_bias,
    out_proj_weight,
    out_proj_bias,
    num_heads,
):
    """
    Multi-Query Attention (MQA).
    Multiple query heads, single shared key/value head.

    Args:
        x: (B, N, D) input tensor
        q_proj_weight: (D, D)
        q_proj_bias: (D,)
        k_proj_weight: (D // num_heads, D) - shared single head
        k_proj_bias: (D // num_heads,)
        v_proj_weight: (D // num_heads, D) - shared single head
        v_proj_bias: (D // num_heads,)
        out_proj_weight: (D, D)
        out_proj_bias: (D,)
        num_heads: number of query heads

    Returns:
        output: (B, N, D)
    """
    B, N, D = x.shape
    H = num_heads
    d_h = D // H

    Q = F.linear(x, q_proj_weight, q_proj_bias).view(B, N, H, d_h)
    K = F.linear(x, k_proj_weight, k_proj_bias).unsqueeze(2)  # shared K/V
    V = F.linear(x, v_proj_weight, v_proj_bias).unsqueeze(2)

    attn = torch.einsum("bnhd,bkhd->bh nk", Q, K) / math.sqrt(d_h)
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum("bhnk,bkhd->bnhd", attn, V).reshape(B, N, D)
    return F.linear(out, out_proj_weight, out_proj_bias)


class Model(nn.Module):
    """
    Multi-Query Attention (MQA)
    - Multiple query heads
    - Single shared key/value head
    """

    def __init__(self, d_model=4096, num_heads=32):
        super().__init__()
        q_proj = nn.Linear(d_model, d_model)
        k_proj = nn.Linear(d_model, d_model // num_heads)  # shared across heads
        v_proj = nn.Linear(d_model, d_model // num_heads)
        out_proj = nn.Linear(d_model, d_model)

        self.q_proj_weight = nn.Parameter(q_proj.weight.data.clone())
        self.q_proj_bias = nn.Parameter(q_proj.bias.data.clone())
        self.k_proj_weight = nn.Parameter(k_proj.weight.data.clone())
        self.k_proj_bias = nn.Parameter(k_proj.bias.data.clone())
        self.v_proj_weight = nn.Parameter(v_proj.weight.data.clone())
        self.v_proj_bias = nn.Parameter(v_proj.bias.data.clone())
        self.out_proj_weight = nn.Parameter(out_proj.weight.data.clone())
        self.out_proj_bias = nn.Parameter(out_proj.bias.data.clone())
        self.num_heads = num_heads

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.q_proj_weight,
            self.q_proj_bias,
            self.k_proj_weight,
            self.k_proj_bias,
            self.v_proj_weight,
            self.v_proj_bias,
            self.out_proj_weight,
            self.out_proj_bias,
            self.num_heads,
        )


batch_size, seq_len, d_model, num_heads = 2, 4096, 4096, 32

def get_inputs():
    x = torch.rand(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, num_heads]
