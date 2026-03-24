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
    window_size,
):
    """
    Sliding-window causal attention (local attention).
    Each token attends only to a fixed window of previous tokens.

    Args:
        x: (B, N, D) input tensor
        q_proj_weight: (D, D)
        q_proj_bias: (D,)
        k_proj_weight: (D, D)
        k_proj_bias: (D,)
        v_proj_weight: (D, D)
        v_proj_bias: (D,)
        out_proj_weight: (D, D)
        out_proj_bias: (D,)
        num_heads: number of attention heads
        window_size: size of the sliding window

    Returns:
        output: (B, N, D)
    """
    B, N, D = x.shape
    H = num_heads
    d_h = D // H

    Q = F.linear(x, q_proj_weight, q_proj_bias).view(B, N, H, d_h)
    K = F.linear(x, k_proj_weight, k_proj_bias).view(B, N, H, d_h)
    V = F.linear(x, v_proj_weight, v_proj_bias).view(B, N, H, d_h)

    attn_out = torch.zeros_like(Q)
    for i in range(0, N):
        start = max(0, i - window_size)
        Qi = Q[:, i : i + 1]
        Ki = K[:, start : i + 1]
        Vi = V[:, start : i + 1]
        scores = torch.einsum("bqhd,bkhd->bhqk", Qi, Ki) / math.sqrt(d_h)
        weights = F.softmax(scores, dim=-1)
        attn_out[:, i : i + 1] = torch.einsum("bhqk,bkhd->bqhd", weights, Vi)
    return F.linear(attn_out.reshape(B, N, D), out_proj_weight, out_proj_bias)


class Model(nn.Module):
    """
    Sliding-window causal attention (local attention).
    Each token attends only to a fixed window of previous tokens.
    Used in Longformer, MPT, and streaming LLMs.
    """

    def __init__(self, d_model=1024, num_heads=16, window_size=256):
        super().__init__()
        q_proj = nn.Linear(d_model, d_model)
        k_proj = nn.Linear(d_model, d_model)
        v_proj = nn.Linear(d_model, d_model)
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
        self.window_size = window_size

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
            self.window_size,
        )


batch_size, seq_len, d_model, num_heads, window_size = 4, 2048, 1024, 16, 256

def get_inputs():
    x = torch.rand(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, num_heads, window_size]
