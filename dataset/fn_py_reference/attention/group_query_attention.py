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
    num_kv_heads,
):
    """
    Grouped-Query Attention (GQA).

    Args:
        x: (B, L, D) input tensor
        q_proj_weight: (D, D)
        q_proj_bias: (D,)
        k_proj_weight: (D * num_kv_heads // num_heads, D)
        k_proj_bias: (D * num_kv_heads // num_heads,)
        v_proj_weight: (D * num_kv_heads // num_heads, D)
        v_proj_bias: (D * num_kv_heads // num_heads,)
        out_proj_weight: (D, D)
        out_proj_bias: (D,)
        num_heads: total number of query heads
        num_kv_heads: number of key/value heads

    Returns:
        output: (B, L, D)
    """
    B, L, D = x.shape
    H = num_heads
    H_kv = num_kv_heads
    head_dim = D // H
    group_size = H // H_kv

    # projections
    Q = F.linear(x, q_proj_weight, q_proj_bias).view(B, L, H, head_dim)
    K = F.linear(x, k_proj_weight, k_proj_bias).view(B, L, H_kv, head_dim)
    V = F.linear(x, v_proj_weight, v_proj_bias).view(B, L, H_kv, head_dim)

    # Expand K/V to match query groups
    if group_size > 1:
        K = K.repeat_interleave(group_size, dim=2)
        V = V.repeat_interleave(group_size, dim=2)

    # attention
    attn = torch.einsum("blhd,bmhd->bh lm", Q, K) / math.sqrt(head_dim)
    attn = F.softmax(attn, dim=-1)
    out = torch.einsum("bhlm,bmhd->blhd", attn, V).reshape(B, L, D)
    return F.linear(out, out_proj_weight, out_proj_bias)


class Model(nn.Module):
    """
    Grouped-Query Attention (GQA)
    -----------------------------
    Like LLaMA-style attention: multiple query heads share a smaller set of key/value heads.
    """

    def __init__(self, d_model=1024, num_heads=16, num_kv_heads=4):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        q_proj = nn.Linear(d_model, d_model)
        k_proj = nn.Linear(d_model, d_model * num_kv_heads // num_heads)
        v_proj = nn.Linear(d_model, d_model * num_kv_heads // num_heads)
        out_proj = nn.Linear(d_model, d_model)

        self.q_proj_weight = nn.Parameter(q_proj.weight.data.clone())
        self.q_proj_bias = nn.Parameter(q_proj.bias.data.clone())
        self.k_proj_weight = nn.Parameter(k_proj.weight.data.clone())
        self.k_proj_bias = nn.Parameter(k_proj.bias.data.clone())
        self.v_proj_weight = nn.Parameter(v_proj.weight.data.clone())
        self.v_proj_bias = nn.Parameter(v_proj.bias.data.clone())
        self.out_proj_weight = nn.Parameter(out_proj.weight.data.clone())
        self.out_proj_bias = nn.Parameter(out_proj.bias.data.clone())

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
            self.num_kv_heads,
        )


# ---------------------------------------------------------------------
# Example kernelbench configuration
# ---------------------------------------------------------------------


batch_size = 8
seq_len = 128
d_model = 1024
num_heads = 16
num_kv_heads = 4

def get_inputs():
    x = torch.rand(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, num_heads, num_kv_heads]
