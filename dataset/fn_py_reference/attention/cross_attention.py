import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    q,
    kv,
    in_proj_weight,
    in_proj_bias,
    out_proj_weight,
    out_proj_bias,
    num_heads,
):
    """
    Cross-attention between encoder and decoder sequences.

    Args:
        q: (batch, len_q, d_model) - query sequence
        kv: (batch, len_kv, d_model) - key/value sequence
        in_proj_weight: (3*d_model, d_model)
        in_proj_bias: (3*d_model,)
        out_proj_weight: (d_model, d_model)
        out_proj_bias: (d_model,)
        num_heads: number of attention heads

    Returns:
        output: (batch, len_q, d_model)
    """
    embed_dim = q.size(2)
    # batch_first input: transpose to (seq_len, batch, d_model)
    q_t = q.transpose(0, 1)
    kv_t = kv.transpose(0, 1)
    out, _ = F.multi_head_attention_forward(
        q_t,
        kv_t,
        kv_t,
        embed_dim,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        None,  # bias_k
        None,  # bias_v
        False,  # add_zero_attn
        0.0,  # dropout_p
        out_proj_weight,
        out_proj_bias,
        training=True,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
    )
    # transpose back to (batch, len_q, d_model)
    return out.transpose(0, 1)


class Model(nn.Module):
    """
    Cross-attention between encoder and decoder sequences.
    """

    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.in_proj_weight = nn.Parameter(mha.in_proj_weight.data.clone())
        self.in_proj_bias = nn.Parameter(mha.in_proj_bias.data.clone())
        self.out_proj_weight = nn.Parameter(mha.out_proj.weight.data.clone())
        self.out_proj_bias = nn.Parameter(mha.out_proj.bias.data.clone())
        self.num_heads = num_heads

    def forward(self, q, kv, fn=module_fn):
        return fn(
            q,
            kv,
            self.in_proj_weight,
            self.in_proj_bias,
            self.out_proj_weight,
            self.out_proj_bias,
            self.num_heads,
        )


batch_size = 16
len_q, len_kv = 64, 128
d_model = 512

def get_inputs():
    q = torch.rand(batch_size, len_q, d_model)
    kv = torch.rand(batch_size, len_kv, d_model)
    return [q, kv]

def get_init_inputs():
    return [d_model, 8]
