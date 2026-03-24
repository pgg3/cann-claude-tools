import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x,
    in_proj_weight,
    in_proj_bias,
    out_proj_weight,
    out_proj_bias,
    num_heads,
):
    """
    Standard multi-head attention (MHA).

    Args:
        x: (batch, seq_len, d_model)
        in_proj_weight: (3*d_model, d_model)
        in_proj_bias: (3*d_model,)
        out_proj_weight: (d_model, d_model)
        out_proj_bias: (d_model,)
        num_heads: number of attention heads

    Returns:
        output: (batch, seq_len, d_model)
    """
    embed_dim = x.size(2)
    # batch_first input: transpose to (seq_len, batch, d_model)
    x_t = x.transpose(0, 1)
    out, _ = F.multi_head_attention_forward(
        x_t,
        x_t,
        x_t,
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
    # transpose back to (batch, seq_len, d_model)
    return out.transpose(0, 1)


class Model(nn.Module):
    """
    Implements standard multi-head attention (MHA).
    """

    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.in_proj_weight = nn.Parameter(mha.in_proj_weight.data.clone())
        self.in_proj_bias = nn.Parameter(mha.in_proj_bias.data.clone())
        self.out_proj_weight = nn.Parameter(mha.out_proj.weight.data.clone())
        self.out_proj_bias = nn.Parameter(mha.out_proj.bias.data.clone())
        self.num_heads = num_heads

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.in_proj_weight,
            self.in_proj_bias,
            self.out_proj_weight,
            self.out_proj_bias,
            self.num_heads,
        )


batch_size = 16
seq_len = 256
d_model = 512
num_heads = 8

def get_inputs():
    x = torch.rand(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, num_heads]
