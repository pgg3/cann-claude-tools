import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x,
    in_proj_weight,
    in_proj_bias,
    out_proj_weight,
    out_proj_bias,
    causal_mask,
    num_heads,
):
    """
    Causal (decoder) self-attention with an upper-triangular mask.

    Args:
        x: (batch, seq_len, d_model)
        in_proj_weight: (3*d_model, d_model)
        in_proj_bias: (3*d_model,)
        out_proj_weight: (d_model, d_model)
        out_proj_bias: (d_model,)
        causal_mask: (max_seq_len, max_seq_len) boolean mask
        num_heads: number of attention heads

    Returns:
        output: (batch, seq_len, d_model)
    """
    seq_len = x.size(1)
    embed_dim = x.size(2)
    attn_mask = causal_mask[:seq_len, :seq_len]
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
        attn_mask=attn_mask,
    )
    # transpose back to (batch, seq_len, d_model)
    return out.transpose(0, 1)


class Model(nn.Module):
    """
    Causal (decoder) self-attention with an upper-triangular mask.
    Mask is precomputed and registered as a non-trainable buffer.
    """

    def __init__(self, d_model=256, num_heads=4, max_seq_len=4096):
        super().__init__()
        mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.in_proj_weight = nn.Parameter(mha.in_proj_weight.data.clone())
        self.in_proj_bias = nn.Parameter(mha.in_proj_bias.data.clone())
        self.out_proj_weight = nn.Parameter(mha.out_proj.weight.data.clone())
        self.out_proj_bias = nn.Parameter(mha.out_proj.bias.data.clone())
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Precompute causal mask (1 for disallowed positions)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.in_proj_weight,
            self.in_proj_bias,
            self.out_proj_weight,
            self.out_proj_bias,
            self.causal_mask,
            self.num_heads,
        )


batch_size = 32
seq_len = 64
d_model = 512

def get_inputs():
    x = torch.rand(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, 4]
