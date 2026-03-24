import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    text,
    img,
    q_proj_weight,
    q_proj_bias,
    kv_proj_weight,
    kv_proj_bias,
    mha_in_proj_weight,
    mha_in_proj_bias,
    mha_out_proj_weight,
    mha_out_proj_bias,
    num_heads,
):
    """
    Cross-modal attention: text queries, visual keys/values.

    Args:
        text: (batch, len_text, d_model_text) - text input
        img: (batch, len_img, d_model_img) - image input
        q_proj_weight: (d_model_text, d_model_text) - query projection weight
        q_proj_bias: (d_model_text,) - query projection bias
        kv_proj_weight: (d_model_text, d_model_img) - kv projection weight
        kv_proj_bias: (d_model_text,) - kv projection bias
        mha_in_proj_weight: (3*d_model_text, d_model_text)
        mha_in_proj_bias: (3*d_model_text,)
        mha_out_proj_weight: (d_model_text, d_model_text)
        mha_out_proj_bias: (d_model_text,)
        num_heads: number of attention heads

    Returns:
        output: (batch, len_text, d_model_text)
    """
    d_model_text = q_proj_weight.size(0)

    # Apply pre-projections
    q = F.linear(text, q_proj_weight, q_proj_bias)
    kv = F.linear(img, kv_proj_weight, kv_proj_bias)

    # batch_first input: transpose to (seq_len, batch, d_model)
    q_t = q.transpose(0, 1)
    kv_t = kv.transpose(0, 1)
    out, _ = F.multi_head_attention_forward(
        q_t,
        kv_t,
        kv_t,
        d_model_text,
        num_heads,
        mha_in_proj_weight,
        mha_in_proj_bias,
        None,  # bias_k
        None,  # bias_v
        False,  # add_zero_attn
        0.0,  # dropout_p
        mha_out_proj_weight,
        mha_out_proj_bias,
        training=True,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
    )
    # transpose back to (batch, len_text, d_model_text)
    return out.transpose(0, 1)


class Model(nn.Module):
    """
    Cross-modal attention: text queries, visual keys/values.
    """

    def __init__(self, d_model_text=256, d_model_img=512, num_heads=8):
        super().__init__()
        q_proj = nn.Linear(d_model_text, d_model_text)
        kv_proj = nn.Linear(d_model_img, d_model_text)
        mha = nn.MultiheadAttention(d_model_text, num_heads, batch_first=True)

        self.q_proj_weight = nn.Parameter(q_proj.weight.data.clone())
        self.q_proj_bias = nn.Parameter(q_proj.bias.data.clone())
        self.kv_proj_weight = nn.Parameter(kv_proj.weight.data.clone())
        self.kv_proj_bias = nn.Parameter(kv_proj.bias.data.clone())
        self.mha_in_proj_weight = nn.Parameter(mha.in_proj_weight.data.clone())
        self.mha_in_proj_bias = nn.Parameter(mha.in_proj_bias.data.clone())
        self.mha_out_proj_weight = nn.Parameter(mha.out_proj.weight.data.clone())
        self.mha_out_proj_bias = nn.Parameter(mha.out_proj.bias.data.clone())
        self.num_heads = num_heads

    def forward(self, text, img, fn=module_fn):
        return fn(
            text,
            img,
            self.q_proj_weight,
            self.q_proj_bias,
            self.kv_proj_weight,
            self.kv_proj_bias,
            self.mha_in_proj_weight,
            self.mha_in_proj_bias,
            self.mha_out_proj_weight,
            self.mha_out_proj_bias,
            self.num_heads,
        )


batch_size = 4
len_text = 32
len_img = 196

def get_inputs():
    text = torch.rand(batch_size, len_text, 256)
    img = torch.rand(batch_size, len_img, 512)
    return [text, img]

def get_init_inputs():
    return [256, 512, 8]
