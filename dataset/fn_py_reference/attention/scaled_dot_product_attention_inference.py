import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(Q, K, V):
    output = F.scaled_dot_product_attention(Q, K, V)
    return output


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q, K, V, fn=module_fn):
        return fn(Q, K, V)


batch_size = 1
seq_len = 2048
d_model = 4096

def get_inputs():
    Q = torch.rand(batch_size, seq_len, d_model)
    K = torch.rand(batch_size, seq_len, d_model)
    V = torch.rand(batch_size, seq_len, d_model)
    return [Q, K, V]

def get_init_inputs():
    return []
