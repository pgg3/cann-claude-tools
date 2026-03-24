import torch
import torch.nn as nn


def module_fn(Q, K, V):
    Q_ = torch.relu(Q) + 1e-6
    K_ = torch.relu(K) + 1e-6
    KV = torch.einsum('bkd,bkv->bdv', K_, V)
    out = torch.einsum('bqd,bdv->bqv', Q_, KV)
    return out


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q, K, V, fn=module_fn):
        return fn(Q, K, V)


batch_size = 8
seq_len = 512
d_model = 1024

def get_inputs():
    Q = torch.rand(batch_size, seq_len, d_model)
    K = torch.rand(batch_size, seq_len, d_model)
    V = torch.rand(batch_size, seq_len, d_model)
    return [Q, K, V]

def get_init_inputs():
    return []
