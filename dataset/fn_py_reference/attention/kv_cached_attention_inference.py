import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def module_fn(Q, K_cache, V_cache):
    output = F.scaled_dot_product_attention(Q, K_cache, V_cache)
    return output


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q, K_cache, V_cache, fn=module_fn):
        return fn(Q, K_cache, V_cache)


batch_size = 1
q_len = 1
kv_len = 2048
d_model = 4096

def get_inputs():
    Q = torch.rand(batch_size, q_len, d_model)
    K = torch.rand(batch_size, kv_len, d_model)
    V = torch.rand(batch_size, kv_len, d_model)
    return [Q, K, V]

def get_init_inputs():
    return []
