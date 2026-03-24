import torch
import torch.nn as nn


def module_fn(x, idx, value):
    x[idx] = value
    return x


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, idx, value, fn=module_fn):
        return fn(x, idx, value)


def get_inputs():
    x = torch.zeros(10000, 1024)
    idx = torch.randint(0, 10000, (2048,))
    value = torch.rand(2048, 1024)
    return [x, idx, value]

def get_init_inputs():
    return []
