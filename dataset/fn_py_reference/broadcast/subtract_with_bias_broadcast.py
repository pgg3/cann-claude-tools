import torch
import torch.nn as nn


def module_fn(x, bias):
    return x - bias


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, bias, fn=module_fn):
        return fn(x, bias)


def get_inputs():
    x = torch.rand(8, 4096)
    bias = torch.rand(4096)
    return [x, bias]

def get_init_inputs():
    return []
