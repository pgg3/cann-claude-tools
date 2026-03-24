import torch
import torch.nn as nn


def module_fn(x, exponent):
    return x ** exponent


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, exponent, fn=module_fn):
        return fn(x, exponent)


def get_inputs():
    x = torch.abs(torch.rand(4, 2048)) + 1e-2
    exponent = torch.full((2048,), 0.5,)
    return [x, exponent]

def get_init_inputs():
    return []
