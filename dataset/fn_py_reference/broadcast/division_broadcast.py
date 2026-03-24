import torch
import torch.nn as nn


def module_fn(x, scale):
    return x / scale


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, scale, fn=module_fn):
        return fn(x, scale)


def get_inputs():
    x = torch.rand(8, 3, 32, 128)
    scale = torch.rand(3, 1, 1)  # broadcast over H and W
    return [x, scale]

def get_init_inputs():
    return []
