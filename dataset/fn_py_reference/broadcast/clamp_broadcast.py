import torch
import torch.nn as nn


def module_fn(x, min_val, max_val):
    return torch.clamp(x, min=min_val, max=max_val)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, min_val, max_val, fn=module_fn):
        return fn(x, min_val, max_val)


def get_inputs():
    x = torch.rand(256, 2048) * 5
    min_val = torch.full((2048,), -2.0)
    max_val = torch.full((2048,), 2.0)
    return [x, min_val, max_val]

def get_init_inputs():
    return []
