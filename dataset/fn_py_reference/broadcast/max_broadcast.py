import torch
import torch.nn as nn


def module_fn(a, b):
    return torch.maximum(a, b)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b, fn=module_fn):
        return fn(a, b)


def get_inputs():
    a = torch.rand(16, 1, 1024)
    b = torch.rand(1, 256, 1)
    return [a, b]

def get_init_inputs():
    return []
