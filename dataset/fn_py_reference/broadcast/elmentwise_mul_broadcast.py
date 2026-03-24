import torch
import torch.nn as nn


def module_fn(a, b):
    return a * b


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b, fn=module_fn):
        return fn(a, b)


def get_inputs():
    a = torch.rand(4, 1, 2048)
    b = torch.rand(1, 4, 2048)
    return [a, b]

def get_init_inputs():
    return []
