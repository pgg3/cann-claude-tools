import torch
import torch.nn as nn


def module_fn(x, bias):
    return x + bias.view(1, -1, 1, 1)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, bias, fn=module_fn):
        return fn(x, bias)


def get_inputs():
    x = torch.rand(8, 512, 32, 32)
    bias = torch.rand(512)
    return [x, bias]

def get_init_inputs():
    return []
