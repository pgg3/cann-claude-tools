import torch
import torch.nn as nn


def module_fn(a, b):
    return a & b


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b, fn=module_fn):
        return fn(a, b)


def get_inputs():
    a = torch.randint(0, 2, (128, 512, 1024), dtype=torch.bool)
    b = torch.randint(0, 2, (1, 512, 1), dtype=torch.bool)
    return [a, b]

def get_init_inputs():
    return []
