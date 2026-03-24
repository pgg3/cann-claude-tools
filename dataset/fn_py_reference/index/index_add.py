import torch
import torch.nn as nn


def module_fn(x, indices, values):
    return x.index_add(dim=0, index=indices, source=values)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, indices, values, fn=module_fn):
        return fn(x, indices, values)


def get_inputs():
    x = torch.zeros(1024, 4096)
    indices = torch.randint(0, 1024, (8192,))
    values = torch.rand(8192, 4096)
    return [x, indices, values]

def get_init_inputs():
    return []
