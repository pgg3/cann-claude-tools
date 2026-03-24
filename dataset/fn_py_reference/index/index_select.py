import torch
import torch.nn as nn


def module_fn(x, indices):
    return torch.index_select(x, dim=1, index=indices)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, indices, fn=module_fn):
        return fn(x, indices)


def get_inputs():
    x = torch.rand(256, 8192)
    indices = torch.randint(0, 8192, (2048,))
    return [x, indices]

def get_init_inputs():
    return []
