import torch
import torch.nn as nn


def module_fn(x, indices, src):
    return x.index_copy(0, indices, src)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, indices, src, fn=module_fn):
        return fn(x, indices, src)


def get_inputs():
    x = torch.zeros(8192, 1024)
    indices = torch.randint(0, 8192, (2048,))
    src = torch.rand(2048, 1024)
    return [x, indices, src]

def get_init_inputs():
    return []
