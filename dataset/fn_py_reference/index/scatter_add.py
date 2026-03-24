import torch
import torch.nn as nn


def module_fn(x, idx, updates):
    return x.scatter_add(dim=1, index=idx, src=updates)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, idx, updates, fn=module_fn):
        return fn(x, idx, updates)


def get_inputs():
    x = torch.zeros(32, 4096)
    idx = torch.randint(0, 4096, (32, 1024))
    updates = torch.rand(32, 1024)
    return [x, idx, updates]

def get_init_inputs():
    return []
