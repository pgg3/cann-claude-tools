import torch
import torch.nn as nn


def module_fn(x, idx, updates):
    return x.scatter(dim=1, index=idx, src=updates)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, idx, updates, fn=module_fn):
        return fn(x, idx, updates)


def get_inputs():
    x = torch.zeros(64, 8192)
    idx = torch.randint(0, 8192, (64, 4096))
    updates = torch.rand_like(idx, dtype=torch.float)
    return [x, idx, updates]

def get_init_inputs():
    return []
