import torch
import torch.nn as nn


def module_fn(x, idx):
    return torch.gather(x, dim=1, index=idx)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, idx, fn=module_fn):
        return fn(x, idx)


def get_inputs():
    x = torch.rand(128, 8192)
    idx = torch.randint(0, 8192, (128, 4096))
    return [x, idx]

def get_init_inputs():
    return []
