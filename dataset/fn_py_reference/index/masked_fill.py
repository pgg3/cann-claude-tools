import torch
import torch.nn as nn


def module_fn(x, mask):
    return x.masked_fill(mask, float('-inf'))


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, mask, fn=module_fn):
        return fn(x, mask)


def get_inputs():
    x = torch.rand(64, 512, 512)
    mask = torch.randint(0, 2, (64, 512, 512), dtype=torch.bool)
    return [x, mask]

def get_init_inputs():
    return []
