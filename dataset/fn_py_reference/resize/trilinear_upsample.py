import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x):
    return F.interpolate(x, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, fn=module_fn):
        return fn(x)


def get_inputs():
    x = torch.rand(2, 16, 32, 32, 32)
    return [x]

def get_init_inputs():
    return []
