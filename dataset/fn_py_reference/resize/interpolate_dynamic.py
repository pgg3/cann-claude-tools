import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, target_size):
    return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, target_size, fn=module_fn):
        return fn(x, target_size)


def get_inputs():
    x = torch.rand(4, 128, 100, 150)
    size = (200, 300)
    return [x, size]

def get_init_inputs():
    return []
