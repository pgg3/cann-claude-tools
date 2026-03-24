import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, theta):
    x_up = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
    grid = F.affine_grid(theta, x_up.size(), align_corners=False)
    return F.grid_sample(x_up, grid, mode='bilinear', align_corners=False)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, theta, fn=module_fn):
        return fn(x, theta)


def get_inputs():
    x = torch.rand(2, 64, 64, 64)
    theta = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]] * 2)
    return [x, theta]

def get_init_inputs():
    return []
