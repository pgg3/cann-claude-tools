import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, theta):
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', align_corners=False)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, theta, fn=module_fn):
        return fn(x, theta)


def get_inputs():
    x = torch.rand(8, 64, 128, 128)
    theta = torch.tensor([[[0.866, -0.5, 0.0], [0.5, 0.866, 0.0]]] * 8)  # rotate 30 deg
    return [x, theta]

def get_init_inputs():
    return []
