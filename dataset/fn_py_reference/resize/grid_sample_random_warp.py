import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, grid):
    return F.grid_sample(x, grid, mode='bilinear', align_corners=False)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, grid, fn=module_fn):
        return fn(x, grid)


def get_inputs():
    N, C, H, W = 4, 3, 256, 256
    x = torch.rand(N, C, H, W)
    # random warped grid
    base = F.affine_grid(torch.eye(2, 3).unsqueeze(0).repeat(N,1,1).to(x), size=x.size(), align_corners=False)
    noise = torch.rand_like(base) * 0.05
    grid = base + noise
    return [x, grid]

def get_init_inputs():
    return []
