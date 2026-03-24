import torch
import torch.nn as nn


def module_fn(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)


class Model(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(Model, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x, fn=module_fn) -> torch.Tensor:
        return fn(x, self.negative_slope)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
