import torch
import torch.nn as nn


def module_fn(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.log_softmax(x, dim=dim)


class Model(nn.Module):
    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, fn=module_fn) -> torch.Tensor:
        return fn(x, self.dim)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
