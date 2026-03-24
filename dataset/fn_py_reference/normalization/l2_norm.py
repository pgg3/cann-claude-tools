import torch
import torch.nn as nn


def module_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L2 normalization to the input tensor.

Args:
    x (torch.Tensor): Input tensor of shape (*, dim, *).

Returns:
    torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
    """
    return x / torch.norm(x, p=2, dim=1, keepdim=True)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, fn=module_fn) -> torch.Tensor:
        return fn(x)


batch_size = 32768
# choose dim so total <2^31
dim = 65535

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
