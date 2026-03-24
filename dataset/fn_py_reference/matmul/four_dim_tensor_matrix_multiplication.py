import torch
import torch.nn as nn


def module_fn(A, B):
    """
    Performs the 4D tensor-matrix multiplication.

Args:
    A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
    B (torch.Tensor): Input matrix of shape (l, k)

Returns:
    torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    return torch.einsum("bijl,lk->bijk", A, B)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)


b = 8
i = 256
j = 512
l = 256
k = 768

def get_inputs():
    A = torch.rand(b, i, j, l)
    B = torch.rand(l, k)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
