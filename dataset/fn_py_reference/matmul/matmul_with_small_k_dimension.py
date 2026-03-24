import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs a matrix multiplication on the inputs.

    Args:
        A (torch.Tensor): Input tensor of shape (M, K).
        B (torch.Tensor): Input tensor of shape (K, N).

    Returns:
        torch.Tensor: Output tensor of shape (M, N).
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(A, B)


M = 16384 * 2
N = 16384 * 2
K = 32 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
