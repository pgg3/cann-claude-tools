import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter,
    normalized_shape: tuple,
) -> torch.Tensor:
    """
    Applies Layer Normalization in a functional manner.

    Args:
        x (torch.Tensor): Input tensor of shape (*, normalized_shape).
        weight (nn.Parameter): Weight parameter (gamma).
        bias (nn.Parameter): Bias parameter (beta).
        normalized_shape (tuple): Shape of the normalized dimensions.

    Returns:
        torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
    """
    return F.layer_norm(x, normalized_shape, weight, bias)


class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.ln_weight = nn.Parameter(ln.weight.data.clone())
        self.ln_bias = nn.Parameter(ln.bias.data.clone())
        self.normalized_shape = ln.normalized_shape

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return fn(x, self.ln_weight, self.ln_bias, self.normalized_shape)

batch_size = 64
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]
