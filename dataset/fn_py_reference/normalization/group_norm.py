import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """
    Applies Group Normalization in a functional manner.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
        weight (nn.Parameter): Weight parameter (gamma).
        bias (nn.Parameter): Bias parameter (beta).
        num_groups (int): Number of groups to divide the channels into.
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
    """
    return F.group_norm(x, num_groups, weight, bias, eps)


class Model(nn.Module):
    """
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(Model, self).__init__()
        gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        self.gn_weight = nn.Parameter(gn.weight.data.clone())
        self.gn_bias = nn.Parameter(gn.bias.data.clone())
        self.num_groups = gn.num_groups
        self.gn_eps = gn.eps

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return fn(x, self.gn_weight, self.gn_bias, self.num_groups, self.gn_eps)

batch_size = 128
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, num_groups] # num_features
