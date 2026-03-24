import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> torch.Tensor:
    """
    Applies Batch Normalization in a functional manner.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
        weight (nn.Parameter): Weight parameter (gamma).
        bias (nn.Parameter): Bias parameter (beta).
        running_mean (torch.Tensor): Running mean for inference.
        running_var (torch.Tensor): Running variance for inference.
        training (bool): Whether the model is in training mode.
        momentum (float): Momentum for running stats update.
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
    """
    return F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)


class Model(nn.Module):
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        bn = nn.BatchNorm2d(num_features=num_features)
        self.bn_weight = nn.Parameter(bn.weight.data.clone())
        self.bn_bias = nn.Parameter(bn.bias.data.clone())
        self.register_buffer("bn_running_mean", bn.running_mean.data.clone())
        self.register_buffer("bn_running_var", bn.running_var.data.clone())
        self.bn_momentum = bn.momentum
        self.bn_eps = bn.eps

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return fn(
            x,
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
            self.training,
            self.bn_momentum,
            self.bn_eps,
        )

batch_size = 64
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
