import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    use_input_stats: bool,
    momentum: float,
    eps: float,
) -> torch.Tensor:
    """
    Applies Instance Normalization in a functional manner.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).
        weight (nn.Parameter): Weight parameter (gamma), or None.
        bias (nn.Parameter): Bias parameter (beta), or None.
        running_mean (torch.Tensor): Running mean, or None.
        running_var (torch.Tensor): Running variance, or None.
        use_input_stats (bool): Whether to use input statistics.
        momentum (float): Momentum for running stats update.
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
    """
    return F.instance_norm(x, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)


class Model(nn.Module):
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        inorm = nn.InstanceNorm2d(num_features=num_features)
        self.inorm_weight = inorm.weight  # None (affine=False by default)
        self.inorm_bias = inorm.bias  # None (affine=False by default)
        self.register_buffer("inorm_running_mean", inorm.running_mean)  # None
        self.register_buffer("inorm_running_var", inorm.running_var)  # None
        self.inorm_momentum = inorm.momentum
        self.inorm_eps = inorm.eps

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return fn(
            x,
            self.inorm_weight,
            self.inorm_bias,
            self.inorm_running_mean,
            self.inorm_running_var,
            self.training,
            self.inorm_momentum,
            self.inorm_eps,
        )

batch_size = 128
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
