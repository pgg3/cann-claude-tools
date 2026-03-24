import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv2d_weight: torch.Tensor,
    conv2d_bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs the 2D convolution using functional calls.
    """
    return F.conv2d(x, conv2d_weight, conv2d_bias, stride, padding, dilation, groups)


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.conv2d_weight = conv2d.weight
        self.conv2d_bias = conv2d.bias if bias else None
        self.stride = conv2d.stride
        self.padding = conv2d.padding
        self.dilation = conv2d.dilation
        self.groups = conv2d.groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return fn(
            x,
            self.conv2d_weight,
            self.conv2d_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# Test code


batch_size = 8
# smaller spatial dims
height = 512
width = 1024
in_channels = 64  # increased channels
out_channels = 128
kernel_size = 3
# asymmetric input: make width considerably larger than height

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
