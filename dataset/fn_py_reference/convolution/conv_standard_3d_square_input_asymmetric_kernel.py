import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride,
    padding,
    dilation,
    groups,
):
    """
    Performs the 3D convolution using functional calls.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).
        weight (torch.Tensor): Convolution weight parameters.
        bias (torch.Tensor or None): Convolution bias parameters (None if bias=False).
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Padding applied to the input.
        dilation (int or tuple): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
    """
    return F.conv3d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.weight = nn.Parameter(conv3d.weight.data.clone())
        if conv3d.bias is not None:
            self.bias = nn.Parameter(conv3d.bias.data.clone())
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# Test code


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel
width = 64
height = 64
depth = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, width, height, depth)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
