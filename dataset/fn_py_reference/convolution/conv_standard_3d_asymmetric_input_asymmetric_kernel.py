import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    """
    Performs a 3D convolution using functional calls.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
        weight (nn.Parameter): 3D convolution weight tensor of shape (out_channels, in_channels/groups, kd, kh, kw).
        bias (nn.Parameter or None): 3D convolution bias tensor of shape (out_channels,). May be None if bias=False.
        stride (tuple): Stride of the convolution (stride_d, stride_h, stride_w).
        padding (tuple): Padding applied to the input (padding_d, padding_h, padding_w).
        dilation (tuple): Spacing between kernel elements (dilation_d, dilation_h, dilation_w).
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
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
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kd, kh, kw).
        stride (tuple, optional): Stride of the convolution (sd, sh, sw). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding (pd, ph, pw). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements (dd, dh, dw). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        dilation: tuple = (1, 1, 1),
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
        self.bias = (
            nn.Parameter(conv3d.bias.data.clone()) if conv3d.bias is not None else None
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return fn(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# Test code


batch_size = 8
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth = 16
height = 128
width = 128

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
