import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs a 1D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        conv_weight (torch.Tensor): The weight of the convolution filter.
        conv_bias (torch.Tensor or None): The bias of the convolution filter, or None if bias=False.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
    """
    return F.conv1d(
        x,
        conv_weight,
        conv_bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a standard 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
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
        conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.conv_weight = conv1d.weight
        self.conv_bias = conv1d.bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# Test code


batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
