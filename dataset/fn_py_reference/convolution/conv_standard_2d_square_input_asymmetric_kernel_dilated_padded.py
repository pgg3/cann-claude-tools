import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: tuple,
    dilation: tuple,
) -> torch.Tensor:
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        weight (torch.Tensor): Weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width).
        bias (torch.Tensor): Bias tensor of shape (out_channels).
        stride (int): Stride of the convolution.
        padding (tuple): Padding applied to the input (top/bottom, left/right).
        dilation (tuple): Spacing between kernel elements (height, width).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv2d(
        x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation
    )


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int,
        padding: tuple,
        dilation: tuple,
        bias: bool,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        # Copy the initialized parameters
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if bias else None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(
        self,
        x: torch.Tensor,
        fn=module_fn,
    ) -> torch.Tensor:
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )


# Constants


batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512
stride = 1
padding = (2, 4)
dilation = (2, 3)

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
