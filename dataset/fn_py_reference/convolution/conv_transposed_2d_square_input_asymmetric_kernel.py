import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs the transposed 2D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        weight (torch.Tensor): The weight tensor of the transposed conv layer.
        bias (torch.Tensor or None): The bias tensor of the transposed conv layer.
        stride (int): Stride of the convolution.
        padding (int or tuple): Padding applied to the input.
        output_padding (int or tuple): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv_transpose2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.weight = nn.Parameter(conv_transpose.weight.data.clone())
        if conv_transpose.bias is not None:
            self.bias = nn.Parameter(conv_transpose.bias.data.clone())
        else:
            self.register_parameter("bias", None)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )


# Constants for default arguments


batch_size = 8
in_channels = 64
out_channels = 64
kernel_size = (3, 7)  # larger asymmetric kernel
width = 512
height = 512

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
