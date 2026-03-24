import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    dilation: tuple,
    groups: int,
):
    """
    Performs a transposed 2D convolution operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
        conv_weight (torch.Tensor): The weights of the transposed convolution.
        conv_bias (torch.Tensor or None): The bias of the transposed convolution, or None if no bias.
        stride (tuple): Stride of the transposed convolution.
        padding (tuple): Padding for the input.
        output_padding (tuple): Additional size added to one side of the output shape.
        dilation (tuple): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv_transpose2d(
        x,
        conv_weight,
        conv_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


class Model(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
        output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv_t = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.conv_weight = nn.Parameter(conv_t.weight.data.clone())
        if conv_t.bias is not None:
            self.conv_bias = nn.Parameter(conv_t.bias.data.clone())
        else:
            self.conv_bias = None

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )


# Constants for default arguments


batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = (3, 5)
height_in = 128
width_in = 256

def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
