import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    groups: int,
) -> torch.Tensor:
    """
    Performs the transposed 3D convolution.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth_in, height_in, width_in).
        weight (torch.Tensor): The transposed convolution weight parameter.
        bias (torch.Tensor or None): The transposed convolution bias parameter.
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding applied to the input.
        output_padding (tuple): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
    """
    return F.conv_transpose3d(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of 3 integers representing the kernel size in the form (depth, height, width).
        stride (tuple, optional): Tuple of 3 integers representing the stride in the form (depth, height, width). Defaults to (1, 1, 1).
        padding (tuple, optional): Tuple of 3 integers representing the padding in the form (depth, height, width). Defaults to (0, 0, 0).
        output_padding (tuple, optional): Tuple of 3 integers representing the output padding in the form (depth, height, width). Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.weight = nn.Parameter(conv.weight.data.clone())
        self.bias = nn.Parameter(conv.bias.data.clone()) if bias else None
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth_in, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )


# Test code
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth_in = 16
height_in = 32
width_in = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth_in, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
