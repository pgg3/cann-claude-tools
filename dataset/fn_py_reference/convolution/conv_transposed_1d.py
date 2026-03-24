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
    dilation: int,
) -> torch.Tensor:
    """
    Performs the transposed 1D convolution via functional API.

    Args:
        x (torch.Tensor): Input of shape (batch_size, in_channels, length).
        weight (torch.Tensor): Transposed convolution weight tensor.
        bias (torch.Tensor): Transposed convolution bias tensor or None.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        output_padding (int): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.
        dilation (int): Spacing between kernel elements.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
    """
    return F.conv_transpose1d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution.
        padding (int, optional): Padding applied to the input.
        output_padding (int, optional): Additional size added to one side of the output shape.
        groups (int, optional): Number of blocked connections from input channels to output channels.
        bias (bool, optional): If `True`, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv1d_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.conv_weight = conv1d_transpose.weight
        self.conv_bias = conv1d_transpose.bias
        self.stride = conv1d_transpose.stride
        self.padding = conv1d_transpose.padding
        self.output_padding = conv1d_transpose.output_padding
        self.groups = conv1d_transpose.groups
        self.dilation = conv1d_transpose.dilation

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.stride[0],
            self.padding[0],
            self.output_padding[0],
            self.groups,
            self.dilation[0],
        )


# Test code


batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
# much larger signal length for heavier workload
length = 65536

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
