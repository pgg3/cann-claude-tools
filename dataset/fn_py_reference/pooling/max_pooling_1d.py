import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    return_indices: bool = False,
) -> torch.Tensor:
    """
    Applies Max Pooling 1D to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).
        kernel_size (int): Size of the window to take a max over.
        stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
        padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.

    Returns:
        torch.Tensor or tuple: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        If return_indices is True, returns a tuple of (output, indices).
    """
    return F.max_pool1d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
    )

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor or tuple: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
            If return_indices is True, returns a tuple of (output, indices).
        """
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
        )


batch_size = 64
features = 192
sequence_length = 2048

kernel_size = 8
stride      = 1
padding     = 4
return_indices = False

def get_inputs():
    x = torch.rand(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, return_indices]
