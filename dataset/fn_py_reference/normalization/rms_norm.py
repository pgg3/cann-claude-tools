import torch
import torch.nn as nn


def module_fn(x: torch.Tensor, num_features: int, eps: float = 1e-5) -> torch.Tensor:
    # Calculate the RMS along the feature dimension
    rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
    
    # Normalize the input by dividing by the RMS
    return x / rms


class Model(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x, fn=module_fn) -> torch.Tensor:
        return fn(x, self.num_features, self.eps)


batch_size = 128
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
