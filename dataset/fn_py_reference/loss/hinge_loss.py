import torch
import torch.nn as nn


def module_fn(predictions, targets):
    return torch.mean(torch.clamp(1 - predictions * targets, min=0))


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=module_fn):
        return fn(predictions, targets)


batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.randint(0, 2, (batch_size,)).float() * 2 - 1]

def get_init_inputs():
    return []
