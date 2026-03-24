import torch
import torch.nn as nn


def module_fn(param, grad, velocity):
    velocity = self.momentum * velocity + grad
    param = param - self.lr * velocity
    return param


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, param, grad, velocity, fn=module_fn):
        return fn(param, grad, velocity)


def get_inputs():
    param = torch.rand(1024, 4096)
    grad = torch.rand(1024, 4096)
    velocity = torch.zeros_like(param)
    return [param, grad, velocity]

def get_init_inputs():
    return []
