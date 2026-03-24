import torch
import torch.nn as nn


def module_fn(param, grad, accum):
    accum = accum + grad.pow(2)
    param = param - self.lr * grad / (accum.sqrt() + self.eps)
    return param


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, param, grad, accum, fn=module_fn):
        return fn(param, grad, accum)


def get_inputs():
    param = torch.rand(1024, 4096)
    grad = torch.rand_like(param)
    accum = torch.zeros_like(param)
    return [param, grad, accum]

def get_init_inputs():
    return []
