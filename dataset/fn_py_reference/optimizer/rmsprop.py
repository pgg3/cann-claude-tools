import torch
import torch.nn as nn


def module_fn(param, grad, v):
    v = self.alpha * v + (1 - self.alpha) * grad.pow(2)
    param = param - self.lr * grad / (v.sqrt() + self.eps)
    return param


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, param, grad, v, fn=module_fn):
        return fn(param, grad, v)


def get_inputs():
    param = torch.rand(2048, 2048)
    grad = torch.rand_like(param)
    v = torch.zeros_like(param)
    return [param, grad, v]

def get_init_inputs():
    return []
