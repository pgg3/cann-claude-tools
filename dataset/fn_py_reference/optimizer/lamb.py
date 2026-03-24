import torch
import torch.nn as nn


def module_fn(param, m, v):
    r = m / (v.sqrt() + self.eps)
    trust_ratio = param.norm(p=2) / (r.norm(p=2) + self.eps)
    param = param - self.lr * trust_ratio * r
    return param


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, param, m, v, fn=module_fn):
        return fn(param, m, v)


def get_inputs():
    param = torch.rand(512, 4096)
    m = torch.rand_like(param)
    v = torch.abs(torch.rand_like(param)) + 1e-3
    return [param, m, v]

def get_init_inputs():
    return []
