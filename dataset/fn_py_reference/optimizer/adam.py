import torch
import torch.nn as nn


def module_fn(param, grad, m, v):
    m = self.beta1 * m + (1 - self.beta1) * grad
    v = self.beta2 * v + (1 - self.beta2) * grad.pow(2)
    m_hat = m / (1 - self.beta1 ** self.step)
    v_hat = v / (1 - self.beta2 ** self.step)
    param = param - self.lr * m_hat / (v_hat.sqrt() + self.eps)
    return param


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, param, grad, m, v, fn=module_fn):
        return fn(param, grad, m, v)


def get_inputs():
    param = torch.rand(512, 2048)
    grad = torch.rand_like(param)
    m = torch.zeros_like(param)
    v = torch.zeros_like(param)
    return [param, grad, m, v]

def get_init_inputs():
    return []
