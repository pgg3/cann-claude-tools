import torch
import torch.nn as nn


def module_fn(x, dim):
    cumsum = torch.cumsum(x.narrow(dim=dim, start=0, length=x.size(dim)-1), dim=dim)
    return torch.cat((torch.zeros_like(x.select(dim, 0).unsqueeze(dim)), cumsum), dim=dim)


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, fn=module_fn):
        return fn(x, self.dim)


batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
