import torch
import torch.nn as nn


def module_fn(predictions, targets):
    cosine_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
    return torch.mean(1 - cosine_sim)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=module_fn):
        return fn(predictions, targets)


batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []
