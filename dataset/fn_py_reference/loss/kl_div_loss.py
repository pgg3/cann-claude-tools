import torch
import torch.nn as nn


def module_fn(predictions, targets):
    return torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=module_fn):
        return fn(predictions, targets)


batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape)*scale).softmax(dim=-1), torch.rand(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []
