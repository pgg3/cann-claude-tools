import torch
import torch.nn as nn


def module_fn(anchor, positive, negative):
    return self.loss_fn(anchor, positive, negative)


class Model(nn.Module):
    """
    Simple model that wraps module_fn.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, anchor, positive, negative, fn=module_fn):
        return fn(anchor, positive, negative)


batch_size = 32768
input_shape = (8192,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape), torch.rand(batch_size, *input_shape)]
    
def get_init_inputs():
    return [1.0]  # Default margin
