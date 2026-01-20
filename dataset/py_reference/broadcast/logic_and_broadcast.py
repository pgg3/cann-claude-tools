import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a & b

def get_inputs():
    a = torch.randint(0, 2, (128, 512, 1024), dtype=torch.bool)
    b = torch.randint(0, 2, (1, 512, 1), dtype=torch.bool)
    return [a, b]

def get_init_inputs():
    return []
