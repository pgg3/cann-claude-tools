import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100000, 768)

    def forward(self, indices):
        return self.embedding(indices)

def get_inputs():
    indices = torch.randint(0, 100000, (512, 128))
    return [indices]

def get_init_inputs():
    return []
