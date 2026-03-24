import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(indices: torch.Tensor, weight: nn.Parameter) -> torch.Tensor:
    """
    Performs an embedding lookup in a functional manner.

    Args:
        indices (torch.Tensor): Index tensor for the embedding lookup.
        weight (nn.Parameter): Embedding weight matrix.

    Returns:
        torch.Tensor: Output tensor with embeddings looked up.
    """
    return F.embedding(indices, weight)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        embedding = nn.Embedding(100000, 768)
        self.embedding_weight = nn.Parameter(embedding.weight.data.clone())

    def forward(self, indices, fn=module_fn):
        return fn(indices, self.embedding_weight)

def get_inputs():
    indices = torch.randint(0, 100000, (512, 128))
    return [indices]

def get_init_inputs():
    return []
