import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        # BUG FIX: 原代码使用 [:-1] 会错误地在第一个维度切片，导致 batch size 减少
        #
        # 原代码:
        #   exclusive_cumsum = torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), x), dim=self.dim)[:-1]
        #   return torch.cumsum(exclusive_cumsum, dim=self.dim)
        #
        # 问题: 输入 [128, 4000] → cat 后 [128, 4001] → [:-1] 后 [127, 4001] (错误!)
        # 修复: 使用 narrow 在正确的维度上切片，保持输出形状与输入一致
        zeros = torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim))
        padded = torch.cat((zeros, x), dim=self.dim)
        exclusive_cumsum = padded.narrow(self.dim, 0, x.size(self.dim))
        return torch.cumsum(exclusive_cumsum, dim=self.dim)

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
