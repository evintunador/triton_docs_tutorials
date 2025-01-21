import torch
import triton
import triton.language as tl

x = torch.randn((2,3,4))
print(x.stride())

x = x.sum(axis=1)
print(x.shape)