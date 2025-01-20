import torch
import triton
import triton.language as tl

x = torch.randn((2,3,4))
print(x.stride())
