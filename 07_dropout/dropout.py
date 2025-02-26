"""
This tutorial on low-memory dropout required the least editing of all the original Triton documentation tutorials

What you'll learn:
- Parallel pseudo-random number generation
"""
import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p, # a float32 probability, so range [0,1]
    seed, # a single int32
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this program
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask) # shape (BLOCK_SIZE)
    # the key insight is that we generate and use a mask entirely in SRAM without ever having to store it in DRAM.
    # this line generates uniformly distributed float32 values in [0, 1), given a seed and a block of int32 oﬀsets
    random = tl.rand(seed, offsets) # shape (BLOCK_SIZE)
    # prune based on our desired probability threshold
    x_keep = random > p # values are either true or false
    output = tl.where(x_keep, x / (1 - p), 0.0)
        # where x_keep is True, the value is x/(1-p), and where False it's 0.0
    # write-back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output

x = torch.randn(size=(8, ), device=DEVICE)
output1 = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)
print(x, output1, output2, output3, sep="\n")
