"""
This "fused softmax" kernel only works on matrices whose rows fit in the GPU's SRAM.

What you'll learn:
- The importance of reducing memory reads/writes
- How to fuse multiple operations into one kernel to reduce memory reads/writes
- How to fetch GPU specifications
- Some parts of the GPU architecture that you don't usually have to think about 
    when writing Triton kernels
- How to define meta-parameters using GPU-specific attributes and rough heuristics
- Pipeline parallelism & the weird way that for-loops work within GPU kernels
- How to choose the value of extra entries when masking

Recommended order to read the code in:
Step 1 - naive implementation
Step 2 - unit test
Step 3 - wrapper
Step 4 - kernel
Step 5 - benchmark

watch the accompanying YouTube video:
https://youtu.be/ftknUZDQCPc
see original triton documentation:
https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
"""
import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

######### Step 1 #########
# first we'll look at the naive implementation jic you need a refresher
def naive_softmax(x):
    '''
    Built for input of size (M,N)
    Safe softmax is when we subtract the maximum element in order to avoid numerical 
    overflows when doing .exp(); softmax is invariant to this shift
    '''
    # read MN elements, find their max along N, and write M elements (the maxes)
    x_max = x.max(dim=1)[0] 
        # pytorch actually outputs a tuple of (values, indices) so [0] grabs the values;
        # we ignored the indices when talking about memory writes above
    # read MN + M elements, subtraction is MN flops, and write MN elements
    z = x - x_max[:, None]
    # read MN elements and write MN elemnts
    numerator = torch.exp(z)
        # exp is actually a lot of flops per element but we're only worried about mem ops rn
    # read MN elements, do MN flops to find M sum values, and then write M elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements, division is MN flops, then write MN elements
    out = numerator / denominator[:, None]

    # in total we did 8MN + 4M memory operations
    # (read 5MN + 2M elements; wrote 3MN + 2M elements)
    return out
"""
that's a whole lot of memory operations. we'd prefer to have a custom "fused" kernel that only 
reads x from DRAM once and does all the necessary computations on SRAM as opposed to repeatedly 
reading & writing to DRAM. that would give a ~4x speedup since 
(8MN + 4M)/2MN = 4 (ignoring the solo M term a la big O notation)

torch.jit.script flag and torch.compile actually aim to do this fusion automatically but can't 
pull it off quite as well as we're about to

our fused softmax kernel will work as follows:
each program (individual call of the kernel) loads a set of rows of the input matrix X which are
  strided by number of programs, softmaxes it and writes back the result to the output Y

note an important limitation of Triton is that each block must have a power-of-two number of
  elements, so we need to internally "pad" each row and guard the memory operations properly
"""

######### Step 4 #########
@triton.jit 
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,    # number of elements to skip when moving to next row
    n_rows, n_cols,                         # matrix dimensions
    BLOCK_SIZE: tl.constexpr,               # lowest power-of-2 greater than n_cols
    num_stages: tl.constexpr,
): 
    # the row that this program starts with is defined by the pid
    row_start = tl.program_id(0) 
    # then this gets the total number of parallel programs, which we'll use to know how large 
    #  of a step to make in our for loop once we finish the first row
    row_step = tl.num_programs(0) 
        # Each program processes rows strided by row_step 
        # (ex. if there are 4 programs, program 0 handles rows 0,4,8...)
    
    # whereas tl.arange() provides an array of values, tl.range() acts as an iterator
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # rather than actually implement each iteration of the for loop sequentially, triton can use
        #  num_stages to work on different interations of the for loop simultaneously. Of course
        #  only do this when the iterations don't depend on each other
        
        # the stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
            # inyuiyively input_row_stride should be 1 as long as the input tensor is contiguous.
            #  but what if a non-contiguous view of a manipulated tensor were passed in? then
            #  input_row_stride matters

        # load the row into SRAM, using a mask since BLOCK_SIZE is > than n_cols if n_cols is not a power of 2
        col_offsets = tl.arange(0, BLOCK_SIZE) # we can fit each row in a single block
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float('-inf')) 
            # we fill in masked out indices with -inf since that's the value that won't influence softmax

        # subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
            # all the invalid -inf values remain -inf when we subtract the max
        # note that exponentiation in Triton is fast but approximate; later we'll learn an even faster alternative
        numerator = tl.exp(row_minus_max)
            # all the -inf values get set to 0 since exp(-inf)=0
        denominator = tl.sum(numerator, axis=0)
            # all the invalid 0 values do get summed but don't matter since they're 0
        softmax_output = numerator / denominator
            # all the invalid 0's are 0/sum and therefore remain 0

        # write output back to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
            # using our mask we only store back the valid n_cols values

######### Step 3 #########
"""
before we create the wrapper function that enqueues the kernel and its meta-parameters, we're going to
 fetch the specifications of our GPU to help later when defining our meta-parameters such that they're 
 especially well suited (fast) to the specific GPU we're using
"""
# fetching a dictionary full of the GPU's specifications
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
# each Streaming Multi-processor (SM) is like a mini-processor that can run multiple programs
NUM_SM = properties["multiprocessor_count"] 
# registers are the fastest memory on the GPU
NUM_REGS = properties["max_num_regs"] 
    # each SM has a limited number of registers; 
    # programs share these registers, so using too many per program limits parallelism
# each SM has a dedicated pool of SRAM that it can access
# since there can be multiple programs per SM, those programs share the same SRAM
    # ^that will be very useful information later in the matmul tutorial
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] 
# a warp is a group of threads that execute together
# a thread can be thought of as analagous to a single CPU core, but far more limited in the operations it can do
WARP_SIZE = properties["warpSize"]# usually 32 on nvidia GPUs and 64 on AMD

def softmax(x):
    '''
    helper/wrapper function to 
        1) allocate the output tensor and 
        2) enque the above kernel with appropriate grid/block sizes
    
    This wrapper function does not connect us to pytorch's graph, meaning it does not
    support backpropogation. That (as well as a backward pass kernel) is for a future lesson
    '''
    # this kernel is only built to support matrices; expanding that support is simple but for a later lesson
    assert x.ndim == 2
    n_rows, n_cols = x.shape

    # the block size is the smallest power of 2 greater than the number of columns in x
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # a trick we can use is to ask the compiler to use more threads per row by
    #  increasing the number of warps (`num_warps`) over which each row is distributed.
    # for now these settings are just a heuristic
    # you will see in the next tutorial how to auto-tune this value in a more natural way
    #   so you don't have to come up with manual heuristics yourself
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # Rather than executing all code within a kernel sequentially, the GPU can actually do multiple things at once.
    # This is called the number of software pipelining stages.
    # For example, with 2 stages we can have one do the operation while the other is loading the next operands 
    #  from DRAM into SRAM. With 3 we can have one do current operations, one load next operands, and one saving 
    #  previous operands.
    # Triton just needs the number of stages and it'll handle how to use them efficiently.
    # Here we use a simple heuristic of "if we've got a lot of memory, use 4. otherwise use 2"
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    # allocate output
    y = torch.empty_like(x)

    # .warmup() pre-compiles kernel and tells us how many registers and how much shared memory it needs
    kernel = _softmax_kernel.warmup(x, y, # this warmup depends on the attributes of the input and output
                                    x.stride(0), y.stride(0), # see below
                                    n_rows, n_cols,
                                    BLOCK_SIZE=BLOCK_SIZE,
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                    grid=(1,))
    # x.stride() for each dimension tells us how many entries in memory a pointer needs to move forward in order
    #  to get to the next element of the tensor along the specified dimension. 
    # For any tensor x that is "contiguous", meaning ~cleanly/simply~ defined in memory and for a shape (M, N, K) 
    #  you can expect x.shape(0) == N*K, x.shape(1)==K, and x.shape(2)==1, or more generally 
    #  x.shape(-Z)==math.prod(x.shape[-Z:])
    # A tensor might be non-contiguous if, for example, it's been saved to memory using torch.view() or some similar
    #  operation that leaves the original data in place but messes with dimensions

    # here's the info that warmup process gave us
    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared 

    # and here's how we use that info to setup our kernel
    # register-based occupancy
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        # each SM has NUM_REGS registers (eg 65536)
        # each program uses
            # n_regs per register thread (eg 32)
            # WARP_SIZE threads per warp (32 on Nvidia, 64 on AMD)
            # num_warps warps per program (4, 8, or 16 in our case with the aforementioned heuristic)
        # so each program needs n_regs * WARP_SIZE * num_warps registers total
        # therefore we can fit reg_occupancy programs per SM
        # ex. 65536 // (32 * 32 * 8) = 8 programs per SM (assuming num_warps=8)
    # shared memory-based occupancy
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    # determines how many programs can run per SM based on register usage and shared memory usage
    programs_per_sm = min(reg_occupancy, sram_occupancy)
        # the former is the optimal allocation assuming we have more than enough SRAM
        # the latter is our limit on SRAM when splitting it equally among all SMs
    # then given our number of SMs, we calculate how many programs to run in total
    num_programs = min(NUM_SM * programs_per_sm, n_rows)
        # ofc we have another limit since we've got no need to surpass the n_rows in the matrix

    # grid configuration; each row gets its own program
    grid = (num_programs, 1, 1)
        # the extra 1's are usually not necessary if they're not being used
        # we use them here because the .warmup() we used earlier has a weird quirk in the way
        #  it's implemented that forces only 3D launch grids to be inputted once it's been used
        # in future lessons we don't use .warmup() so we'll not be required to do this again

    # And now we get to run the kernel with our heuristics-based launch grid
    kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
    )
    return y

######### Step 2 #########
def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    """
    Here is where we test the wrapper function and kernel that we wrote 
    above to ensure all our values are correct, using pytorch as the 
    correct answer to compare against

    we'll use an irregular number of rows & cols to verify that our padding mechanism works
    """
    # create input data
    torch.manual_seed(0)
    assert type(size) is tuple and len(size) == 2
    x = torch.randn(size[0], size[1], device=DEVICE)
    # run kernel & pytorch reference implementation
    z_tri = softmax(x)
    z_ref = torch.softmax(x, axis=1)
        # notice our implementation doesn't give a choice for what axis to softmax along.
        # this is a common theme of custom GPU kernels; because pytorch has to write code that
        #  is more general, it is slower than it could be
    # compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")

######### Step 5 #########
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=["Triton", "Torch"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={'M': 4096} # values for function arguments not in x_names
    ))
def benchmark(M, N, provider):
    # making the input data
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    # these two lines ensure more accurate benchmarks; i usually forget to use them but it's not a big deal
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        # 2 = number of memory operations (1 read + 1 write)
        # x.numel() = number of elements
        # x.element_size() = bytes per element (4 for float32)
        # 1e-9 converts bytes to GB
        # 1e-3 converts milliseconds to seconds
    return gbps(ms)

if __name__ == "__main__":
    # always run unit-tests
    test_softmax_kernel(size=(1823, 781))

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)