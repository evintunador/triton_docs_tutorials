# https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
# this "fused softmax" operation will be significantly faster than pytorch's native op
#   for a particular class of matrices: those whose rows can fit in the GPU's SRAM

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

# first we'll look at how pytorch does it jic you need a refresher
def naive_softmax(x):
    '''
    Built for input of size (M,N)
    we subtract the maximum element in order to avoid numerical overflows when doing .exp()
        softmax is invariant to this shift
    '''
    # read MN elements; write M elements
    x_max = x.max(dim=1)[0] #[0] grabs the values as opposed to the indicees
    # read MN + M lements; write MN elements
    z = x - x_max[:, None]
    # read MN elements; write MN elemnts
    numerator = torch.exp(z)
    # read MN elements; write M elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements; write MN elements
    out = numerator / denominator[:, None]
    # in total 8MN + 4M (read 5MN + 2M elements; wrote 3MN + 2M elements)
    return out

# we'd prefer to have a custom "fused" kernel that only reads x from DRAM once and does all the necessary
# computations on SRAM as opposed to repeatedly reading & writing to DRAM
# that would give a ~4x speedup since 
# (8MN + 4M)/2MN = 4 (ignoring the M term a la big O notation)
# torch.jit.script flag actually aims to do this fusion automatically but can't pull it off quite as well

# our fused softmax kernel works as follows:
# each program (individual call of the kernel) loads a set of rows of the input matrix X which are
#   strided by number of programs, softmaxes it and writes back the result to the output Y

# note an important limitation of Triton is that each block must have a power-of-two number of
#   elements, so we need to internally "pad" each row and guard the memory operations properly

@triton.jit # this decorator tells Triton to compile this function into GPU code
def _softmax_kernel(input_ptr, output_ptr, # raw memory pointers to input and output data
                    input_row_stride, output_row_stride, # number of elements to skip when moving to next row
                    n_rows, n_cols, # matrix dimensions
                    BLOCK_SIZE: tl.constexpr, # power-of-2 size for processing blocks
                    num_stages: tl.constexpr): 
    # num_stages relates to overlapping memory operations with computation operations;
    # when one piece of data is being processed, the GPU can simultaneously load the next piece
    # more stages -> more overlapping, but requires more memory
    # tl.constexpr is a type that tells the compiler that the value must be known at compile-time (not runtime)
    
    # there are multiple "programs" processing data (a program is a unique instantiation of this kernel)
    # programs can be defined along multiple dimensions when the inputs have multiple dimensions
    # this op is 1D so axis=0 is the only option, but bigger operations later may define pid as a tuple
    # here we identify which program we are:
    row_start = tl.program_id(0) 
    # then this gets the total number of parallel programs
    row_step = tl.num_programs(0) 
        # Each program processes rows strided by row_step 
        # (ex. if there are 4 programs, program 0 handles rows 0,4,8...)
    
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # the stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
            # if intuitively you think that input_row_stride should be 1, then in this case you're right,
            #  but what if a view of a manipulated tensor were passed in? for example, if our matrix's rows
            #  were twice the size of what conveniently fits in SRAM, then we'd make a view of that matrix
            #  where each row is split into two rows in order to take advantage of this kernel. we don't 
            #  actually take advantage of this idea in this code; rather we're just writing it this way 
            #  to prepare ourselves for good practices in future lessons
        # the block size is the next power of two greater than n_cols, 
        #   so we can fit each row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf')) # fill in masked out indices with -inf
        # subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
            # all the invalid -inf values remain -inf when we subtract the max
        # note that exponentiation in Triton is fast but approximate
        numerator = tl.exp(row_minus_max)
            # all the -inf values get set to 0 since exp(-inf)=0
        denominator = tl.sum(numerator, axis=0)
            # all the invalid 0 values do get summed but don't matter since they're 0
        softmax_output = numerator / denominator
            # all the invalid 0's are 0/sum and therefore remain 0
        # write output back to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
            # using our mask we only store back the valid values

# and now we'll create a helper function that enqueues the kernel and its meta-arguments
#   for any given input tensor. these properties will be used in the helper function to calculate
#   how many parallel programs we can run efficiently
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"] 
    # each Streaming Multi-processor (SM) is like a mini-processor that can run multiple programs
NUM_REGS = properties["max_num_regs"] # registers are the fastest memory on the GPU
    # each SM has a limited number of registers; 
    # programs share these registers, so using too many per program limits parallelism
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] # this is total SRAM; each SM has a fixed amount of SRAM
WARP_SIZE = properties["warpSize"]
    # a warp is a group of threads that execute together; usually 32 on nvidia GPUs and 64 on AMD
target = triton.runtime.driver.active.get_current_target() # TODO
kernels = {} # this would be used for caching compiled kernels if we were running multiple operations # TODO

def softmax(x):
    n_rows, n_cols = x.shape

    # the block size of each loop iteration is the smallest power of 2 greater than the
    #   number of columns in x
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # another trick we can use is to ask the compiler to use more threads per row by
    #   increasing the number of warps (`num_warps`) over which each row is distributed.
    # for now these settings are just a heuristic
    # you will see in the next tutorial how to auto-tune this value in a more natural way
    #   so you don't have to come up with manual heuristics yourself
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # number of software pipelining stages, meaning how we let the GPU do multiple things at once
    # so with 2 stages we can have one do the operation while the other is loading the next operands into memory
    # with 4 we can have one do operations, one load next operands, one saving previous operands, 
    #   and one pre-loading future operands
    # Triton just needs the number of stages and it'll handle how to use them efficiently
    # here we use a simple heuristic of "if we've got a lot of memory, use 4. otherwise use 2"
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    # allocate output
    y = torch.empty_like(x)

    # .warmup() pre-compiles kernel and tells us how many registers and how much shared memory it needs
    kernel = _softmax_kernel.warmup(x, y,
                                    x.stride(0), y.stride(0),
                                    n_rows, n_cols,
                                    BLOCK_SIZE=BLOCK_SIZE,
                                    num_stages=num_stages,
                                    num_warps=num_warps, # @triton.jit has extra arguments we didnt' define
                                    grid=(1,))
    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared 

    # register-based occupancy
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        # each SM has NUM_REGS registers (eg 65536)
        # each program uses
            # n_regs per register thread (eg 32)
            # WARP_SIZE threads per warp (32 on Nvidia)
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
    # then given our number of SMs, we calculate ho wmany programs to run in total
    num_programs = min(NUM_SM * programs_per_sm, n_rows)
        # ofc we have another limit since we've got no need to surpass the n_rows in the matrix

    # grid configuration
    grid_config = (num_programs, 1, 1)
        # first dimension: number of programs in x-directoin
        # second? number of prgrams in y-direction: 
        # etc
        # our data parallelism is only along rows (first dimension) so we don't need 2-3D parallelism
            # for matrix multiplication you would use (M, N, 1)
            # for 3D convolution you'd use (X, Y, Z)
        # even if you only need 1D, it's normal to specify all three for consistency & clarity

    # create a number of persistent programs
    kernel[grid_config](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
    )
    return y

# test our kernel on a matrix w/ an irregular number of rows & cols to verify that our padding mechanism works
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch) # shouldn't print anything if successful

# benchmark
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
        args={'M': 4096} # values for function arguments not in x_names and y_name
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
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
        # ms * 1e-3 converts milliseconds to seconds
    return gbps(ms)

benchmark.run(print_data=False, save_path='.')