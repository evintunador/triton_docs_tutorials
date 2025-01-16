# https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py
import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
print(DEVICE)

@triton.jit # this decorator tells Triton to compile this function into GPU code
def add_kernel(x_ptr, y_ptr,# pointers to input vectors
               output_ptr, # ptr to output vector
                    # each torch.tensor object is implicitly converted into a pointer to its first element
               n_elements, # size of vector
               BLOCK_SIZE: tl.constexpr): # number of elements each program should process
    # tl.constexpr is a type that tells the compiler that the value must be known at compile-time (not runtime)
    # there are multiple "programs" processing data (a program is a unique instantiation of this kernel)
    # programs can be defined along multiple dimensions when the inputs have multiple dimensions
    # this op is 1D so axis=0 is the only option, but bigger operations later may define program_id as a tuple
    # here we identify which program we are:
    program_id = tl.program_id(axis=0) 
        # Each program instance gets a unique ID along the specified axis
        # For a vector of length 256 and BLOCK_SIZE=64:
        # program_id=0 processes elements [0:64]
        # program_id=1 processes elements [64:128]
        # program_id=2 processes elements [128:192]
        # program_id=3 processes elements [192:256]

    # this program will process inputs that are offset from the initial data (^ described above)
    # note that offsets is a list of pointers a la [0, 1, 2, ...., 62, 63]
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements

    # load x and y from DRAM (global GPU memory) into SRAM (on-chip memory)
    # SRAM is much faster but limited in size
    # The mask ensures we don't access memory beyond the vector's end
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # perform the operation on SRAM
    # triton has its own internal definitions of all the basic ops
    output = x + y

    # write back to DRAM, being sure to mask to avoid out-of-bounds accesses
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    '''
    helper/wrapper function to 
        1) allocate the output tensor and 
        2) enque the above kernel with appropriate grid/block sizes
    '''
    # preallocating the output
    output = torch.empty_like(x)

    # Ensures all tensors are on the same GPU device
    # This is crucial because Triton kernels can't automatically move data between devices
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE,\
        f'DEVICE: {DEVICE}, x.device: {x.device}, y.device: {y.device}, output.device: {output.device}'
    
    # getting length of the vectors
    n_elements = output.numel() # .numel() returns total number of entries in tensor of any shape
 
    # grid defines the number of kernel instances that run in parallel
    # it can be either Tuple[int] or Callable(metaparameters) -> Tuple[int]
    # in this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), ) 
        # so 'BLOCK_SIZE' is a parameter to be passed into meta() at compile-time, not runtime
        # triton.cdiv = ceil(n_elements / meta['BLOCK_SIZE'])
        # then meta() returns a Tuple with the number of kernel programs we want to instantiate at once
        #   which are compile-time constants
    
    # `triton.jit`'ed functionis can be indexed with a launch grid to obtain a callable GPU kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        # BLOCK_SIZE of 1024 is a common heuristic choice because:
        # It's a power of 2 (efficient for memory access patterns)
        # It's large enough to hide memory latency
        # It's small enough to allow multiple blocks to run concurrently on a GPU
    
    # the kernel writes to the output in-place rather than having to return anything
    # once all the kernel programs have finished running then the output gets returned here
    return output

# we can now use the above function to comput ethe element-wise sum of two torch.tensor objects
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The max diff bw torch & triton is: '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

# BENCHMARK
# triton has a set of built-in utilities that make it easy for us to plot performance of custom ops
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)], # different possible values for x_name
        x_log = True, # makes x-axis logarithmic
        line_arg='provider', # title of the legend 
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s', # label name for y-axis
        plot_name='vector-add-performance', # also used as file name for saving plot
        args={}, # values for funciton arguments not in x_names and y_names; need it even if not using
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        # 3 = number of memory operations (2 reads + 1 write)
        # x.numel() = number of elements
        # x.element_size() = bytes per element (4 for float32)
        # 1e-9 converts bytes to GB
        # ms * 1e-3 converts milliseconds to seconds
    return gbps(ms), gbps(max_ms), gbps(min_ms)
benchmark.run(print_data=True, save_path='./benchmark_results/') # show_plots=True, 