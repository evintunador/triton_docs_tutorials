"""
In this document we'll implement basically the simplest possible Triton GPU Kernel, which does entry-wise
addition for vectors. 

What you'll learn:
- How to build a test to ensure your Triton kernels are numerically correct
- Basics of Triton kernels (syntax, pointers, launch grids, DRAM vs SRAM, etc)
- How to benchmark your Triton kernels against PyTorch

Recommended order to read the code in:
Step 1 - unit test
Step 2 - wrapper
Step 3 - kernel
Step 4 - benchmark

watch the accompanying YouTube video:
https://youtu.be/fYMS4IglLgg
see original triton documentation:
https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py
"""
import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

######### Step 3 #########
# this `triton.jit` decorator tells Triton to compile this function into GPU code
@triton.jit # only a subset of python capabilities are useable within a triton kernel
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr): 
    """
    This entry-wise addition kernel is relativevly simple; it's designed to only
     take in vectors as input and does not support any kind of broadcasting

    Each torch.tensor object that gets passed into a Triton kernelis implicitly 
     converted into a pointer to its first element

    x_ptr:          pointer to first entry of input vector of shape (n_elements)
    y_ptr:          pointer to first entry of input vector of shape (n_elements)
    output_ptr:     pointer to first entry of output vector of shape (n_elements)
    n_elements:     size of our vectors
    BLOCK_SIZE:     number of elements each kernel instance should process; should be a power of 2
    
    tl.constexpr designates BLOCK_SIZE as a compile-time variable (rather than run-time), 
     meaning that every time a different value for BLOCK_SIZE is passed in you're actually 
     creating an entirely separate kernel. I may sometimes refer to arguments with this 
     designation as "meta-parameters"
    """
    # There are multiple "programs" processing data; a program is a unique instantiation of this kernel.
    # Programs can be defined along multiple dimensions (defined by your launch grid in the wrapper).
    # this op is 1D so axis=0 is the only option, but bigger operations later may define program_id as a tuple
    # here we identify which program we are:
    pid = tl.program_id(axis=0) 
        # Each program instance gets a unique ID along the specified axis
        # For example, for a vector of length 256 and BLOCK_SIZE=64:
        # pid=0 might processe elements [0:64]
        # pid=1 might processe elements [64:128]
        # pid=2 might processe elements [128:192]
        # pid=3 might processe elements [192:256]
        # I said "might" because the specific elements that get processed depend on the code below

    # herewe tell the program to process inputs that are offset from the initial data (^ described above)
    block_start = pid * BLOCK_SIZE

    # offsets is an array of int32 that act as pointers
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # sticking with the above example, if pid=1 then offsets=[64, 65, 66, ...., 126, 127]

    # create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements
        # if we didn't do this AND n_elements were not a multiple of BLOCK_SIZE then
        #  the kernel might read entries that are actually part of some other tensor in memory
        #  and use them in calculations

    # here load x and y 
        # from (DRAM / VRAM / global GPU memory / high-bandwidth memory) which is slow to access
        # onto (SRAM / on-chip memory) which is much faster but very limited in size.
        # We store data not currently in-use on DRAM and do calculations on data that's in SRAM
    x = tl.load(x_ptr + offsets, mask=mask, other=None) # shape (BLOCK_SIZE)
    y = tl.load(y_ptr + offsets, mask=mask, other=None) # shape (BLOCK_SIZE)
        # The mask ensures we don't access memory beyond the vector's end.
        # `other` refers to what value to put in place of any masked-out values; it defaults
        #  to None (so we didn't have to actually write it here) but depending on the operation
        #  it may make more sense to use a value like 0.0 (we'll see this in a later tutorial)
        # Whenever you see a tl.load that is a memory operation which is expensive so we want to
        #  keep track of how many memory operations we do. We count them by the total number of
        #  entries being read/written to memory, in this case BLOCK_SIZE per kernel and 
        #  therefore n_elements in total across all running kernels for EACH of these two lines

    # here we perform the operation on SRAM
    # triton has its own internal definitions of all the basic ops that you'll need
    output = x + y
        # For the masked-out entries, None + None = None (really no operation happens at all).
        # Similar to keeping track of memory operations, we also keep track of floating point 
        #  operations (flops) using the shape of the blocks involved. Here this line does BLOCK_SIZE
        #  flops for each pid, meaning n_elements flops total


    # write back to DRAM, being sure to mask in order to avoid out-of-bounds accesses
    tl.store(output_ptr + offsets, output, mask=mask)
        # here is a memory write operation of size BLOCK_SIZE per pid and therefore n_elements 
        # in aggregate across all pids combined

######### Step 2 #########
def add(x: torch.Tensor, y: torch.Tensor):
    '''
    helper/wrapper function to 
        1) allocate the output tensor and 
        2) enque the above kernel with appropriate grid/block sizes
    
    This wrapper function does not connect us to pytorch's graph, meaning it does not
    support backpropogation. That (as well as a backward pass kernel) is for a future lesson
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
        # triton.cdiv = (n_elements + (BLOCK_SIZE - 1)) // BLOCK_SIZE
        # then meta() returns a Tuple with the number of kernel programs we want to 
        #  instantiate at once which is a compile-time constant, meaning that if it
        #  changes Triton will actually create an entirely new kernel for that value
    
    # `triton.jit`'ed functionis can be indexed with a launch grid to obtain a callable GPU kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        # BLOCK_SIZE of 1024 is a heuristic choice
        # It's a power of 2 (efficient for memory access patterns)
        # It's large enough to hide memory latency
        # It's small enough to allow multiple blocks to run concurrently on a GPU
        # in a later lesson we'll learn better methods than heuristics
    
    # the kernel writes to the output in-place rather than having to return anything
    # once all the kernel programs have finished running then the output gets returned here
    return output

######### Step 1 #########
def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    """
    Here is where we test the wrapper function and kernel that we wrote 
    above to ensure all our values are correct, using pytorch as the 
    correct answer to compare against
    """
    # create data
    torch.manual_seed(0)
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    # run kernel & pytorch reference implementation
    z_tri = add(x, y)
    z_ref = x + y
    # compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")

######### Step 4 #########
# Triton has a set of built-in utilities that make it easy for us to plot performance of custom ops.
# This decorator tells Triton that the below function is a benchmark and what benchmark conditions to run
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)], # different values of x_names to benchmark
        x_log = True, # makes x-axis logarithmic
        line_arg='provider', # title of the legend 
        line_vals=['triton', 'torch'], # designators of the different entries in the legend
        line_names=['Triton', 'Torch'], # names to visibly go in the legend
        styles=[('blue', '-'), ('green', '-')], # triton will be blue; pytorch will be green
        ylabel='GB/s', # label name for y-axis
        plot_name='vector-add-performance', # also used as file name for saving plot
        args={}, # we'll see how this is used in a later tutorial; need it even if it's empty
    )
)
def benchmark(size, provider):
    # creating our input data
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    # each benchmark runs multiple times and quantiles tells matplotlib what confidence intervals to plot
    quantiles = [0.5, 0.05, 0.95]
    # defining which function this benchmark instance runs
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    # turning the raw millisecond measurement into meaninful units
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        # 3 = number of memory operations (2 reads + 1 write)
        # x.numel() = number of elements
        # x.element_size() = bytes per element (4 for float32, 2 for float16)
        # 1e-9 converts bytes to GB
        # 1e-3 converts milliseconds to seconds
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    # always run unit-tests
    test_add_kernel(size=98432)

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
