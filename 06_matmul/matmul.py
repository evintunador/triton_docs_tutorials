"""
This matmul kernel can be a bit confusing but is very crucial to understand

 What you'll learn:
- Automatic performance tuning
- Program re-ordering for improved SRAM hit rate
- Multi-dimensional pointer arithmetic
- High precision data type accumulation
- using the Triton interpreter (kind of)

Recommended order to read the code in:
Step 1 - unit test
Step 2 - wrapper
Step 3 - kernel
Step 4 - benchmark

For matmul of A @ B = C of shapes (M, K) @ (K, N) = (M, N), the following
algorithm is numerically equivalent to what our code will output, but we'll
get to the answer in a different way
for m in range(0, M, BLOCK_SIE_M): # do in parallel
    for n in range(0, N, BLOCK_SIZE_N): # do in parallel
        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
            b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
            acc += dot(a,b)
        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc

see original
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
"""
import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

######### Step 3 #########

# un-comment this to run a numpy emulation of Triton on CPU & be able to debug with print() statements
#import os
#os.environ["TRITON_INTERPRET"] = "1"

# autotuning is just setting up a bunch of different potential meta-parameters configurations that Triton will automatically
# choose from later based on which one performs best on our specific GPU. Triton will figure out for us which one to use. They're 
# all values chosen heuristically, but notice everything is a multiple of 32 in sticking w/ the number of threads in a warp.
autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator which consumes
#   1) a list of `triton.Config` objects that define different configs of meta-parameters and compilation options
#   2) an auto-tuning *key* whose change in values will trigger a new evaluation of all the provided configs, meaning
#       that any time either M, N, or K changes with a new input, Triton will check which config is best all over again
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):
    """
    First we need to map each program id to the block of C it will compute. 
    Let's look at an example where
    M = N = K = 8,
    BLOCK_SIZE_M = BLOCK_SIZE_K = BLOCK_SIZE_N = 2
    A naive implementation might do something like
    [0,   1,  2,  3]
    [4,   5,  6,  7]
    [8,   9, 10, 11]
    [12, 13, 14, 15]
    where each of those PIDs corresponds to a 2x2 block of C (which is of size 8x8). 
    What parts of A and B do we need in order to compone one of them, say 0? 
    Let's look at how matmul works, where x's will denote the blocks in A and B that we need to use to create 
    the block of C corresponding to PID=0
        A           @       B           =       C
    [x, x, x, x]        [x, _, _, _]        [0, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    So in order to create the 2x2 block of C that corresponds to PID=0, we need four 2x2 blocks from the top rows 
    of A and four 2x2 blocks from the first columns of B. 
    Note that rather than loading all 8 blocks into SRAM at the same time, we can iterate over the columns/rows 
    of A/B (respectively), doing a kind of mini matmul between two corresponding blocks and adding them together as we go.
        A           @       B
    [--------->]        [ | , _, _, _]
    [_, _, _, _]        [ | , _, _, _]
    [_, _, _, _]        [ | , _, _, _]
    [_, _, _, _]        [\|/, _, _, _]
    If this fact isn't intuitive, check out `./block_wise_matmul.png`
    Great, if we were to implement this algorithm as described it'd work. 
    However, it would not be nearly as fast as PyTorch's method, which implements something far more clever. 
    To see why, we need to think about our SRAM usage. 
    Notice that PIDs 0 through 3 all utilize the same row of blocks of A, and remember that we can have 
    multiple programs per SM all sharing the same pool of SRAM. 
    That means that rather than each of them loading that same row of blocks of A separately, leading to a bunch of 
    duplicates, once one PID loads a block of A along that row then the other 3 could just re-use it! 
    
    Luckily we do not have to ~explicitly~ tell Triton to do this; every time a PID runs tl.load() it'll first automatically
    check to see if that data already exists in SRAM thanks to some other PID sharing the same SM that got to it first.
    While we don't have to explicitly tell Triton to do this, we should think very carefully about helping Triton take 
    best advantage of this ability by manipulating the orderin gof the PIDs. 
    Importantly, PIDs get assigned to SMs IN ORDER!! 
    To re-state, the order of your PIDs determines which blocks of C get to share SRAM!!
    Let's look again at PIDs 0 through 3, specifically at which blocks of A and B they need to load:
    PID = 0
    [x, x, x, x]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    PID = 1
    [x, x, x, x]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    PID = 2
    [x, x, x, x]        [_, _, x, _]
    [_, _, _, _]        [_, _, x, _]
    [_, _, _, _]        [_, _, x, _]
    [_, _, _, _]        [_, _, x, _]
    PID = 3
    [x, x, x, x]        [_, _, _, x]
    [_, _, _, _]        [_, _, _, x]
    [_, _, _, _]        [_, _, _, x]
    [_, _, _, _]        [_, _, _, x]
    Notice that although they can all share the first row of blocks of A and therefore avoid loading the other three
    rows of blocks, they actually end up loading every single column of blocks of B. 
    Can we do better? 
    Can we get the same number of PIDs (and therefore the same number of blocks of C) to be constructed using fewer 
    total blocks of A and B? 
    Currently, with this method that we'll call "row-major ordering", we're loading:
        (1 row of blocks of A) + (4 columns of blocks of B) = 5 total rows/cols of blocks loaded to SRAM
    
    Well what if instead of putting PIDs 0 through 3 onto the same SM, we could put  PIDs 0, 1, 4, and 5 on the same SM?
    Taking a look at what PIDs 4 and 5 need to load:
    PID = 4
    [_, _, _, _]        [x, _, _, _]
    [x, x, x, x]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    PID = 5
    [_, _, _, _]        [_, x, _, _]
    [x, x, x, x]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    Now suddenly with this hypoethetical new setup, we would only need to load 
        (2 rows of blocks of A) + (2 columns of blocks of B) = 4 total rows/cols of blocks loaded to SRAM
    And yet we're still getting the same number of blocks of C as output! 
    This strategy is called "group-major ordering".
    The effect doesn't seem too huge with this tiny example, but as the number of blocks increases it becomes increasingly 
    effective at saving us from having to do so many duplicate loads of blocks of A and B onto different SMs. 
    
    However, remember that Triton loads blocks into SMs based on the order of PIDs, meaning that even though we'd love it
    if PIDs 0, 1, 4, and 5 all shared the same SRAM, in reality PIDs 3 and 4 are likely going to get in the way of 
    that happening. 
    So how do we ensure the blocks of C corresponding to PIDs 4 and 5 get loaded onto the same SM as 0 and 1? 
    We'll actually have to re-order our PIDs, meaning re-assign them to different blocks of C. 
    Remember our input launch grid is 1-dimensional (like all previous launch grids we've seen), meaning it 
    was defined by a tuple with only one entry. 
    It's our job once inside the kernel to take that 1D list of PIDs and morph them into the shape we desire. 
    I'll reiterate, the key thing to note here is that PIDs that are numerically closer together are more likely to 
    end up on the same SM, meaning that even though we said earlier it'd be great if 0, 1, 4, and 5 all
    shared SRAM, in reality according to our earlier "naive" launch grid, 0, 1, 2, and 3 are going to be grouped together.
    So what we need to do instead is move 2 and 3 such that they correspond to the blocks of C that we previously had
    assigned to 4 and 5. 
    Instead of explaining, check out this new visual ordering:
    [0,  2,  4,  6]
    [1,  3,  5,  7]
    [8, 10, 12, 14]
    [9, 11, 13, 15] 
    Now, 0 through 3 correspond to group-major ordering! Notice in this example we can visualize it as splitting our
    PIDs into "groups" demarcated by the dashed lines
    [0,  2, |  4,  6]
    [1,  3, |  5,  7]
    --------|--------
    [8, 10, | 12, 14]
    [9, 11, | 13, 15] 
    The size of these groups is defined by our "GROUP_SIZE" meta-parameter.
    To get this re-ordering of our PIDs we'll need to do some technically simple but surprisingly difficult to keep 
    track of math. 
    """
    # we start with a 1D launch grid that we will turn into a 2D grid with a complicated "group-wise" ordering
    PID = tl.program_id(axis=0) 
    # defining the size of groups
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M) # the number of blocks along M dimension
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N) # the number of blocks along N dimension
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    # figurinig out which group this PID is in
    group_id = PID // num_PID_in_group 
    # tells us which row to start at for this group
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    # this is usually equal to GROUP_SIZE; the alternative case happens when we're at edge of the tensor and 
    #  its dimensions don't cleanly divde into GROUP_SIZE # TODO is this true?
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    # this is the bulk of the actual mapping of PIDs to group-major ordering
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
        # (PID % num_PID_in_group) puts the current program id into the context of a group
        # (first_PID_in_group_along_m + ...) shifts the PID into the correct group
        # (... % group_size_adj) removes the column component to get us onto the correct row
    PID_N = (PID % num_PID_in_group) // group_size_adj
        # (... // group_size_adj) removes the row component to get us onto the correct column
    
    # Now that the PID nightmare is done we can move onto the kernel code you're more used to seeing.

    # Let's create pointer vectors for the first group of blocks of the input matrices
    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    # in previous lessons the blocks we loaded into SRAM were vectors; here they are matrices
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N
    """
    [:, None] turns [m1,m2,m3] into [[m1],[m2],[m3]] 
    [None, :] turns [n1,n2,n3] into [[n1,n2,n3]]
    combining them gives the matrix
    [[m1n1, m1n2, m1n3],
     [m2n1, m2n2, m2n3],
     [m3n1, m3n2, m3n3]] 
    """

    # inputs tensors are fp16 but we accumulate into a block of fp32 values for higher accuracy (we'll revert later)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # the full C is shape (M, N)
        # for a demonstration of why accumulation works, check out `./block_wise_matmul.png`
        
    # we'll iterate along the K dimension of both A and B to compute a single block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # out-of-bounds entries (along K) need to be masked out
        mask = offsets_K < K - k * BLOCK_SIZE_K
            # k * BLOCK_SIZE_K is the current starting index of offsets_k.
            # so this only really activates when k is within BLOCK_SIZE_K entries from K
            # meaning this gets triggered on the last iteration of the loop, and only if K is not a multiple of BLOCK_SIZE_K
        
        # Now we load blocks of A and B matrices. If multiple blocks in a group are on the same SM, 
        # they can share these loaded values, which reduces the number of expensive loads from DRAM
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0) # shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
            # fill in any masked-out parts with 0.0's since they don't have any effect on the summation in the next step

        # we accumulate along the K dimension
        accumulator = tl.dot(a, b, acc=accumulator)
            # triton is weird with operation notation; this is actually a tiny matmul not a dot product
            #   shape (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N) = (BLOCK_SIZE_M, BLOCK_SIZE_N)
            # `acc` tells Triton to write the output of the matmul directly to accumulator, which is more efficient than
            #   accumulator += tl.dot(a, b)

        # advance the pointers to the next block along K
        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K
        """
        A visual representation of the accumulation movement for PID=0
            A           @       B
        [--------->]        [ | , _, _, _]
        [_, _, _, _]        [ | , _, _, _]
        [_, _, _, _]        [ | , _, _, _]
        [_, _, _, _]        [\|/, _, _, _]
        """

    # and now we reset the data type to the expected output
    accumulator = accumulator.to(tl.float16)

    # write back the block of the output matrix C with masks
    c_offsets = stride_c_M * offsets_M[:, None] + stride_c_N * offsets_N[None, :]
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) # notice the 2D mask
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask) # shape (BLOCK_SIZE_M, BLOCK_SIZE_N)


######### Step 2 #########
def matmul(a, b):
    # check constraints
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    #assert a.is_contiguous() and b.is_contiguous, "input matrices must be contiguous"
    a, b = a.to(torch.float16), b.to(torch.float16)
    
    # get dimesion lengths
    (M, K), (_, N) = a.shape, b.shape

    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # cdiv(x, y) = (x + (y - 1)) // y
    # A naive (slow) launch grid might try to separate our axes of parallelizatio into 2 dimensions, one
    #  for cdiv(M, BLOCK_SIZE_M) and the other for cdiv(N, BLOCK_SIZE_N)
    # Here instead we use a 1D launch kernel defined by cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)
    # The reasoning behind this is explained inside the kernel
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

######### Step 1 #########
def test_matmul_kernel(size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE): # TODO does rtol=0 mean we don't use rtol?
    """
    Here is where we test the wrapper function and kernel that we wrote 
    above to ensure all our values are correct, using pytorch as the 
    correct answer to compare against

    We use higher tolerance values than previous tests because all the flop 
    accumulation can really compound when it comes to a matmul; even slight
    differences in the block size and launch grid ordering from what PyTorch 
    does can result in pretty sizeable discrepancies
    """
    # create input data
    torch.manual_seed(0)
    assert type(size) == tuple and len(size) == 2
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    # run kernel & pytorch reference implementation
    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)
    # compare
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")

######### Step 4 #########
configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"], # we can increase multiple dimensions simultaneously while benchmarking
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["torch", "triton"],
        line_names = ["PyTorch", "Triton"],
        styles = [("green", "-"), ("blue", "-")],
        ylabel = "TFLOPS", 
        plot_name = "matmul-performance",
        args={},
    )
]
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 3 * M * N * K * 1e-12 / (ms * 1e-3)
        # 3 = number of memory operations (2 read + 1 write)
        # M * N * K = number of elements per memory op
        # 1e-12 converts flops to Teraflops
        # 1e-3 converts milliseconds to seconds
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    # always run unit-tests
    test_matmul_kernel(size=(1024, 1024))

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)