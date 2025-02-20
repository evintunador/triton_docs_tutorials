"""
This matmul kernel can be a bit confusing but is very crucial to understand

 What you'll learn:
- Multi-dimensional pointer arithmetic
- Automatic performance tuning
- High precision data type accumulation
- Program re-ordering for improved SRAM hit rate

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
    stride_am, stride_ak, # how much to increase the ptr by when moving by 1 element along that dimension
    stride_bk, stride_bn, # ex: increase b_ptr by stride_bk to get the element one row down (B has K rows)
    stride_cm, stride_cn,
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # first we map program ids (pids) to the block of C it should compute
    # this is done in grouped ordering to promote SRAM data reuse (pids within a group share data along k dimension)
    # for a visual see https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    # Each GPU has multiple Streaming Multiprocessors (SMs), and each SM has its own SRAM.
    # Thread blocks (represented by PIDs) that run on the same SM can share SRAM, but blocks on different SMs cannot.
    # Triton is smart enough to reuse data already loaded onto the same SRAM than load on a duplicate copy.
    # The grouping strategy below tries to arrange the computation so that blocks that need the same data
    # are more likely to run on the same SM, allowing them to share data through SRAM.
    # we start with a 1D launch grid that we will turn into a 2D grid WITH A COMPLICATED ORDERING
    pid = tl.program_id(axis=0) 
    num_pid_along_m = tl.cdiv(M, BLOCK_SIZE_M) # the number of blocks along M dimension
    num_pid_along_n = tl.cdiv(N, BLOCK_SIZE_N) # the number of blocks along N dimension
    num_pid_in_group = GROUP_SIZE * num_pid_along_n 
        # so for the GROUP_SIZE number of rows we'll be using, we need to grab a matching set of columns, specifically
        # we'll grab GROUP_SIZE number of columns each with num_pid_along_n number of blocks in them.
        # meaning these are groups OF BLOCKS
    group_id = pid // num_pid_in_group # figurinig out which group we are
    first_pid_in_group_along_m = group_id * GROUP_SIZE # tells us which row to start at for this group
    group_size_adj = min(num_pid_along_m - first_pid_in_group_along_m, GROUP_SIZE) # usually equal to GROUP_SIZE.
        # alternative case happens when we're at edge of tensor and tensor dimensions don't cleanly divde into GROUP_SIZE
        # ^ is this true? idk

    # this is the bulk of the actual mapping from 1D to 2D - the weird pattern that is the reason we didn't use a 2D grid
    # (pid % num_pid_in_group) puts the current program id into the context of a group
    pid_m = first_pid_in_group_along_m + ((pid % num_pid_in_group) % group_size_adj)
        # (first_pid_in_group_along_m +) shifts the pid into the correct group
        # (% group_size_adj) removes the column component to get us onto the correct row
    pid_n = (pid % num_pid_in_group) // group_size_adj
        # (// group_size_adj) removes the row component to get us onto the correct column
    """
    as an example, pick any PID from the first matrix and follow it through the above calculations to the second matrix
    M=N=K=8
    BLOCK_SIZE_M=BLOCK_SIZE_N=BLOCK_SIZE_K=2
    GROUP_SIZE=2
    # naive 1D placement of pid's
    [' 0', ' 1', ' 2', ' 3']
    [' 4', ' 5', ' 6', ' 7']
    [' 8', ' 9', '10', '11']
    [' 12', '13', '14', '15']
    # final 2D pid access grid
    [' 0', ' 2', ' 4', ' 6']
    [' 1', ' 3', ' 5', ' 7']
    [' 8', '10', '12', '14']
    [' 9', '11', '13', '15'] 
    note that the pid number is the order in which these get loaded into an SM and onto SRAM, meaning that
    if a given SM can handle 4 pid's, then 0, 1, 2, and 3 will all get loaded onto the same one. Since 0 and 1
    make tl.load() calls to the same columns, first 0 will load it and then when 1 goes to load it Triton will
    say "oh, it's already here, just use that" and same for the column that 2 & 3 share as well as the row that
    0 & 2 share and the row that 1 & 3 share. It's not guaranteed that the SM handles an ideal number of pid's 
    for your matrix shape and the way you loaded it in, but that's why we did autotuning earlier, to find 
    ideal block and group sizes given the properties of our hardware, such as the number of pid's an SM can handle.
    """

    # now we'll create pointer vectors for the first group of blocks of the input matrices
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    offsets_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) #% M <- not sure why that was originally here, edge case?
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offsets_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) #% N <- not sure why that was originally here, edge case?
    offsets_k = tl.arange(0, BLOCK_SIZE_K) # this is used to setup k dimension of initial a & b offsets
    # now we convert our group/block/pid based indices to their actual tensor mappings using first-entry pointers & stride length
    a_ptrs = a_ptr + (offsets_am.expand_dims(1) * stride_am + offsets_k.expand_dims(0) * stride_ak)
    b_ptrs = b_ptr + (offsets_k.expand_dims(1) * stride_bk + offsets_bn.expand_dims(0) * stride_bn)
        
    # iterate to compute a block of the C matrix
    # inputs are fp16. we accumulate into a block of fp32 values for higher accuracy
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)): # iterate from 0 to the number of blocks along K dimension

        # out-of-bounds entries (along K) need to be masked out
        mask = offsets_k < K - k * BLOCK_SIZE_K
            # k * BLOCK_SIZE_K is the current starting index of offsets_k.
            # so this only really activates when K is within BLOCK_SIZE_K entries from the starting index.
            # when that does happen, suddenly how K - k * BLOCK_SIZE_K gives us a number of entries <= BLOCK_SIZE_K
            # meaning anything in offsets_k above this value needs to be masked out.
            # so this gets triggered on the last iteration of the loop, and only when K is not a multiple of BLOCK_SIZE_K
        
        # Now we load blocks of A and B matrices. If multiple blocks in a group are on the same SM, 
        # they can share these loaded values, which reduces the number of expensive loads from DRAM
        a = tl.load(a_ptrs, mask=mask.expand_dims(0), other=0.0)
        b = tl.load(b_ptrs, mask=mask.expand_dims(1), other=0.0)
            # fill in any masked-out parts with 0.0
            # 0.0's don't have any effect on the summation in the next step

        # we accumulate along the K dimension
        accumulator = tl.dot(a, b, accumulator)
            # triton is weird with operation notation; this is actually a tiny matmul not just a dot product

        # advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(tl.float16)

    # write back the block of the output matrix C with masks
    offsets_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offsets_cm.expand_dims(1) + stride_cn * offsets_cn.expand_dims(0)
    c_mask = (offsets_cm.expand_dims(1) < M) & (offsets_cn.expand_dims(0) < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


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
    
    # 1D launch kernel where each block gets its own program; this is explained inside the kernel
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE_M']) * triton.cdiv(n, meta['BLOCK_SIZE_N']), )
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
    assert type(size) == tuple and len(size == 2)
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    # run kernel & pytorch reference implementation
    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)
    # compare
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")

# benchmark
configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"],
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["cublas", "triton"],
        line_names = ["cuBLAS", "Triton"],
        styles = [("green", "-"), ("blue", "-")],
        ylabel = "TFLOPS", 
        plot_name = "matmul-performance",
        args={}, # values for funciton arguments not in x_names and y_names; need it even if not using
    )
]
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        # 2 = number of memory operations (1 read + 1 write)
        # M * N * K = number of elements
        # 1e-12 converts flops to Teraflops
        # ms * 1e-3 converts milliseconds to seconds
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    # always run unit-tests
    test_matmul_kernel(size=(512, 512))

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)