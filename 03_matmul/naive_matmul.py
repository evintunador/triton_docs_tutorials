import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _naive_matmul_kernel(
    a_ptr, b_ptr, c_ptr, # pointers to first element of the respective tensors
    m, n, k, # columns and rows
    stride_am, stride_ak, # number of location steps to jump in order to move to the next entry in that dimension
    stride_bk, stride_bn, # ex: increase b_ptr by stride_bk to get the element one row down (B has K rows)
    stride_cm, stride_cn,
    # meta-parameters; solidified at compiletime rather than runtime 
    block_size_m: tl.constexpr, block_size_n: tl.constexpr, block_size_k: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # figuring out which program we are
    program_id_m, program_id_n = tl.program_id(0), tl.program_id(1)

    # chunks along m/n/k dimensions (in the case of m & n, it's with respect to the desired output location on c?)
    chunk_m = program_id_m * block_size_m + tl.arange(0, block_size_m)
    chunk_n = program_id_n * block_size_n + tl.arange(0, block_size_n)
    chunk_k = tl.arange(0, block_size_k)

    # relevant offsets of a and b
    offsets_a = a_ptr + chunk_m.expand_dims(1) * stride_am + chunk_k.expand_dims(0) * stride_ak
    offsets_b = b_ptr + chunk_k.expand_dims(1) * stride_bk + chunk_n.expand_dims(0) * stride_bn
        # expand_dims(1) turns [,,,] into [[],[],[]] 
        # expand_dims(0) turns [,,,] into [[,,,]]
        # creates 2D indices where top-left is lowest number, bottom-right is highest, and 
        # change each direction is determined by stride length of that dimension

    # make a mask to prevent utilizing threads that go out-of-bounds of the matrix's dimensions
    mask = (chunk_m.expand_dims(1) < m) & (chunk_n.expand_dims(0) < n)

    # initialize accumulator
    accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32) # put straight into SRAM upon initialization
            # to understand why we accumulate, see # TODO: put file path to ipad screenshot here

    # iteratively update the accumulator
    for _ in range(0, k, block_size_k): # iterate from 0 to k-1 with jumps of block_size_k
        # load inputs into SRAM
        a = tl.load(offsets_a, mask=mask) # mask prevents loading
        b = tl.load(offsets_b, mask=mask)

        # do the actual operation
        accumulator += tl.dot(a, b)
            # unlike CUDA which works entry-wise, Triton lets us think in terms of vector and matrix-wise operations
            # you'd assume .dot is a vector dot product but remember a & b are of shape (block_size_m, block_size_k)
            #  and (block_size_k, block_size_m) so really it itself is a matmul. don't ask me why Triton is misnaming the op
        
        # increase offets, so next iteration loads next chunks
        offsets_a += block_size_k * stride_ak
        offsets_b += block_size_k * stride_bk
            # notice that we move each offset only along the k dimension
            # so we started in the top-left corner and are now moving offsets_a to the right and offsets_b downward
            # again see # TODO: put file path to ipad screenshot here

    # you can fuse arbitrary activation functions here while the accumulator is still in FP32
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    # because ACTIVATION is determined at compile-time rather than run-time, we dont' actually have to worry about
    #  the O(1) operation of running the if statement every time the kernel is called

    # the accumulation happened in tl.float32 but we save back to DRAM in tl.float16 to save memory
    accumulator = accumulator.to(tl.float16)

    # find the desired location for this chunk of c and store it there
    c_chunk_ptrs = c_ptr + chunk_m.expand_dims(1) * stride_cm + chunk_n.expand_dims(0) * stride_cn

    # store accumulated result back onto relevant part of c in DRAM
    tl.store(c_chunk_ptrs, accumulator, mask=mask)
        # ofc making sure not to write to entries outside the bounds of our dimension so we use the mask

# we can fuse a nonlinearity (here `leaky_relu`) by providing it as an `ACTIVATION` 
#  meta-parameter in `_naive_matmul_kernel`
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def naive_matmul(a, b, block_size=64, activation=""): 
    # you may need to lower block_size if you get an SRAM error. in `matmul.py` we learn how to autotune it
    
    # check constraints
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    assert a.is_contiguous(), "matrix A must be contiguous" # Returns True if tensor is contiguous in memory
        # i think this means that all elements are lined up back-to-back without interruption
        # needs to be true so that our indexing makes sense

    # get dimesion lengths
    (m, k), (_, n) = a.shape, b.shape

    # allocates output
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    
    # 2D launch kernel, meaning we parallelize across both rows and columns (called "row-major ordering")
    grid = lambda meta: (triton.cdiv(m, meta['block_size_m']),  triton.cdiv(n, meta['block_size_n']))
    _naive_matmul_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1), # the jump necessary to go from one element to the next one in that dimension
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        block_size_m=block_size, block_size_n=block_size, block_size_k=block_size,
        ACTIVATION=activation # not used by default since "" is being passed in
    )
    return c

# unit test
torch.manual_seed(0)
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
triton_output = naive_matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

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
        plot_name = "naive_matmul-performance",
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
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        # 2 = number of memory operations (1 read + 1 write)
        # M * N * K = number of elements
        # 1e-12 converts flops to Teraflops
        # ms * 1e-3 converts milliseconds to seconds
    return perf(ms), perf(max_ms), perf(min_ms)
benchmark.run(print_data=False, save_path='.')