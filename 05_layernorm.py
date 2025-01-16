"""
see original
https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py
"""
import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

try: # TODO remove all references to APEX
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

@triton.jit
def _layer_norm_fwd_fused(
    x_ptr, # pointer to first entry of the input
    y_ptr, # pointer to first entry of the output
    w_ptr, # pointer to first entry of the weights
    b_ptr, # pointer to first entry of the biases
    mean_ptr, # pointer to first entry of the mean
    rstd_ptr, # pointer to first entry of the reciprocal of the standard deviation
    stride_row, # how much to increase the X pointer when moving through memory to the next row along x
    N, # number of columns in x, aka the tensor's embedding dimension
    eps, # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    x_ptr += row * stride_row
    y_ptr += row * stride_row
    # Compute mean
    sum_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x_ptrs = tl.load(x_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        sum_accumulator += x_ptrs
    mean = tl.sum(sum_accumulator, axis=0) / N
    # Compute variance
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x_ptrs = tl.load(x_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
            # we need to mask a first time for the loading onto SRAM
        diff = tl.where(cols < N, x_ptrs - mean, 0.)
            # and then mask here as well to prevent the out-of-bounds operations from happening
        acc += diff * diff
            # no need to mask operations here since 0.*0.=0.
    var = tl.sum(acc, axis=0) / N
    # calculate reciprocal standard deviation
    rstd = 1 / tl.sqrt(var + eps) # eps is a small number (eg 1e-6) there to prevent division by 0
    # Write mean and rstd to the correct entry in the pre-allocated mean and rstd
    tl.store(mean_ptr + row, mean) # we keep them to use for the backward pass later
    tl.store(rstd_ptr + row, rstd)
    # Normalize and apply linear transformation
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w_ptrs = tl.load(w_ptr + cols, mask=mask)
        b_ptrs = tl.load(b_ptr + cols, mask=mask)
        x_ptrs = tl.load(x_ptr + cols, mask=mask)
        x_hat = (x_ptrs - mean) * rstd
        y = x_hat * w_ptrs + b_ptrs
        # Write output
        tl.store(y_ptr + cols, y, mask=mask)

@triton.jit
def _layer_norm_backward_dx_fused(DX,  # pointer to the input gradient
                                DY,  # pointer to the output gradient
                                DW,  # pointer to the partial sum of weights gradient
                                DB,  # pointer to the partial sum of biases gradient
                                X,  # pointer to the input
                                W,  # pointer to the weights
                                Mean,  # pointer to the mean
                                Rstd,  # pointer to the 1/std
                                Lock,  # pointer to the lock
                                stride,  # how much to increase the pointer when moving by 1 row
                                N,  # number of columns in X
                                GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    there's a weird grouping strategy being used here for the dw and db that has visuals on the website
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py
    the idea is that each pid is assigned some subset of rows (which are interleaved rather than next to each other)
    and it's that pid's job to accumulate the gradients over all of the rows it has been assigned
    then once each pid is done, in the next kernel we'll accumulate all of those individiual partial sums
    """
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M # TODO i think my triple quotes explanation was wrong; different PIDs contribute to same group thanks to lock somehow
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = tl.where(mask, (x - mean) * rstd, 0.)
    wdy = tl.where(mask, w * dy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1: # TODO: figure out what this does
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_backward_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)






class LayerNorm(torch.autograd.Function): 
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(
        ctx, # TODO when is ctx defined? how do we define it? or is it implied?
        x, 
        normalized_shape, # TODO wouldn't this be the same as the input shape?
        weight, # so LayerNorm is in fact a function rather than a module 
        bias, # because it's requiring us to pass in weight & bias parameters instead of stroing them
        eps # very small value (eg 1e-6) to prevent division by zero in the std calculation
    ):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # reshape to 2D tensor and grab said shapes
        M, N = x.reshape(-1, x.shape[-1]).shape
            # because we use stride to move through memory we only need N as opposed to the dimensions that make it up
        # allocate intermediary tensors and final output
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        y = torch.empty_like(x)
        # if there's less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size() # .element_size() returns number of bytes per a single entry
            # so this is used to calculate how many elements can fit within a 64KB block of memory
            # fp32 element_size = 4, fp16 element_size = 2, fp8 element_size = 1
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            # either define block_size by 
            # - the maximum amount of entries that a 64kb block of memory can hold or
            # - the smallest size that can hold the dimension N
        if N > BLOCK_SIZE: # so if we used MAX_FUSED_SIZE
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
            # aka this kernel will not support parallelizing within the embedding dimension N
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M, )](  # grid parallelizes across all non-embedding dimensions (which were flattened into M earlier)
            x, y, # input and pre-allocated output
            weight, bias, # pre-determined parameters
            mean, rstd,  # pre-allocated intermediary useful tensors
            x.stride(0), # number of memory items needed to move forward to hit the next row of x (should be = N, no?)
            N, # model embedding dimension; will be used for hardware mask
            eps,  # small number to prevent division by 0 in std calculation
            # meta-paramaters
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, 
            num_ctas=1 # number of blocks in a block cluster; newer GPU architectures can use it, but older should set to 1
                # honestly we don't need to put it here since it defaults to 1
            ) 
        # now we save a bunch of intermediary tensors and values that'll be useful for the backward pass later
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        # save_for_backward is for tensors, whereas meta-parameters get saved as individual entries in the object
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        # and finally return our output
        return y

    @staticmethod
    def backward(ctx, dy):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # fetcing the original input, weights, biases, means, and reciprocal standard deviations
        x, w, b, mean, rstd = ctx.saved_tensors 
        # heuristics for amount of parallel reduction stream for DW/DB
        M, N = x.reshape(-1, x.shape[-1]).shape
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        _layer_norm_backward_dx_fused[(M, )](  # parallelize across rows
            dx, dy, _dw, _db, x, w, mean, rstd, locks,  #
            x.stride(0), N,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps)
        # now is a seperate call to the other kernel; we do this instead of putting all the same operations into one kernel in order
        # to prevent out-of-sync SMs from messing with each other across these operations since the ops are sequentially dependent
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])] # parallelize within rows
        # this kernel will accumulate the partial sums
        _layer_norm_backward_dwdb[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, 
            num_ctas=1)
        return dx, None, dw, db, None # TODO why would this return None's? something to do with what pytroch's backward expects?


# just creates a reference to the apply function of LayerNorm
layer_norm = LayerNorm.apply 


def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
            return apex_layer_norm(x)  # noqa: F811, E704

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


test_layer_norm(1151, 8192, torch.float16)
bench_layer_norm.run(print_data=True, save_path='./benchmark_results/')