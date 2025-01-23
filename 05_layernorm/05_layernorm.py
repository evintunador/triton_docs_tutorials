"""
see original
https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py
"""
import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

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
def _layer_norm_backward_dx_fused(DX_ptr,  # pointer to the input gradient dL/dx where L is loss
                                DY_ptr,  # pointer to the output gradient dL/dy
                                DW_ptr,  # pointer to the partial sum of weights gradient dL/dw
                                DB_ptr,  # pointer to the partial sum of biases gradient dL/db
                                X_ptr,  # pointer to the input
                                W_ptr,  # pointer to the weights
                                Mean_ptr,  # pointer to the mean
                                Rstd_ptr,  # pointer to the 1/std
                                Lock_ptr,  # pointer to the lock
                                stride,  # how much to increase the pointer when moving by 1 row
                                N,  # number of columns in X
                                GROUP_SIZE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    there's a weird grouping strategy being used here for the dw and db that has visuals on the website
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py
    the idea is that each pid is assigned some subset of rows (which are interleaved rather than next to each other)
    and it's that pid's job to accumulate the gradients over all of the rows it has been assigned
    then once each pid is done, in the next kernel we'll accumulate all of those individiual partial sums

    we're using 'd' instead of 'partial' for our derivatives bc it's more concise
    and just saying 'dx' instead of 'dL/dx' for a similar reason
    (as a math major this annoyed me)
    """

    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N # since we're holding an entire row within a single block
    X_ptr += row * stride
    DY_ptr += row * stride
    DX_ptr += row * stride

    # Load data to SRAM
    x = tl.load(X_ptr + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY_ptr + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean_ptr + row) # there's only one mean and std value per row
    rstd = tl.load(Rstd_ptr + row)
    # Compute dx
    xhat = tl.where(mask, (x - mean) * rstd, 0.) # normalized x
    wdy = tl.where(mask, w * dy, 0.) # aka dy/dxhat
    c1 = tl.sum(xhat * wdy, axis=0) / N # c1 and c2 are just intermediary labels; no real meaning
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX_ptr + cols, dx, mask=mask)

    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    # now we'll offset locks and weights/biases gradient pointer for parallel reduction.
    # getting an ID for the block is used in knowing when to let a given pid accumulate
    lock_id = row % GROUP_SIZE # so there are GROUP_SIZE number of locks
    Lock_ptr += lock_id
    # the first GROUP_SIZE entries in Lock hold the state of that lock in the entry Lock_ptr for each pid
    # the next GROUP_SIZE entries hold the count of how many accumulations have already happened on that lock
    Count_ptr = Lock_ptr + GROUP_SIZE
    DW_ptrs = DW_ptr + lock_id * N + cols # we can use N instead of .stride here since these tensors are generated
    DB_ptrs = DB_ptr + lock_id * N + cols #  specifically for this purpose and therefore guaranteed to be contiguous in memory
    # the lock is used to ensure that only one kernel instance writes to the buffer at a time.
    # .atomic_cas() compares the contents of a memory location with a given value and, 
    #  only if they are the same, modifies the contents of that memory location to a new given value.
    # so here, we're looking at the location Lock_ptr and
    # - If it's 0 (unlocked), change it to 1 (locked) and return 0
    # - If it's 1 (already locked), leave it as 1 and return 1
    while tl.atomic_cas(Lock_ptr, 0, 1) == 1:
        pass # this lets us wait until it's unlocked again
    # once it's unlocked we get to move on from the while loop
    # so then here we grab the number of times this lock position has already been accumulated into
    count = tl.load(Count_ptr)
    if count == 0: # if this pid is the first time
        # then no need to accumulate, we can just set equal to
        # .atomic_xchg() sets the value at Count_ptr equal to 1
        tl.atomic_xchg(Count_ptr, 1)
        # so that future pids know that they are not the first
    else: # but if this is not the first pid in the accumulation process,
        # then we've actually gotta accumulate by grabbing the values already there in 
        #  DRAM and adding them to our parial_dw which is in SRAM
        partial_dw += tl.load(DW_ptrs, mask=mask) # we load and add in one step (+= operator)
        partial_db += tl.load(DB_ptrs, mask=mask) #  so as not to consume unnecessary SRAM
    # now we get to store our accumulated values back to DRAM
    tl.store(DW_ptrs, partial_dw, mask=mask)
    tl.store(DB_ptrs, partial_db, mask=mask)
    # and finally release the lock so that any pids waiting in their while loop can take their turn
    tl.atomic_xchg(Lock_ptr, 0) # so we set the value equal to 0
    # whichever pid gets to the 0 value first with its .atomic_cas() will get to go next


@triton.jit
def _layer_norm_backward_dwdb(PARTIAL_DW_ptr,  # pointer to the partial sum of weights gradient
                         PARTIAL_DB_ptr,  # pointer to the partial sum of biases gradient
                         FINAL_DW_ptr,  # pointer to the weights gradient
                         FINAL_DB_ptr,  # pointer to the biases gradient
                         M,  # GROUP_SIZE
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    col_ptrs = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # here is where we'll accumulate the stored group values into as we read them
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        row_ptrs = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_ptrs.expand_dims(1) < M) & (col_ptrs.expand_dims(0) < N)
        offsets = row_ptrs.expand_dims(1) * N + col_ptrs.expand_dims(0)
        # load the partial sums from all that group locking nonsense earlier and add them to our final output
        dw += tl.load(PARTIAL_DW_ptr + offsets, mask=mask, other=0.) # default mask values to 0 so they don't affect sum
        db += tl.load(PARTIAL_DB_ptr + offsets, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW_ptr + col_ptrs, sum_dw, mask=col_ptrs < N)
    tl.store(FINAL_DB_ptr + col_ptrs, sum_db, mask=col_ptrs < N)






class LayerNorm(torch.autograd.Function): 
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(
        ctx, # ctx is an object we use to store info that'll be used later in the backward pass
            # it doesn't actually get inputted to .forward(), rather it's handled by parent torch.autograd.Function
        x, # the input, duh
        normalized_shape, # this never gets used, but putting it here keeps arguments consistent with pytorch which does use it
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
            # in order to support feature_dim bigger than this we'd prolly have to parallelize within feature_dim given common SRAM sizes
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
    def backward(
        ctx, # object holding tensors defined in the forward pass that we saved for use here
            # when calling .backward() we don't actually input ctx; rather it is handled by torch.autograd.Function
        dy # partial derivative of the loss with respect to y, the output of this function's forward method
    ):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # fetcing the original input, weights, biases, means, and reciprocal standard deviations
        x, w, b, mean, rstd = ctx.saved_tensors
        # and our tensor shape
        M, N = x.reshape(-1, x.shape[-1]).shape
            # we only care about the final dimension since parallelizing across the others involves treating them all the same
        # heuristics for amount of parallel reduction stream for DW/DB
        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE, dtype=torch.int32, device=w.device)
            # the first GROUP_SIZE entries in our locks tensor will be used to determine whether a lock is on or off
            # the second will keep track of whether the lock has been used before, since we treat the first use differently
        _dw = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device) # the '_d' versions are intermediaries used for accumulation
        _db = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device) # the non-'_' versions are final gradients
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel that uses forward pass heuristics to calculate both dx and the partial sums for dw and db
        _layer_norm_backward_dx_fused[(M, )](  # parallelize across rows
            dx, dy, _dw, _db, x, w, mean, rstd, locks,  # all of our tensors that'll get turned into pointers
            x.stride(0), N,  # our integer values (determined at run-time)
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  # our meta-parameters (determined at compile-time)
            GROUP_SIZE=GROUP_SIZE, 
            num_warps=ctx.num_warps) # triton handles implementation of num_warps for us so no need to think about it
        # now for a seperate call to the other kernel who's job is to accumulate the partial sums for dw and db.
        # we do this in a separate kernel since this final set of operations requires fewer pids 
        #  (potentially as few as 1 if BLOCK_SIZE_N > N), as opposed to the previous kernel which called M pids
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])] # parallelize within rows
        _layer_norm_backward_dwdb[grid](
            _dw, _db, dw, db, # intermediary and final tensors
            min(GROUP_SIZE, M), N,  # run-time integer values
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, # heuristically chosen compile-time values
            num_ctas=1) # again pls ignore num_ctas; it defaults to 1 so there's no real reason to define it here
        # it's not necessary for our uses since we write to tensors in-place, but it's a PyTorch convention to always return these.
        return dx, None, dw, db, None 
            # the None values correspond to the inputs of the forward pass that don't need gradients (order matters!)


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
        # setting this here instead of x's initial definition means the graph doesn't have to move through the -2.3 and 0.5 operations
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True) # doesn't write to anything because it writed directly to .grad of inputs
        # retain_graph is used to control whether the computation graph should be kept in memory after the backward pass. 
        # Setting retain_graph=True allows you to perform multiple backward passes on the same graph, but it can increase memory usage, 
        #  so it's generally recommended to use it only when necessary
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None # equivalent to setting gradients to 0 after each SGD step
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
        x_vals=[512 * i for i in range(2, 32)], # if you increase past 32 performance will tank since features are larger than 64kb
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'}, # so we're actually only benchmarking the backward pass
    ))
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)#, requires_grad=True)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True) 
        # setting this here instead of x's initial definition means the graph doesn't have to move through the -2.3 and 0.5 operations
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton":
            return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704
        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

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
bench_layer_norm.run(print_data=False, save_path='.')