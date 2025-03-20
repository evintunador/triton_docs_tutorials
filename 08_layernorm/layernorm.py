"""
In this lesson on LayerNorm we'll finally connect our kernels to PyTorch's backpropogation graph. Keep in mind
this kernel is fast but it only works for normalizing vectors that fit within SRAM, so we've done a trade-off
of better speed for worse generalize-abililty

What you'll learn:
- Writing a backward pass kernel
- re-using intermediate values from a forward pass in the backward pass
- Using torch.nn.functional to connect to PyTorch's backpropogation graph
- Locks and atomic operations
- How to use sequential kernels with intermediate tensors to complete a calculation 
    more efficiently than one kernel alone could

Recommended order to read the code in:
Step 1 - unit test
Step 2 - wrapper
Step 3 - forward pass kernel
Step 4 - backward pass kernels
Step 5 - benchmark

see original
https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py
"""
import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

######### Step 3 #########
@triton.jit
def _layernorm_forward(
    x_ptr, y_ptr, # points to first entry of tensors of shape (M, N)
    w_ptr, b_ptr, # points to first entry of tensors of shape (N)
    mean_ptr, rstd_ptr, # points to first entry of tensors of shape (M)
    stride_M, # how much to increase the X pointer when moving through memory to the next row along x
    N, # number of columns in x, aka the tensor's embedding dimension
    eps, # small value used to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # use the program id to move x_ptr and y_ptr to the row of X and Y they should compute
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    # Compute mean
    sum_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
            # we're assuming in this over-simplified example that x is contiguous along N dimension, 
            #  so no need to multiply cols by a stride (which should just be equal to 1)
        x_ptrs = tl.load(x_ptr + cols, mask=cols < N, other=0.).to(tl.float32) # shape (BLOCK_SIZE)
            # x is fp16 but we want to accumulate in fp32 for increased accuracy
            # other=0.0 since zeros don't affect summation
        sum_accumulator += x_ptrs
    mean = tl.sum(sum_accumulator, axis=0) / N
        # shape goes from (BLOCK_SIZE) to (1)

    # Compute variance & reciprocal standard deviation
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x_ptrs = tl.load(x_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        diff = tl.where(cols < N, x_ptrs - mean, 0.)
            # mask here to prevent (0.0 - mean) at the masked out-of-bounds values
        acc += diff * diff
            # no need to mask operations here since 0 * 0 = 0 
    var = tl.sum(acc, axis=0) / N # shape goes from (BLOCK_SIZE) to (1)
    rstd = 1 / tl.sqrt(var + eps) 
        # eps is a small number (eg 1e-6) there to prevent division by 0

    # we save mean and rstd for the backward pass later
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Normalize and apply linear transformation
    for offset in range(0, N, BLOCK_SIZE):
        # load input and parameters
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w_ptrs = tl.load(w_ptr + cols, mask=mask)
        b_ptrs = tl.load(b_ptr + cols, mask=mask)
        x_ptrs = tl.load(x_ptr + cols, mask=mask)

        # Normalize and apply linear transformation
        x_hat = (x_ptrs - mean) * rstd
        y = x_hat * w_ptrs + b_ptrs

        # Write output
        tl.store(y_ptr + cols, y, mask=mask)


######### Step 4 #########
@triton.jit
def _layernorm_backward_dLdx(
    x_ptr, dLdx_ptr, dLdy_ptr,                              # pointers to first entries of tensors of shape (M, N)
    w_ptr,                                                  # pointers to first entries of tensors of shape (N)
    dLdw_intermediate_ptr, dLdb_intermediate_ptr,           # pointers to first entries of tensors of shape (GROUP_SIZE, N)
    mean_ptr, rstd_ptr,                                     # pointers to first entries of tensors of shape (M)
    locks_ptr,                                              # pointers to first entry of tensor of shape (2 * GROUP_SIZE)
    stride, N,                                              # dynamic variables determined at run-time
    GROUP_SIZE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr    # static variables determined at compile-time
):
    """
    there's a weird grouping strategy being used here for the _dLdw and _dLdb that has visuals on the website
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py
    the idea is that each pid is assigned some subset of rows (which are interleaved rather than next to each other)
    and it's that pid's job to accumulate the gradients over all of the rows it has been assigned
    then once each pid is done, in the next kernel we'll accumulate all of those individiual partial sums
    """
    # Map the program id to the elements of x, dLdx, and dLdy it should compute.
    PID = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N # since we're holding an entire row within a single block
    x_ptr += PID * stride
    dLdx_ptr += PID * stride
    dLdy_ptr += PID * stride

    # Load data to SRAM
    # it's generally faster to do a bunch of loads before a bunch of flops rather than alternating back & forth
    x = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)            # shape (BLOCK_SIZE_N)
    dLdy = tl.load(dLdy_ptr + cols, mask=mask, other=0).to(tl.float32)      # shape (BLOCK_SIZE_N)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)                     # shape (BLOCK_SIZE_N)
    mean = tl.load(mean_ptr + PID)                                          # shape (1)
    rstd = tl.load(rstd_ptr + PID)                                          # shape (1)

    # Compute dLdx
    x_normalized = tl.where(mask, (x - mean) * rstd, 0.)        # shape (BLOCK_SIZE_N)
    dydx_normed = tl.where(mask, w * dLdy, 0.)                  # shape (BLOCK_SIZE_N)
    # c1 and c2 are just intermediary labels; the names don't have any real meaning
    c1 = tl.sum(x_normalized * dydx_normed, axis=0) / N         # shape (1)
    c2 = tl.sum(dydx_normed, axis=0) / N                        # shape (1)
    dLdx = (dydx_normed - (x_normalized * c1 + c2)) * rstd      # shape (BLOCK_SIZE_N)

    # Write dLdx back to DRAM
    tl.store(dLdx_ptr + cols, dLdx, mask=mask)

    # Here we'll accumulate partial sums for dLdw and dLdb, meaning these are only the single rows of 
    #  the dLdw and dLdb gradients that this PID had the job of calculating
    dLdw_contribution = (dLdy * x_normalized).to(w.dtype)
    dLdb_contribution = (dLdy).to(w.dtype)

    """
    Now we'd like to take our single contributions to dLdw and dLdb and somehow aggregate them with
    the portions that all of the other PIDs have calculated.
    The reason this aggregation has to happen is because the input x is of shape (M, N) while
    the weights and biases are of shape (N), meaning they receive gradients from all M rows of x,
    and this PID holds the gradient of one of those rows, but it's not easy to communicate
    that information between PIDs.
    The specific operation to do between all these rows is to sum them up, but we can't just
    naively tl.load(), then add our row, then tl.store() because all of the PIDs would do so
    at slightly different and completely unpredictable times, meaning all the tl.store() calls
    would overwrite each other.
    What we need first a way to ensure that only one PID at a time does the read, flop, and 
    write while all the other PIDs others wait their turn.
    For this we can use what's called a lock, which is a way for us to ensure that only one 
    PID can work on a given part of a tensor in DRAM at a time, AKA "locking" it.

    However, even that's not great because if only one PID can do work at a time and we have a 
    lot of PIDs, then that's a whole lot of time leaving a large majority of the GPU sitting 
    idle while they wait in line. 
    What we need then is a way for GROUPS of PIDs to work sequentially with a lock while each
    group works in parallel to the others. 
    This is why we created dLdw_intermediate and dLdb_intermediate, each of which has shape 
    (GROUP_SIZE, N).
    We're going to assign every PID to a group, and then use our locking mechanism to ensure
    that only (M // GROUP_SIZE) PIDs attempt to wait around for their turn to work on a row
    of dLdw_intermediate and dLdb_intermediate at a time.
    In this way we've now gone from a sequential process with M steps to one with 
    (M // GROUP_SIZE) steps. 
    Then in the next kernel we'll take these (GROUP_SIZE, N) matrices and reduce them further
    down to the desired shape (N) matrices of dLdw and dLdb.

    But how do locks actually work?
    In this case we've got a tensor of shape (2 * GROUP_SIZE) and datatype int32 that's
    initialized to all zeroes. 
    The first GROUP_SIZE entries are for holding an indicator of the state of that lock;
    0 means unlocked and 1 means locked for the row of dLdw_intermediate and dLdb_intermediate 
    that it corresponds to.
    The latter GROUP_SIZE entries are for holding an indicator of whether this lock has
    ever been used before, which is useful because we'll want to run different code if
    this PID happens to be the first one to add its values to dLdw_intermediate and dLdb_intermediate.
    To use the lock, we check if the entry corresponding to the group that our PID is
    in is locked or unlocked:
    - if it's locked, then we wait and check again in a moment until it's unlocked
    - if it's unlocked then we'll lock it, load the current value of our group's row of 
        dLdw_intermediate and dLdb_intermediate, add our dLdw_contribution and dLdb_contribution 
        respectively, write those new values back to DRAM, and finally unlock it
    """
    # To start we figure out which lock ID corresponds to our PID and move our pointers accordingly
    lock_id = PID % GROUP_SIZE # so there are GROUP_SIZE number of locks
    # the first GROUP_SIZE entries in Lock hold the state of that lock in the entry locks_ptr for each pid
    locks_ptr += lock_id
    # the next GROUP_SIZE entries hold the count of how many accumulations have already happened on that lock
    count_ptr = locks_ptr + GROUP_SIZE
    # then we figre out which row of dLdw_intermediate and dLdb_intermediate we're meant to point to
    dLdw_intermediate_ptrs = dLdw_intermediate_ptr + lock_id * N + cols 
    dLdb_intermediate_ptrs = dLdb_intermediate_ptr + lock_id * N + cols 
        # we can use N in place of a .stride() here since these tensors are generated specifically for 
        #  this purpose and therefore guaranteed to be contiguous in memory
    
    # atomic_cas() compares the contents of a memory location with a given value and, 
    #  only if they are the same, modifies the contents of that memory location to a new given value.
    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass
        # so here, we're looking at the location locks_ptr_ptr and:
        # - If it's 0 (unlocked), change it to 1 (locked) and return 0 (False) to exit the while loop
        # - If it's 1 (already locked), leave it as 1 and return 1 (True) so that we stay in the while loop
    
    # then here we grab the number of times this lock position has already been accumulated into
    count = tl.load(count_ptr) # shape (1)
    if count == 0: # if this PID is the first one to access the lock
        # then no need to do the memory reads & flops; we can just set the row of dLdw_intermediate & 
        #  dLdB_intermediate equal to dLdw_contribution and dLdb_contribution (done below, outside the if/else)
        # atomic_xchg() sets the value at Count_ptr equal to 1 so the next PID knows we've been here
        tl.atomic_xchg(count_ptr, 1)
    else: # but if this is not the first pid in the accumulation process,
        # then we've actually gotta accumulate by grabbing the values already there in 
        #  DRAM and adding them to the rows of dLdw_contribution and dLdb_contribution that our PID generated
        dLdw_contribution += tl.load(dLdw_intermediate_ptrs, mask=mask) # we load and add in one step (+= operator)
        dLdb_contribution += tl.load(dLdb_intermediate_ptrs, mask=mask) #  so as not to consume unnecessary SRAM
    
    # now we get to store our accumulated values back to DRAM
    tl.store(dLdw_intermediate_ptrs, dLdw_contribution, mask=mask)
    tl.store(dLdb_intermediate_ptrs, dLdb_contribution, mask=mask)

    # and finally release the lock so that any pids waiting in their while loop can take their turn
    tl.atomic_xchg(locks_ptr, 0) # we set the value at our lock equal to 0
    # whichever pid gets to the 0 value first with its .atomic_cas() will get to go next

@triton.jit
def _layernorm_backward_dLdw_dLdb(
    dLdw_intermediate_ptr,  dLdb_intermediate_ptr, # pointers to first entries of tensors of shape (GROUP_SIZE, N)
    dLdw_ptr, dLdb_ptr,                            # pointers to first entries of tensors of shape (N)
    GROUP_SIZE,  N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # our PIDs are split up within the N dimension
    PID = tl.program_id(0)
    col_ptrs = PID * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # here is where we'll accumulate the stored group values into as we read them
    dLdw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate through the rows of _dLdw and _dLdb to sum them up
    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_ptrs = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_ptrs[:, None] < GROUP_SIZE) & (col_ptrs[None, :] < N)
        offsets = row_ptrs[:, None] * N + col_ptrs[None, :]

        # load the partial sums from all that group locking nonsense earlier and add them to our final output
        dLdw_acc += tl.load(dLdw_intermediate_ptr + offsets, mask=mask, other=0.) 
        dLdb_acc += tl.load(dLdb_intermediate_ptr + offsets, mask=mask, other=0.)
            # masked-out values get set to 0 so they don't affect sum

    # sum along our BLOCK_SIZE_M dimension to get the final BLOCK_SIZE_N chunk of dLdw & dLdb that this 
    #  PID was assigned to
    sum_dLdw = tl.sum(dLdw_acc, axis=0) # shape (BLOCK_SIZE_N)
    sum_dLdb = tl.sum(dLdb_acc, axis=0)

    # Write the final sum to the output.
    tl.store(dLdw_ptr + col_ptrs, sum_dLdw, mask=col_ptrs < N)
    tl.store(dLdb_ptr + col_ptrs, sum_dLdb, mask=col_ptrs < N)


######### Step 2 #########
class LayerNorm(torch.autograd.Function): 
    """
    We can implement our own custom functions that play nice with PyTorch's autograd graph
    by subclassing torch.autograd.Function and implementing the forward and backward passes
    with static methods forward() and backward(). 
    """

    @staticmethod
    def forward(
        ctx, # ctx is an object we use to store info that'll be used later in the backward pass
            # it doesn't actually get inputted when using .forward(), rather it's handled by the parent class
        x, # the input; however many dimensions will be turned into a matrix of shape (M, N)
        normalized_shape, # this never gets used, but putting it here keeps arguments consistent with pytorch which does use it
        weight, # so this LayerNorm class is in fact acting as a function rather than a module since w&b are stored elsewhere
        bias, # weight and bias both of shape (x.shape[-1])
        eps # very small value (eg 1e-6) to prevent division by zero in the reciprocal standard deviation calculation
    ):
        # reshape to 2D tensor and grab said shapes
        M, N = x.reshape(-1, x.shape[-1]).shape
        # allocate intermediary tensors and final output
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        y = torch.empty_like(x)

        # if there's less than 64KB per feature then we can use our fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size() 
            # .element_size() returns number of bytes per a single entry
                # fp32 element_size = 4, fp16 element_size = 2, fp8 element_size = 1
            # so this is used to calculate how many elements can fit within a 64KB block of memory
            # 64KB is a heuristic for the smallest possible SRAM size our GPU is likely to have; it'd be beter
            #  if we got our GPU's actual SRAM size and used that (look back at lesson 5 for how to do this)
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            # we'll either define block_size by 
            # - the maximum amount of entries that a 64kb block of memory can hold or
            # - the smallest size that can hold the dimension N
        if N > BLOCK_SIZE: # so if we used MAX_FUSED_SIZE
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
            # in order to support feature_dim bigger than SRAM size we'd have to parallelize within feature_dim

        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        
        _layernorm_forward[(M, )](  # grid parallelizes using a separate program for each non-embedding dimension entry
            x, y, weight, bias,
            mean, rstd,  # pre-allocated intermediary useful tensors
            x.stride(0), # number of memory items needed to move forward to hit the next row of x (should be = N if x is contiguous)
            N, # model embedding dimension will be used for hardware mask
            eps,  # small number to prevent division by 0 in reciprocal standard deviation calculation
            # meta-paramaters
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, 
        ) 

        # ctx is an object that can be used to stash information that's useful for the backward pass computation
        # You can cache arbitrary objects using the ctx.save_for_backward method
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        # save_for_backward is mostly for tensors, whereas meta-parameters get saved as individual entries in the object
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        # and finally return our output
        return y

    @staticmethod
    def backward(
        ctx, # when calling .backward() we don't actually input ctx; rather it is handled by torch.autograd.Function
        dLdy # partial derivative of the loss with respect to y
    ):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss with respect to the output, and 
        we need to compute the gradient of the loss with respect to the input(s).
        """
        # fetcing the original inputs, intermediary tensors, and meta-parameters
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape

        # allocate gradients of original inputs
        dLdw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dLdb = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dLdx = torch.empty_like(dLdy)

        # heuristics for amount of parallel reduction stream for dLdw & dLdB; explained a bit below but mostly in the kernel
        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256

        # Rather than computing all three gradients immediately in one kernel, we're actually going to call two kernels.
        # The first will compute dLdx and intermediary steps on the way to dLdw and dLdb; we'll call these _dLdw and _dLdb
        dLdw_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)
        dLdb_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)

        # When multiple programs want to edit the same entries in a tensor stored in DRAM, we need a way to prevent them from
        #  doing so out of order and from overwriting each other's work. For that we can use a lock, which is another tensor 
        #  with the job of keeping track of which entries are currently being worked on by a different program and which are
        #  free to be edited
        locks = torch.zeros(2 * GROUP_SIZE, dtype=torch.int32, device=w.device)
            # the first GROUP_SIZE entries in our locks tensor will be used to determine whether a lock is on or off
                # (AKA whether the important tensor is occupied or available)
            # the second will keep track of whether the lock has been used before, since in the kernel we will need to 
            #  treat the first use differently from all successive uses
        
        # enqueue kernel that uses forward pass heuristics to calculate both dLdx and the partial contributions to dLdw and dLdb
        _layernorm_backward_dLdx[(M, )](  # parallelize across rows
            x, dLdx, dLdy, 
            w, dLdw_intermediate, dLdb_intermediate, 
            mean, rstd, 
            locks,  
            x.stride(0), N,  # dynamic run-time variables
            GROUP_SIZE = GROUP_SIZE, BLOCK_SIZE_N = ctx.BLOCK_SIZE, num_warps = ctx.num_warps) # static compile-time variables
        
        # Now we'll do a seperate call to the second kernel, who's job is to accumulate dLdw_intermediate into dLdw and 
        #  dLdb_intermediate into dLdb.
        # We do this in a separate kernel since this final set of operations requires 
        #  1) fewer pids as opposed to the previous kernel which called M pids
        #  and 2) dLdw_intermediate and dLdb_intermediate to be completed before it can begin
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])] # parallelize within rows
        _layernorm_backward_dLdw_dLdb[grid](
            dLdw_intermediate, dLdb_intermediate, dLdw, dLdb, 
            min(GROUP_SIZE, M), N,  # run-time integer values
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, # heuristically chosen compile-time values
        )
        
        # pytorch expects .backward() to return a value for every single input into .forward() in order (except for ctx)
        #  so that it can keep track for the backpropogation graph
        return dLdx, None, dLdw, dLdb, None 
            # the None values correspond to the inputs of .forward() that don't need gradients (order matters!)

# this line just creates a reference to the apply function of LayerNorm rather than having it act like an object
layernorm = LayerNorm.apply 


######### Step 1 #########
def test_layernorm_kernel(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    weight = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    dLdy = .1 * torch.randn_like(x)
    # setting requires_grad to True here instead of x's initial definition means the graph doesn't have to move through 
    #  the -2.3 and 0.5 operations. That's not a big deal here for testing but if we didn't do it in the benchmark then
    #  those results would be confounded by the kernels pytorch implements for entry-wise multiplication and addition
    x.requires_grad_(True)
    # forward pass
    y_tri = layernorm(x, (N,), weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps).to(dtype)
    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0) 
    print("Passed fwd")
    # backward pass (triton)
    y_tri.backward(dLdy, retain_graph=True) # this writes directly to x.grad, weight.grad and bias.grad
        # retain_graph is used to control whether the computation graph should be kept in memory after the backward pass. 
        # Setting retain_graph=True allows you to perform multiple backward passes on the same graph, but it can increase 
        # memory usage, so it's generally recommended to use it only when necessary for a scenario like this
    # This detaches our gradients so that we can run pytorch on the same input tensors and test against each other later
    dLdx_tri, dLdw_tri, dLdb_tri = [_.grad.clone() for _ in [x, weight, bias]]
        # when denoting derivatives, it's always with respect to the loss function L and we use "d" instead of "partial"
        #  because it's more concise albiet bad practice from a mathematician's perspective
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dLdy, retain_graph=True)
    dLdx_ref, dLdw_ref, dLdb_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # compare
    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)
        # rtol=0 means we don't use relative tolerance 
    print("Passed bwd")


######### Step 5 #########
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)], # if you increase past 32 the kernel will break since features become larger than 64kb
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'}, # so we're actually only benchmarking the backward pass
    ))
def benchmark(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (N, )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)#, requires_grad=True)
    dLdy = .1 * torch.randn_like(x)
    x.requires_grad_(True) 
        # setting this here instead of x's initial definition means the graph doesn't have to move through the -2.3 and 0.5 operations
    quantiles = [0.5, 0.05, 0.95]

    def y_fwd():
        if provider == "triton":
            return layernorm(x, w_shape, weight, bias, eps) 
        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps) 

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dLdy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # always run unit-tests
    test_layernorm_kernel(1151, 8192, torch.float16)

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)