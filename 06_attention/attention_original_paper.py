"""
this is an implementation of the flash-attention pseudocode from the original paper
https://arxiv.org/pdf/2307.08691
"""

import torch

import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptrs,
    V_block_ptrs,
    block_index_q,
    softmax_scale,
    #stride_K_seq, # if you go for a manual pointer implementation then you'll need to pass in these strides.
    #stride_V_seq, # in the automatic implementation relevant stride lengths are saved in the pointer object
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    CAUSAL: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_q: tl.constexpr,
    offsets_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if CAUSAL and DIAGONAL:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    elif CAUSAL: # any blocks in the causal mask below the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    else: # runs on every single block for the case that we're not using a causal mask
        lo, hi = 0, SEQ_LEN

    K_block_ptrs = tl.advance(K_block_ptrs, (0, lo))
    V_block_ptrs = tl.advance(V_block_ptrs, (lo, 0))
    """
    Here are the above ^ two lines implemented with manual pointers.
    Remember you can't mix automatic & manual pointer implementations; choose one & stick to it
    K_block_ptrs += lo * stride_K_seq
    V_block_ptrs += lo * stride_V_seq
    """

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
            # TODO what exactly is this optimizing? need to know that in order to know when to use it

        # compute (Q @ K^T) / sqrt{head_dim}
        K_block = tl.load(K_block_ptrs)
        QK_block = tl.dot(Q_block, K_block) * softmax_scale # becomes shape (BLOCK_SIZE_Q, BLOCK_SIZE_KV)

        if CAUSAL and DIAGONAL: # if causal mask and we're currently on a block containing the diagonal
            mask = offsets_q.expand_dims(1) >= (start_kv + offsets_kv.expand_dims(0))
            QK_block += tl.where(mask, 0, -1.0e6) # TODO is -1.e6 the most efficient choice for -inf in Triton?
        
        # find the max values of the new block and compare them to those of all previous blocks to get an update
        m_ij = tl.maximum(m_i, tl.max(QK_block, 1)) # 1 is the axis to find the max along, so shape is (BLOCK_SIZE_Q)
        # adjust QK block for safe softmax
        QK_block -= m_ij.expand_dims(1)

        # Compute the exponential of each safe dot product, which is the numerator of our softmax
        P_block = tl.math.exp(QK_block)
        # P_block = tl.math.exp2(QK_block) 
            # TODO try out .exp2(), should be faster & softmax should be invariant to it

        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1) # 1 is the axis to compute sum along, so we get shape (BLOCK_SIZE_Q)
        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij) # shape (BLOCK_SIZE_Q)
            # TODO try out .exp2(), should be faster & softmax should be invariant to it
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij
        
        # This computes O_new = P x V + O_old * alpha
        V_block = tl.load(V_block_ptrs) # shape (BLOCK_SIZE_KV, HEAD_DIM)
        P_block = P_block.to(tl.float16) # TODO not sure why this op needs to be in fp16; when i tried fp32 it broke
        O_block = O_block * alpha.expand_dims(1) # adjusts previous values based on potential new max
        O_block = tl.dot(P_block, V_block, acc=O_block) # accumulated P and V block dot product into O block
            # so we get shape (BLOCK_SIZE_Q, HEAD_DIM)
            # notice we're doing this before we've actually divided by our softmax denominator l_i
            # which is possible because TODO

        m_i = m_ij # sets old max equal to new max, ready to be used by next iteration of for loop

        # Move to the next block of K and V along the SEQ_LEN dimension
        V_block_ptrs = tl.advance(V_block_ptrs, (BLOCK_SIZE_KV, 0))
        K_block_ptrs = tl.advance(K_block_ptrs, (0, BLOCK_SIZE_KV))
        """
        Here are the above ^ two lines implemented with manual pointers.
        Remember you can't mix automatic & manual pointer implementations; choose one & stick to it
        V_block_ptrs += BLOCK_SIZE_KV * stride_V_seq
        K_block_ptrs += BLOCK_SIZE_KV * stride_K_seq
        """

    return O_block, l_i, m_i # we save these three specifically for use later in the backward pass


@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128] # values chosen heuristically
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"], # auto-tune will re-run every time either of these values changes in a new input
)
@triton.jit
def _attn_fwd(
    Q_ptr, K_ptr,  V_ptr,  # each (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    softmax_scale,
    M_ptr,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O_ptr,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim, # dist to move thru mem to next entry in that dim of that tensor
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim,
    BATCH_SIZE, # unlike other tensor dimensions, batch size needs to be flexible for runtime differences
    # meta-parameters (decided at compile-time)
    NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # as opposed to regular assert, static_assert occurs at compile-time
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
        # head_dim is usually relatively small (128 or 256) so it wouldn't make sense to parallelize within it.
        # TODO not sure why this specific criteria is needed tho. are we trying to cap memory consumption
        #  of each component at HEAD_DIM^2?

    # This indicates which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch_head = tl.program_id(1)
    # This indicates which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicates the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS
    # TODO could we have instantiated batch & heads separately in the launch grid instead of this calc? 
    #  would that change anything? my intuition is that doing it this way is better at sharing SRAM.
    # yeah since this grid has us iterate through all heads first and then go to the next batch, and IF we're
    #  using multi-query attention then it'd be useful to have the loading of kv heads happen in the same SRAM
    #  for any given SM. I think this code does to MHA so it doesn't quite apply but u get the point

    # This allows to get the (SEQ_LEN, HEAD_DIM) block in the Q, K, V by indexing it by batch and head
    qkv_offset = index_batch * stride_Q_batch + index_head * stride_Q_head

    # so here's a new function that does the math of finding the right pointer for us
    Q_block_ptrs = tl.make_block_ptr(
        base=Q_ptr + qkv_offset, # base pointer to the parent tensor
        shape=(SEQ_LEN, HEAD_DIM), # shape of the parent tensor
            # notice our parent tensor is actually a single splice of the original Q rather than the full Q
            # meaning this function is pretty flexible since it only has to calculate in terms of strides
        strides=(stride_Q_seq, stride_Q_dim), # strides of the parent tensor
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # offsets to the block
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM), # shape of the block
        order=(1, 0), # TODO still don't know what order does
    )
    """
    # Here is the above ^ function implemented manually.

    # Our base pointer is actually going to be a specific batch and head, meaing we're working with a (SEQ_LEN,HEAD_DIM) matrix.
    Q_ptr += qkv_offset 

    # Offsets for seq_len are split by pids but for head_dim we keep the whole thing in SRAM.
    offsets_Q_seq_len = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offsets_Q_head_dim = tl.arange(0, HEAD_DIM) # TODO should it be triton.next_power_of_2(HEAD_DIM))
    
    # putting it all together we 
        # 1. start at the first entry of the (SEQ_LEN,HEAD_DIM matrix),
        # 2. turn the seq_len and head_dim components into 2D tensors to match
        # 3. adjust for stride length in memory
    Q_block_ptrs = Q_ptr + (offsets_Q_seq_len.expand_dims(1) * stride_Q_seq + offsets_Q_head_dim.expand_dims(0) * stride_Q_dim)
    
    HERE'S THE THING:
    when writing a kernel, you have to choose between whether you're going to use make_block_ptr or do it manually.
    once you choose you can't mix the two because make_block_ptr actually returns a weird triton object that is
    incompatible with your manually created pointers. throughout the rest of this kernel (and the sub-kernel it calls)
    i am going to be adding in the manual version of each relevant line of code as a comment and reminding you to not mix them
    """

    V_block_ptrs = tl.make_block_ptr(
        base=V_ptr + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0), # no seq_len offsets because for V we parallelize across seq_len in for loop in _attn_inner_fwd() TODO confirm
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    """
    Here is the above ^ function implemented manually. 
    Remember you can't mix automatic & manual pointer implementations; choose one & stick to it
    V_ptr += qkv_offset
    offsets_V_seq_len = tl.arange(0, BLOCK_SIZE_KV)
    offsets_V_head_dim = tl.arange(0, HEAD_DIM) # TODO should it be triton.next_power_of_2(HEAD_DIM))
    V_block_ptrs = V_ptr + (offsets_V_seq_len.expand_dims(1) * stride_V_seq + offsets_V_head_dim.expand_dims(0) * stride_V_dim)
    #"""
    
    K_block_ptrs = tl.make_block_ptr(
        base=K_ptr + qkv_offset,
        shape=(HEAD_DIM, SEQ_LEN), # we transpose K in a fused manner here instead of making an entire separate kernel for transpose
        strides=( # by inverting the strides, we are transposing the matrix
            stride_K_dim,
            stride_K_seq,
        ),  
        offsets=(0, 0), # no seq_len offsets because for K we parallelize across seq_len in for loop in _attn_inner_fwd() TODO confirm
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV), # don't forget transpose means this shape is flipped
        order=(0, 1), # TODO ok why is this order different?
    )
    """
    Here is the above ^ function implemented manually. 
    Remember you can't mix automatic & manual pointer mplementations; choose one & stick to it
    K_ptr += qkv_offset
    offsets_K_seq_len = tl.arange(0, BLOCK_SIZE_KV)
    offsets_V_head_dim = tl.arange(0, HEAD_DIM)
    K_block_ptrs = K_ptr + (offsets_K_seq_len.expand_dims(0) * stride_V_seq + offsets_V_head_dim.expand_dims(1) * stride_V_dim)
    #"""

    O_block_ptrs = tl.make_block_ptr( # this should all look the same as Q
        base=O_ptr + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0), # TODO still don't know what order does
    )
    """
    Here is the above ^ function implemented manually. 
    Remember you can't mix automatic & manual pointer implementations; choose one & stick to it
    O_ptr += qkv_offset
    offsets_O_seq_len = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offsets_V_head_dim = tl.arange(0, HEAD_DIM)
    O_block_ptrs = O_ptr + (offsets_O_seq_len.expand_dims(1) * stride_O_seq + offsets_V_head_dim.expand_dims(0) * stride_O_dim)
    #"""

    # these next two were calculated internally by calls of make_block_ptr() but not given to us and still needed by us.
    # the offsets for the tokens in the Q to process
    offsets_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # the offsets for the tokens in the K and V sequence to process
    offsets_kv = tl.arange(0, BLOCK_SIZE_KV)

    # the running maximum. We have one for each query in the block we're currently working on
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf") 
    # the running sum. We have one for each query (since we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0 # TODO why +1? is it bc of weird exponential rules? e^0=1
    
    # the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptrs) # shape (BLOCK_SIZE_Q, HEAD_DIM)

    # calculate attention for dense blocks (those where the mask if full of 1's). This step runs for 
    # the entirety of non-causal attention and for the blocks below the diagonal in causal attention
    O_block, l_i, m_i = _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptrs,
        V_block_ptrs,
        block_index_q,
        softmax_scale,
        #stride_K_seq, # if you go for a manual pointer implementation then you'll need to pass in these strides.
        #stride_V_seq, # in the automatic implementation relevant stride lengths are saved in the pointer object
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        CAUSAL, #
        False, # blocks on the DIAGONAL get special treatment if this is set to true; we use it below
        offsets_q,
        offsets_kv,
        SEQ_LEN,
    )

    if CAUSAL: # This step runs for the blocks on the diagonal in the causal attention mask
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptrs,
            V_block_ptrs,
            block_index_q,
            softmax_scale,
            #stride_K_seq, # if you go for a manual pointer implementation then you'll need to pass in these strides.
            #stride_V_seq, # in the automatic implementation relevant stride lengths are saved in the pointer object
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            CAUSAL,
            True, # blocks on the diagonal get special masking treatment
            offsets_q,
            offsets_kv,
            SEQ_LEN,
        )
    
    # finally dividing by the denominator of our softmax.
    # notice we've already multiplied by V to get O, so this was done out-of-order from naive softmax implementations
    O_block = O_block / l_i.expand_dims(1) # shapes (BLOCK_SIZE_Q, HEAD_DIM) / (BLOCK_SIZE_Q) = (BLOCK_SIZE_Q, HEAD_DIM)
        # we can do this out-of-order since the matmul (the tl.dot in _attn_fwd_inner) and this entry-wise division 
        #  are associative. matmul and entry-wise-ops are not normally, but at this level of granularity  it's no longer
        #  actually a matmul but instead individual dot-products. for an example see # TODO put file path here

    # This is needed to compute the logsumexp (LSE) for the backwards pass. basically instead of saving the maxes 
    #  and the sums separately, we save them together which still works thanks to exponential arithmetic
    m_i += tl.math.log(l_i)  # l_i was composed using the sum & exp operations in _attn_fwd_inner()
        # this will work because softmax(x_i) = exp(x_i - m_i) / l_i 
        #                                     = exp(x_i - m_i) / exp(log(l_i)) 
        #                                     = exp(x_i - m_i - log(l_i))
        # personally I'd prefer to do change the name and do LSE_i = m_i + tl.math.log(l_i) since
        #  that'd be more clear about what we're doing. However, it's inefficient since it'd require
        #  that we allocate an entire new tensor for LSE. Instead we opt to re-use tensor m to store
        #  our logsumexp. it's ugly but it gets the job done # TODO is this correct?
    
    # storing it all back to DRAM
    m_block_ptrs = M_ptr + index_batch_head * SEQ_LEN + offsets_q
    tl.store(m_block_ptrs, m_i)
    tl.store(O_block_ptrs, O_block.to(O_ptr.type.element_ty)) # TODO what's with this type changing stuff?


@triton.jit
def _attn_backward_preprocess(
    O_ptr,
    dO_ptr,
    D_ptr,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """the job of this kernel is to pre-compute D since D is used by both of the following two kernels"""
    block_index_q = tl.program_id(1) # SEQ_LEN / BLOCK_SIZE_Q number of pids
    offsets_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(0) # BATCH_SIZE * NUM_HEADS number of pids
    offsets_dim = tl.arange(0, HEAD_DIM)

    # Load BLOCK_SIZE_Q rows of O
    O_ptr += index_batch_head * HEAD_DIM * SEQ_LEN # moves O_ptr to the correct batch & head for this pid.
        # HEAD_DIM * SEQ_LEN is equal to stride_num_heads. we can use them instead of .stride() since we know dO is contiguous
    O_block_ptrs = O_ptr + offsets_q.expand_dims(1) * HEAD_DIM + offsets_dim.expand_dims(0) # TODO what does multiplying by HEAD_DIM do??
        # so does *HEAD_DIM actually mean our block actually has gaps in-between each row?
    O_block = tl.load(O_block_ptrs) # shape (BLOCK_SIZE_Q, HEAD_DIM)

    # Load BLOCK_SIZE_Q rows of dO
    dO_ptr += index_batch_head * HEAD_DIM * SEQ_LEN
    dO_block_ptrs = dO_ptr + offsets_q.expand_dims(1) * HEAD_DIM + offsets_dim.expand_dims(0) # TODO what does multiplying by HEAD_DIM do??
    dO_block = tl.load(dO_block_ptrs).to(tl.float32) # shape (BLOCK_SIZE_Q, HEAD_DIM) 
        # TODO why does this one get sent to float32 but not O_block? store tensors in 16 but grads in 32?

    # so D is the dot product of O and dO along HEAD_DIM, giving us a single scalar Di per token in SEQ_LEN
    D_block = tl.sum(dO_block * O_block, axis=1)  
        # shape: sum((BLOCK_SIZE_Q, HEAD_DIM) * (BLOCK_SIZE_Q, HEAD_DIM), axis=1) -> sum((BLOCK_SIZE_Q, HEAD_DIM), axis=1) -> (BLOCK_SIZE_Q)
        # TODO the OG paper says this is actually of size (HEAD_DIM) meaning either it has a typo or i'm misinterpreting what axis=1 means
    D_block_ptrs = D_ptr + index_batch_head * SEQ_LEN + offsets_q
    tl.store(D_block_ptrs, D_block) # TODO figure out & explain why D is useful & why it's called D


@triton.jit
def _attn_backward(
    Q_ptr, K_ptr, V_ptr,
    softmax_scale,
    dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    L_ptr, D_ptr,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS, SEQ_LEN,
    BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # selecting which pid we are in respect to BATCH_SIZE and NUM_HEADS
    idx_batch_head = tl.program_id(1)
    idx_batch = idx_batch_head // NUM_HEADS # getting the sequence in the batch this pid is assigned to
    idx_head = idx_batch_head % NUM_HEADS # getting the head this pid is assigned to

    # This offset allows us to select the right (SEQ_LEN, HEAD_DIM) matrix from a tensor of shape 
    #  (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM) given the batch and head
    offset_batch_head = (idx_batch * stride_batch + idx_head * stride_head).to(tl.int64)
        # TODO i presume int64 is bc BATCH_SIZE * NUM_HEADS could legitimately exhaust int32's representation range?
    # The reason we don't access the blocks through the automated method make_block_ptr is because we need to use the range of 
    #  offsets to apply the masking, meaning there are times when building pointers manually provides more control
    Q_ptr += offset_batch_head
    K_ptr += offset_batch_head
    V_ptr += offset_batch_head
    dO_ptr += offset_batch_head
    dQ_ptr += offset_batch_head
    dK_ptr += offset_batch_head
    dV_ptr += offset_batch_head

    # This offset allows us to select the right (SEQ_LEN) vector from a tensor of shape 
    #  (BATCH_SIZE, NUM_HEADS, SEQ_LEN) given the batch and head
    #offset_batch_head_seq = (idx_batch_head * SEQ_LEN).to(tl.int64) # TODO is this original correct or my new line below?
    offset_batch_head_seq = (idx_batch * NUM_HEADS * SEQ_LEN + idx_head * SEQ_LEN).to(tl.int64) # using NUM_HEADS & SEQ_LEN to manually calculate stride
    L_ptr += offset_batch_head_seq
    D_ptr += offset_batch_head_seq

    # selecting which pid we are in respect to column splits of the (SEQ_LEN, SEQ_LEN) attention matrix
    idx_block = tl.program_id(0) # AKA our 'j' idx
    idx_block_col = idx_block * BLOCK_SIZE_COL

    offsets_row = tl.arange(0, BLOCK_SIZE_ROW)
    offsets_col = tl.arange(0, BLOCK_SIZE_COL)
    offsets_dim = tl.arange(0, HEAD_DIM)

    # Split Q, dQ, O and dO into blocks of size (BLOCK_SIZE_ROW, HEAD_DIM)
    # Split K, dK, V and dV into blocks of size (BLOCK_SIZE_COL, HEAD_DIM)
    # Split L and D into blocks of size (BLOCK_SIZE_ROW)

    # step 6
    K_j = tl.load(K_ptr + idx_block_col * stride_seq \
                    + offsets_col.expand_dims(1) * stride_seq \
                    + offsets_dim.expand_dims(0) * stride_dim)
    V_j = tl.load(V_ptr + idx_block_col * stride_seq \
                    + offsets_col.expand_dims(1) * stride_seq \
                    + offsets_dim.expand_dims(0) * stride_dim)

    # step 7
    dK_j = tl.zeros([BLOCK_SIZE_COL, HEAD_DIM], dtype=tl.float32)
    dV_j = tl.zeros([BLOCK_SIZE_COL, HEAD_DIM], dtype=tl.float32)

    T_row = tl.cdiv(SEQ_LEN, BLOCK_SIZE_ROW)
    for i in range(0, T_row): # AKA our 'i' idx
        start_row = i * BLOCK_SIZE_ROW
        start_row = tl.multiple_of(start_row, BLOCK_SIZE_ROW)

        ### step 9
        # the ones of shape (BLOCK_SIZE_ROW, HEAD_DIM)
        Q_i = tl.load(Q_ptr + start_row * stride_seq \
                        + offsets_row.expand_dims(1) * stride_seq \
                        + offsets_dim.expand_dims(0) * stride_dim)
        dO_i = tl.load(dO_ptr + start_row * stride_seq \
                        + offsets_row.expand_dims(1) * stride_seq \
                        + offsets_dim.expand_dims(0) * stride_dim)
        # the ones of shape (BLOCK_SIZE_ROW)
        L_i = tl.load(L_ptr + start_row + offsets_row)
        D_i = tl.load(D_ptr + start_row + offsets_row)

        # step 10
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * softmax_scale 
            # shape (BLOCK_SIZE_ROW, HEAD_DIM) @ (HEAD_DIM, BLOCK_SIZE_COL) -> (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        # step 11
        P_ij = tl.exp(S_ij - L_i.expand_dims(1)) # TODO try exp2
            # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL) - (BLOCK_SIZE_ROW, 1) -> (BLOCK_SIZE_ROW, BLOCK_SIZE_COL) 

        if CAUSAL:
            mask = start_row + offsets_row.expand_dims(1) >= idx_block_col + offsets_col.expand_dims(0)
            P_ij = tl.where(mask, P_ij, 0.0)

        # step 12
        dV_j += tl.dot(tl.trans(P_ij).to(tl.float16), dO_i)
            # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW) @ (BLOCK_SIZE_ROW, HEAD_DIM) -> (BLOCK_SIZE_COL, HEAD_DIM)
        # step 13
        dP_ij = tl.dot(dO_i, tl.trans(V_j)).to(tl.float32)
            # shape (BLOCK_SIZE_ROW, HEAD_DIM) @ (HEAD_DIM, BLOCK_SIZE_COL) -> (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        # step 14
        dS_ij = P_ij * (dP_ij - D_i.expand_dims(1)) 
            # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL) * [(BLOCK_SIZE_ROW, BLOCK_SIZE_COL) - (BLOCKK_SIZE_ROW, 1)] -> (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        dS_ij = dS_ij.to(tl.float16)
            # TODO build some understanding on why which operations are which dtype

        # step 15
        dQ_block_ptrs = dQ_ptr + start_row * stride_seq + offsets_row.expand_dims(1) * stride_seq + offsets_dim.expand_dims(0) * stride_dim
        delta = softmax_scale * tl.dot(dS_ij, K_j)
        tl.atomic_add(dQ_block_ptrs, delta)

        # step 16
        dK_j += softmax_scale * tl.dot(tl.trans(dS_ij), Q_i)

    dK_block_ptrs = dK_ptr + idx_block_col * stride_seq \
                    + offsets_col.expand_dims(1) * stride_seq \
                    + offsets_dim.expand_dims(0) * stride_dim
    tl.store(dK_block_ptrs, dK_j)
    dV_block_ptrs = dV_ptr + idx_block_col * stride_seq \
                    + offsets_col.expand_dims(1) * stride_seq \
                    + offsets_dim.expand_dims(0) * stride_dim
    tl.store(dV_block_ptrs, dV_j)






class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        Q, K, V, # this function assumes input has already been projected into q, k, & v's
        causal, # bool
        softmax_scale # float, almost always 1/sqrt(head_dim)
    ):
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        assert HEAD_DIM == K.shape[-1] and HEAD_DIM == V.shape[-1]
            # our implementation assumes number of query heads is the same as number of key & value heads,
            #  meaning it only supports multi-head-attention and not multi-query attention

        # pre-allocate output tensor O
        O = torch.empty_like(Q) # output tensor will be pre head concatenation and mixing

        # M is the logsumexp for the backward pass, one for each query
            # TODO logsumexp is for ...
            # TODO i don't like that we're using M for both the max values and the logsumexp. does logsumexp have a relation to max values? if not pls change
        M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32)

        # program scheduling grid
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), # primary parallelizatoin is across seq_len
            BATCH_SIZE * NUM_HEADS, # parallelize across the dimensions that don't matter
            1, # include the 1 for clarity of total dims even though it's not strictly necessary
        ) # TODO remove the 1?
        
        # calling the forward kernel
        _attn_fwd[grid](
            Q_ptr=Q, K_ptr=K, V_ptr=V,
            softmax_scale=softmax_scale,
            M_ptr=M,
            O_ptr=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=BATCH_SIZE,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            CAUSAL=causal,
        )

        # savings values for use later in backward pass
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal

        return O

    @staticmethod
    def backward(ctx, dO): # dO is the derivative of the loss function with respect to the output of the forward method
        """
        one of the optimizations flash-attention makes involves not saving all forward-pass information 
        in order to avoid storing it in DRAM, so now in the backward-pass we'll have to re-compute some of
        that information. Although this does involve redundant calculations and therefore hamper runtime,
        it's worth it in terms of how much DRAM we're saving which will effectively allow us to use a larger
        and therefore more capable model
        """
        # grabbing the saved forward pass info
        Q, K, V, O, M = ctx.saved_tensors

        # indexing is easier if all the entries in dO are lined up cleanly in DRAM as opposed to spotted around
        assert dO.is_contiguous() 
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
            # each stride tuple is (NUM_HEADS * SEQ_LEN * HEAD_DIM,     SEQ_LEN * HEAD_DIM,     HEAD_DIM,       1)
            #                      (stride_batch,                       stride_heads,           stride_seq,     stride_dim)

        # pre-allocating our eventual output gradients
        dQ = torch.zeros_like(Q) # since we use atomic_add to accumulate here directly, we use zeros instead of empty
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3] 
            # TODO why do we use ctx.HEAD_DIM instead of getting it here? is it because of the tl.constexpr?

        # heuristic meta-parameters that we could've used an autotune on when defining the kernel instead
        NUM_WARPS, NUM_STAGES = 4, 1
        # here micro and macro block sizes will trade positions over the final two kernels, one # TODO finish explaining
        #BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128
        BLOCK_SIZE_COL, BLOCK_SIZE_ROW = 32, 128 # 128, 32 # TODO what's the deal with these?
            # TODO switch this to autotune. num_warps in [4,8] and block sizes both in [32, 64, 128]

        # so as usual we combine batch_size & num_heads along the same parallelization axis
        # and here our preprocessing will also parallelize within the sequence length
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_ROW, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
            # TODO why do we call it D? what's its purpose?

        # Pre-compute D which is shape (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
        _attn_backward_preprocess[preprocess_grid](
            O_ptr=O, dO_ptr=dO, D_ptr=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_ROW, # BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM, # TODO why do we use the one from ctx instead of grabbing it above?
        )

        # the actual bulk of the kernel
        grid = (SEQ_LEN // BLOCK_SIZE_COL, BATCH_SIZE * NUM_HEADS)
        _attn_backward[grid](
            Q_ptr=Q, K_ptr=K, V_ptr=V,
            softmax_scale=ctx.softmax_scale,
            dO_ptr=dO, dQ_ptr=dQ, dK_ptr=dK, dV_ptr=dV,
            L_ptr=M, # TODO make max & logsumexp notation consistent
            D_ptr=D,
            stride_batch=Q.stride(0), stride_head=Q.stride(1), stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_ROW=BLOCK_SIZE_ROW, BLOCK_SIZE_COL=BLOCK_SIZE_COL,
            HEAD_DIM=ctx.HEAD_DIM, # TODO again why we using ctx again?
            CAUSAL=ctx.causal,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )
        
        # the None's are because pytorch expects a gradient for every input into .forward(), or a None if not applicable
        return dQ, dK, dV, None, None 


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 2e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol), f"{ref_dQ}\n{tri_dQ}"
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol), f"{ref_dK}\n{tri_dK}"
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol), f"{ref_dV}\n{tri_dV}"



BATCH, N_HEADS, HEAD_DIM = 32, 32, 64 # LOWER THESE IF YOU DON'T HAVE ENOUGH RAM
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["SEQ_LEN"],
                x_vals=[2**i for i in range(8, 14)], # LOWER 14 IF YOU DON'T HAVE ENOUGH RAM
                line_arg="provider",
                line_vals=["triton", "torch"],
                line_names=["Triton", "Torch"],
                styles=[("red", "-"), ("blue", "-")],
                ylabel="TFLOPS",
                plot_name=f"attn_original_paper-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, SEQ_LEN, HEAD_DIM, causal, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    q = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1.3
    if provider == 'triton':
        fn = lambda: TritonAttention.apply(q, k, v, causal, sm_scale)
    if provider == 'torch':
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True)
    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)



if __name__ == "__main__":
    test_op(BATCH_SIZE=1, NUM_HEADS=1, SEQ_LEN=256, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=1, NUM_HEADS=1, SEQ_LEN=256, HEAD_DIM=64, causal=False)
    print("PASSED")

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        bench_flash_attention.run(save_path='.', print_data=False)