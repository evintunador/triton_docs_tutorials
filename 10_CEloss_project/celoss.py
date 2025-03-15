import torch
import triton
import triton.language as tl
import math

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

#import os
#os.environ["TRITON_INTERPRET"] = "1"

def naive_CELoss(x, E, targets):
    # Compute logits: (B, N, D) @ (D, V) -> (B, N, V)
    logits = x @ E
    
    # Reshape logits to (B*N, V) for cross entropy
    B, N, _ = x.shape
    V = E.shape[1]
    logits_2d = logits.reshape(-1, V)
    
    # Reshape targets to (B*N) for cross entropy
    targets_1d = targets.reshape(-1)
    
    # Compute cross entropy loss using log-softmax directly
    # 1. Apply log-softmax to logits (with numerical stability)
    max_logits, _ = torch.max(logits_2d, dim=1, keepdim=True) # (B*N)
    logits_shifted = logits_2d - max_logits # (B*N, V)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=1, keepdim=True)) + max_logits # (B*N, V)
    log_softmax = logits_2d - log_sum_exp # (B*N, V)
    
    # 2. Get negative log probabilities of target classes (NLL loss)
    nll = -log_softmax[torch.arange(log_softmax.size(0), device=targets_1d.device), targets_1d]
    #nll = -log_softmax[:, targets_1d]
    
    # 3. Average the loss
    loss = torch.mean(nll)
    
    return loss


@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config(
            {"bsN": bsN, "bsD": bsD, "bsV": bsV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for bsN in [16]#, 32, 64, 128]
        for bsD in [16]#, 32, 64, 128]
        for bsV in [16]#, 32, 64, 128]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
    ],
    key=["N", "D", "V"],
)
@triton.jit
def fused_CELoss_kernel(
    x_ptr, E_ptr, targets_ptr, out_ptr,
    stride_x_B, stride_x_N, stride_x_D,
    stride_E_D, stride_E_V,
    stride_tar_B, stride_tar_N,
    stride_out_B, stride_out_N,
    B, N, D: tl.constexpr, V: tl.constexpr,
    bsN: tl.constexpr, bsD: tl.constexpr, bsV: tl.constexpr, 
):
    pid = tl.program_id(0)
    offsets_N = tl.arange(0, bsN)
    offsets_V = tl.arange(0, bsV)

    x_ptr += pid * bsN * stride_x_B
    targets_ptr += pid * bsN * stride_tar_N
    out_ptr += pid * bsN * stride_out_N 

    M = tl.full((bsN,), value=-1e6, dtype=tl.float32)
    denominator = tl.full((bsN,), value=1.0, dtype=tl.float32)
    numerator_selected = tl.zeros((bsN,), dtype=tl.float32)

    targets = tl.load(targets_ptr + offsets_N * stride_tar_N).to(tl.int32) # (bsN)

    # moves along V dimension of (B, N, V) logits computing live softmax
    for block_start_outer in range(0, V, bsV):

        logits = tl.zeros((bsN, bsV), dtype=tl.float32)
        offsets_D = tl.arange(0, bsD)

        # moves along D dimension of (B*N, D) @ (D, V) computing matmul
        for block_start_inner in range(0, D, bsD):
            # load blocks of x and E shape (bsN, bsD) and (bsD, bsV) respectively
            x_offsets = offsets_N[:, None] * stride_x_N + offsets_D[None, :] * stride_x_D
            E_offsets = offsets_D[:, None] * stride_E_D + offsets_V[None, :] * stride_E_V
            x = tl.load(x_ptr + x_offsets)
            E = tl.load(E_ptr + E_offsets)
            logits = tl.dot(x, E, acc=logits) # shape (bsN, bsV)

            offsets_D += bsD
        offsets_V += bsV

        # find max of logits
        M_new = tl.maximum(M, tl.max(logits, axis=1)) # (bsN)
        # use logits & its max to do live softmax
        logits_shifted = logits - M_new[:, None] # (bsN, bsV)
        numerator = tl.exp(logits_shifted) # (bsN, bsV)
        alpha = tl.exp(M - M_new) # (bsN)
        denominator_new = tl.sum(numerator, axis=1)
        denominator = denominator * alpha + denominator_new # (bsN)

        # need to use targets to select values from numerator when applicable
        #targets_mask = (targets >= block_start_outer) & (targets < block_start_outer + bsV) # (bsN)
        targets_adj = targets - block_start_outer # (bsN)
        # Only select the numerator for the target class in this block
        mask = tl.arange(0, bsV)[None, :] == targets_adj[:, None] # (bsN, bsV)
        numerator_selected += tl.sum(tl.where(mask, numerator, 0.), axis=1)

        M = M_new

    P = numerator_selected / denominator
    nll = - tl.log(P)

    tl.store(out_ptr + offsets_N * stride_out_N, nll)


def fused_CELoss(x, E, targets):
    assert x.shape[-1] == E.shape[0]
    B, N, D = x.shape
    _, V = E.shape

    # pre-allocate output
    out = torch.empty((B, N), dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(B*N, meta['bsN']),)

    fused_CELoss_kernel[grid](
        x, E, targets, out,
        x.stride(0), x.stride(1), x.stride(2), 
        E.stride(0), E.stride(1), 
        targets.stride(0), targets.stride(1),
        out.stride(0), out.stride(1),
        B, N, D, V,
    )
    
    return torch.mean(out)

def test_naiveCELoss(B, N, D, V, device=DEVICE, atol=1e-3):
    torch.cuda.empty_cache()
    assert V <= 32_768
    # create data
    x = torch.randn((B, N, D), dtype=torch.float32, device=device, requires_grad=False)
    E = torch.randn((D, V), dtype=torch.float32, device=device, requires_grad=False)
    targets = torch.randint(0, V, (B, N), device=device, requires_grad=False)
    # forward passes
    naive_loss = naive_CELoss(x, E, targets)
    logits = (x @ E).reshape(-1, V)
    targets_1d = targets.reshape(-1)
    ref_loss = torch.nn.functional.cross_entropy(logits, targets_1d)
    # compare
    torch.testing.assert_close(naive_loss, ref_loss, atol=atol, rtol=0)
    print(f"naive passed {V}")


def test_fusedCELoss(B, N, D, V, device=DEVICE, atol=1e-3):
    torch.cuda.empty_cache()
    # create data
    x = torch.randn((B, N, D), dtype=torch.float32, device=device, requires_grad=False)
    E = torch.randn((D, V), dtype=torch.float32, device=device, requires_grad=False)
    targets = torch.randint(0, V, (B, N), device=device, requires_grad=False)
    # forward passes
    logits = (x @ E).reshape(-1, V)
    targets_1d = targets.reshape(-1)
    ref_loss = torch.nn.functional.cross_entropy(logits, targets_1d)
    tri_loss = fused_CELoss(x, E, targets)
    # compare
    torch.testing.assert_close(tri_loss, ref_loss, atol=atol, rtol=0)
    print(f"triton passed {V}")
    

# vary seq length for fixed head and batch=4
configs = [
    triton.testing.Benchmark(
        x_names=["V"],
        x_vals=[2 ** i for i in range(10, 14)], # LOWER IF YOU DON'T HAVE ENOUGH RAM
        line_arg="provider",
        line_vals=[
            "torch", 
            'triton'
            ],
        line_names=[
            "torch.nn.functional.cross_entropy()", 
            "Fused & sparse Triton implementation"
            ],
        styles=[
            ("red", "-"), 
            ("blue", "-")
            ],
        ylabel="TFLOPS",
        plot_name=f"CELoss-performance",
        args={},
    )
]
@triton.testing.perf_report(configs)
def bench_CELoss(V, provider, device=DEVICE):
    dtype = torch.float32
    B, N, D = 32, 1024, 384 # LOWER THESE IF YOU DON'T HAVE ENOUGH RAM
    x = torch.randn((B, N, D), dtype=dtype, device=device, requires_grad=False)
    E = torch.randn((D, V), dtype=dtype, device=device, requires_grad=False)
    targets = torch.randint(0, V, (B, N), device=device, requires_grad=False)
    if provider == 'torch':
        logits = (x @ E).reshape(-1, V)
        targets_1d = targets.reshape(-1)
        fn = lambda: torch.nn.functional.cross_entropy(logits, targets_1d)
    if provider == 'triton':
        fn = lambda: fused_CELoss(x, E, targets)
    
    # Calculate FLOPS:
    ms = triton.testing.do_bench(fn)
    # Matrix multiplication: 2*B*N*D*V (each element requires D multiplications and D-1 additions)
    # Softmax and CE loss operations: approximately 6*B*N*V
    total_flops = 2 * B * N * D * V + 6 * B * N * V
    return total_flops * 1e-12 / (ms * 1e-3)

if __name__ == "__main__":
    # always run unit-tests
    #test_naiveCELoss(32, 1024, 384, 8192) 
    #test_naiveCELoss(32, 1024, 384, 16_384) 
    #test_naiveCELoss(32, 1024, 384, 32_768) 

    test_fusedCELoss(32, 1024, 384, 32_768) 
    #test_fusedCELoss(32, 1024, 384, 65_536) 
    #test_fusedCELoss(32, 1024, 384, 131_072) 
    #test_fusedCELoss(32, 1024, 384, 262_144) 

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        bench_CELoss.run(save_path='.', print_data=True)
    


