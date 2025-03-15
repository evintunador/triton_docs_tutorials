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
    max_logits, _ = torch.max(logits_2d, dim=1, keepdim=True)
    logits_shifted = logits_2d - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=1, keepdim=True)) + max_logits
    log_softmax = logits_2d - log_sum_exp
    
    # 2. Get negative log probabilities of target classes (NLL loss)
    nll = -log_softmax[torch.arange(log_softmax.size(0), device=targets_1d.device), targets_1d]
    
    # 3. Average the loss
    loss = torch.mean(nll)
    
    return loss


@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config(
            {"bs": bs},
            num_stages=num_stages, num_warps=num_warps,
        )
        for bs in [16]#, 32, 64, 128]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
    ],
    key=["???"],
)
@triton.jit
def fused_CELoss_kernel():
    pass


def fused_CELoss(x, E, targets):
    pass


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
    test_naiveCELoss(32, 1024, 384, 8192) 
    test_naiveCELoss(32, 1024, 384, 16_384) 
    test_naiveCELoss(32, 1024, 384, 32_768) 

    #test_fusedCELoss(32, 1024, 384, 32_768) 
    #test_fusedCELoss(32, 1024, 384, 65_536) 
    #test_fusedCELoss(32, 1024, 384, 131_072) 
    #test_fusedCELoss(32, 1024, 384, 262_144) 

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        bench_CELoss.run(save_path='.', print_data=True)
    


