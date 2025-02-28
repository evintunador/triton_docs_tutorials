import torch
import triton
import math

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

from alternative_implementations.original_paper import TritonAttention as original_paper
from alternative_implementations.umar import TritonAttention as umar
from alternative_implementations.triton_docs import _attention as triton_docs
from flash_attention import _flashattention as this_tutorial


BATCH, N_HEADS, HEAD_DIM = 32, 4, 64 # LOWER THESE IF YOU DON'T HAVE ENOUGH RAM
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True]:#, False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["SEQ_LEN"],
                x_vals=[2**i for i in range(8, 10)], # LOWER IF YOU DON'T HAVE ENOUGH RAM
                line_arg="provider",
                line_vals=["torch", 'original_paper', 'umar', 'triton_docs'] + (['this_tutorial'] if causal else []),
                line_names=[
                    "torch.nn.functional.scaled_dot_product_attention", 
                    "Original Paper's Psuedocode Replication", 
                    "Umar's Implementation", 
                    "Official Triton Docs Implementation",
                    ] + (["This tutorial's optimized implementation"] if causal else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("purple", "-")] + ([("pink", "-")] if causal else []),
                ylabel="TFLOPS",
                plot_name=f"attention-performance-{mode}-causal={causal}",
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
    q = 0.02 * torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = 0.02 * torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = 0.02 * torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1 / math.sqrt(HEAD_DIM)
    if provider == 'torch':
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    if provider == 'original_paper':
        fn = lambda: original_paper.apply(q, k, v, causal, sm_scale)
    if provider == 'umar':
        fn = lambda: umar.apply(q, k, v, causal, sm_scale)
    if provider == 'triton_docs':
        fn = lambda: triton_docs.apply(q, k, v, causal, sm_scale)
    if provider == 'this_tutorial':
        fn = lambda: this_tutorial.apply(q, k, v, sm_scale)
    if mode == "bwd":
        O = fn()
        dLdO = torch.randn_like(O)
        fn = lambda: O.backward(dLdO, retain_graph=True)
    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)

bench_flash_attention.run(save_path='./benchmark_results/', print_data=True)