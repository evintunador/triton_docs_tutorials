import torch

import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

from attention_original_paper import TritonAttention as attn_original_paper
from attention_umar import TritonAttention as attn_umar
from attention_triton_docs import _attention as attn_triton_docs


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
                line_vals=["torch", 'attn_original_paper', 'attn_umar', 'attn_triton_docs'],
                line_names=[
                    "torch.nn.functional.scaled_dot_product_attention", 
                    "Original Paper's Psuedocode Replication", 
                    "Umar's Implementation", 
                    "Official Triton Docs Implementation"
                    ],
                styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("purple", "-")],
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
    q = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1.3
    if provider == 'torch':
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    if provider == 'attn_original_paper':
        fn = lambda: attn_original_paper.apply(q, k, v, causal, sm_scale)
    if provider == 'attn_umar':
        fn = lambda: attn_umar.apply(q, k, v, causal, sm_scale)
    if provider == 'attn_triton_docs':
        fn = lambda: attn_triton_docs.apply(q, k, v, causal, sm_scale)
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

bench_flash_attention.run(save_path='./benchmark_results/', print_data=True)