"""
Qwen3.5 GatedDeltaNet kernel benchmark — B200 before/after.

Two benchmark modes:

  1. kernel  — micro-benchmarks Qwen3_5GatedDeltaNet.forward() in isolation
               across PyTorch / Triton / CUTE-DSL backends.
               Reports: latency (ms), throughput (tok/s), peak GPU memory.

  2. model   — loads full Qwen3.5-0.8B, runs generation with each backend,
               reports end-to-end tokens/sec and GPU memory.

Usage:
  python benchmark.py kernel               # kernel micro-bench only
  python benchmark.py model                # full model bench only
  python benchmark.py                      # both (default)

  python benchmark.py --seq_lens 512 1024 2048 4096
  python benchmark.py --batch 4
  python benchmark.py --model_id Qwen/Qwen3.5-0.8B
  python benchmark.py --backends torch triton   # skip cute if not installed
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ── pretty table helper ────────────────────────────────────────────────────

def _sep(widths):
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

def _row(cells, widths):
    parts = [f" {str(c):<{w}} " for c, w in zip(cells, widths)]
    return "|" + "|".join(parts) + "|"

def print_table(headers, rows, title=""):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = _sep(widths)
    if title:
        print(f"\n{'─'*len(sep)}")
        print(f"  {title}")
        print(f"{'─'*len(sep)}")
    print(sep)
    print(_row(headers, widths))
    print(sep)
    for row in rows:
        print(_row(row, widths))
    print(sep)


# ── GPU info ───────────────────────────────────────────────────────────────

def gpu_info() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    name  = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name}  SM{major}{minor}  {mem_gb:.0f} GB"

def peak_mem_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2

def reset_mem():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ── timing utility ────────────────────────────────────────────────────────

def bench(fn, warmup=5, iters=20) -> Tuple[float, float]:
    """Returns (mean_ms, std_ms)."""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Kernel micro-benchmark
# ══════════════════════════════════════════════════════════════════════════════

def build_gated_delta_net(cfg: dict, device: torch.device):
    """Build a single Qwen3_5GatedDeltaNet layer."""
    from qwen3_5_transformers import Qwen3_5GatedDeltaNet

    class _Cfg:
        hidden_size           = cfg["emb_dim"]
        linear_num_value_heads= cfg["linear_num_value_heads"]
        linear_num_key_heads  = cfg["linear_num_key_heads"]
        linear_key_head_dim   = cfg["linear_key_head_dim"]
        linear_value_head_dim = cfg["linear_value_head_dim"]
        linear_conv_kernel_dim= cfg["linear_conv_kernel_dim"]
        hidden_act            = "silu"
        rms_norm_eps          = 1e-6
        dtype                 = cfg["dtype"]

    layer = Qwen3_5GatedDeltaNet(_Cfg(), layer_idx=0).to(device)
    layer.eval()
    return layer


def run_kernel_bench(
    backends:  List[str],
    seq_lens:  List[int],
    batch:     int,
    device:    torch.device,
    cfg:       dict,
    warmup:    int = 5,
    iters:     int = 20,
):
    from qwen3_5_cutedsl_kernels import (
        fast_chunk_gated_delta_rule,
        get_backend,
        _TRITON_OK,
        _CUTE_OK,
    )
    from qwen3_5_transformers import torch_chunk_gated_delta_rule

    avail = {
        "torch":  True,
        "triton": _TRITON_OK,
        "cute":   _CUTE_OK,
    }
    backends = [b for b in backends if avail.get(b, False)]
    if not backends:
        print("No backends available — skipping kernel bench.")
        return

    H   = cfg["linear_num_value_heads"]
    Dk  = cfg["linear_key_head_dim"]
    Dv  = cfg["linear_value_head_dim"]
    C   = 64   # chunk_size

    rows = []
    headers = ["Backend", "SeqLen", "Batch", "Mean (ms)", "Std (ms)",
               "Tok/s", "Peak Mem (MB)"]

    for seq_len in seq_lens:
        NC = (seq_len + C - 1) // C   # number of chunks (padded)

        # Build random inputs in the shape GatedDeltaNet expects
        def make_inputs():
            q  = torch.randn(batch, seq_len, H, Dk, device=device, dtype=torch.float32)
            k  = torch.randn_like(q)
            v  = torch.randn(batch, seq_len, H, Dv, device=device, dtype=torch.float32)
            g  = torch.randn(batch, seq_len, H,  1, device=device, dtype=torch.float32) * 0.1
            bt = torch.sigmoid(torch.randn(batch, seq_len, H, device=device))
            # Transpose to (B, H, T, D) as expected by the kernel
            q  = q.transpose(1, 2)
            k  = k.transpose(1, 2)
            v  = v.transpose(1, 2)
            g  = g.transpose(1, 2).squeeze(-1)
            bt = bt.transpose(1, 2)
            return q, k, v, g, bt

        q, k, v, g, bt = make_inputs()

        for backend in backends:
            reset_mem()
            if backend == "torch":
                fn = lambda: torch_chunk_gated_delta_rule(
                    q, k, v, g, bt, chunk_size=C,
                    use_qk_l2norm_in_kernel=True
                )
            else:
                fn = lambda: fast_chunk_gated_delta_rule(
                    q, k, v, g, bt, chunk_size=C,
                    use_qk_l2norm_in_kernel=True,
                    backend=backend,
                )

            try:
                mean_ms, std_ms = bench(fn, warmup=warmup, iters=iters)
                mem_mb = peak_mem_mb()
                toks_per_s = (batch * seq_len) / (mean_ms / 1000)
                rows.append([
                    backend,
                    seq_len,
                    batch,
                    f"{mean_ms:.2f}",
                    f"{std_ms:.2f}",
                    f"{toks_per_s:,.0f}",
                    f"{mem_mb:.1f}",
                ])
            except Exception as e:
                rows.append([backend, seq_len, batch, "ERROR", "─", "─", str(e)[:40]])

        # blank separator between seq_len groups
        rows.append(["─"] * len(headers))

    print_table(headers, rows[:-1],   # drop trailing separator
                title=f"GatedDeltaNet kernel  |  {gpu_info()}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Full model benchmark (tokens / sec)
# ══════════════════════════════════════════════════════════════════════════════

QWEN3_5_CONFIG = {
    "vocab_size": 248_320,
    "context_length": 262_144,
    "emb_dim": 1_024,
    "n_heads": 8,
    "n_layers": 24,
    "hidden_dim": 3_584,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-6,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "dtype": torch.bfloat16,
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ],
}


def patch_model_backend(backend: str):
    """
    Monkey-patch Qwen3_5GatedDeltaNet to use the requested kernel backend.
    Called before model instantiation so every linear-attention layer picks
    up the right kernel.
    """
    import qwen3_5_transformers as t
    from qwen3_5_cutedsl_kernels import fast_chunk_gated_delta_rule, _TRITON_OK, _CUTE_OK

    if backend == "torch":
        # Restore original PyTorch fallback
        t.chunk_gated_delta_rule = None   # Qwen3_5GatedDeltaNet will use torch_chunk_gated_delta_rule
    elif backend in ("triton", "cute"):
        if backend == "triton" and not _TRITON_OK:
            print(f"  [warn] Triton not available, falling back to torch for model bench.")
            patch_model_backend("torch")
            return
        if backend == "cute" and not _CUTE_OK:
            print(f"  [warn] CUTE-DSL not available, falling back to Triton.")
            patch_model_backend("triton")
            return
        # Replace the module-level function used by Qwen3_5GatedDeltaNet.forward()
        t.chunk_gated_delta_rule = lambda *a, **kw: fast_chunk_gated_delta_rule(
            *a, **kw, backend=backend
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def load_model(model_id: str, device: torch.device):
    """Download weights and load full model. Returns (model, tokenizer)."""
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download, hf_hub_download
    from qwen3_5 import Qwen3_5Model, load_weights_into_qwen3_5, Qwen3_5Tokenizer

    local_dir = Path(model_id).parts[-1]
    print(f"  Downloading {model_id} ...")
    repo_dir  = snapshot_download(repo_id=model_id, local_dir=local_dir)
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weights = {}
    for fn in sorted(set(index["weight_map"].values())):
        weights.update(load_file(os.path.join(repo_dir, fn)))

    model = Qwen3_5Model(QWEN3_5_CONFIG).to(device)
    load_weights_into_qwen3_5(model, QWEN3_5_CONFIG, weights)
    model.to(device)
    del weights
    gc.collect()
    torch.cuda.empty_cache()

    hf_hub_download(repo_id=model_id, filename="tokenizer.json", local_dir=local_dir)
    tokenizer = Qwen3_5Tokenizer(
        tokenizer_file_path=f"{local_dir}/tokenizer.json",
        repo_id=model_id,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )
    return model, tokenizer


def generate_n_tokens(model, prompt_ids: torch.Tensor, n: int, eos_id: int) -> float:
    """
    Generate n tokens (or until EOS).  Returns wall-clock seconds.
    """
    model.eval()
    ids = prompt_ids.clone()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            logits = model(ids)[:, -1]
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            if torch.all(next_tok == eos_id):
                break
            ids = torch.cat([ids, next_tok], dim=1)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_model_bench(
    backends:   List[str],
    model_id:   str,
    gen_tokens: int,
    device:     torch.device,
    warmup_tok: int = 10,
):
    from qwen3_5_cutedsl_kernels import _TRITON_OK, _CUTE_OK

    avail = {"torch": True, "triton": _TRITON_OK, "cute": _CUTE_OK}
    backends = [b for b in backends if avail.get(b, False)]

    PROMPT = "Explain the significance of the Qwen3.5 architecture for efficient inference."

    rows = []
    headers = ["Backend", "Prompt tok", "Gen tok", "Time (s)", "Tok/s",
               "Peak Mem (GB)"]

    for backend in backends:
        print(f"\n  Loading model with backend={backend} ...")
        patch_model_backend(backend)
        model, tokenizer = load_model(model_id, device)
        model.eval()

        prompt_ids = torch.tensor(
            tokenizer.encode(PROMPT), device=device
        ).unsqueeze(0)
        prompt_len = prompt_ids.shape[1]

        # Warmup
        print(f"    Warming up ({warmup_tok} tokens) ...")
        generate_n_tokens(model, prompt_ids, warmup_tok, tokenizer.eos_token_id)

        # Measure
        reset_mem()
        print(f"    Measuring ({gen_tokens} tokens) ...")
        elapsed = generate_n_tokens(model, prompt_ids, gen_tokens, tokenizer.eos_token_id)
        mem_gb  = peak_mem_mb() / 1024

        tps = gen_tokens / elapsed
        rows.append([
            backend,
            prompt_len,
            gen_tokens,
            f"{elapsed:.2f}",
            f"{tps:.1f}",
            f"{mem_gb:.2f}",
        ])

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print_table(headers, rows,
                title=f"Full model  Qwen3.5-0.8B  |  {gpu_info()}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="both",
                   choices=["kernel", "model", "both"],
                   help="What to benchmark (default: both)")
    p.add_argument("--backends", nargs="+", default=["torch", "triton", "cute"],
                   help="Backends to test (default: torch triton cute)")
    p.add_argument("--seq_lens", nargs="+", type=int,
                   default=[256, 512, 1024, 2048, 4096],
                   help="Sequence lengths for kernel micro-bench")
    p.add_argument("--batch", type=int, default=1,
                   help="Batch size (default: 1)")
    p.add_argument("--gen_tokens", type=int, default=100,
                   help="Tokens to generate in model bench (default: 100)")
    p.add_argument("--warmup", type=int, default=5,
                   help="Warmup iterations for kernel bench (default: 5)")
    p.add_argument("--iters", type=int, default=20,
                   help="Measurement iterations for kernel bench (default: 20)")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-0.8B",
                   help="HuggingFace model ID (default: Qwen/Qwen3.5-0.8B)")
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional path to save results as JSON")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device found. Exiting.")
        return

    device = torch.device("cuda")
    print(f"\nDevice : {gpu_info()}")
    print(f"PyTorch: {torch.__version__}")

    try:
        import triton
        print(f"Triton : {triton.__version__}")
    except ImportError:
        print("Triton : not installed")

    try:
        import cutlass
        print(f"CUTLASS: {cutlass.__version__}")
    except ImportError:
        print("CUTLASS: not installed")

    from qwen3_5_cutedsl_kernels import get_backend
    print(f"Auto backend: {get_backend()}")

    results = {}

    if args.mode in ("kernel", "both"):
        print("\n" + "═" * 60)
        print("  KERNEL MICRO-BENCHMARK")
        print("═" * 60)
        run_kernel_bench(
            backends=args.backends,
            seq_lens=args.seq_lens,
            batch=args.batch,
            device=device,
            cfg=QWEN3_5_CONFIG,
            warmup=args.warmup,
            iters=args.iters,
        )

    if args.mode in ("model", "both"):
        print("\n" + "═" * 60)
        print("  FULL MODEL BENCHMARK")
        print("═" * 60)
        run_model_bench(
            backends=args.backends,
            model_id=args.model_id,
            gen_tokens=args.gen_tokens,
            device=device,
            warmup_tok=5,
        )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
