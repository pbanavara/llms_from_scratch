"""
TTFT benchmark: default vLLM kernels vs custom Triton/CUTE-DSL kernel
----------------------------------------------------------------------
Measures Time-To-First-Token (TTFT) for:
  1. Default vLLM  — uses vLLM's built-in FLA / FlashInfer GDN kernels
  2. Custom kernel — monkey-patches ChunkGatedDeltaRule with our
                     fast_chunk_gated_delta_rule (Triton on B200, or CUTE-DSL)

TTFT proxy: llm.generate(prompt, SamplingParams(max_tokens=1))
  → only the prefill pass runs, so wall time ≈ TTFT.

Usage (activate venv first):
  source /mnt/sharefs/user11/.venv/bin/activate
  cd ~/llms_from_scratch

  # Both backends, single prompt
  python vllm_inference.py

  # Sweep prompt lengths
  python vllm_inference.py --sweep

  # Only default vLLM backend
  python vllm_inference.py --backend default

  # Only custom kernel backend
  python vllm_inference.py --backend custom

Model checkpoint expected at ./Qwen3.5-0.8B/
"""

import argparse
import sys
import time

import torch

MODEL_DIR = "./Qwen3.5-0.8B"

# Prompts of increasing length for the sweep
SWEEP_PROMPTS = {
    64:   "Explain the attention mechanism in transformers." * 1,
    256:  "Explain the attention mechanism in transformers." * 4,
    512:  "Explain the attention mechanism in transformers." * 8,
    1024: "Explain the attention mechanism in transformers." * 16,
    2048: "Explain the attention mechanism in transformers." * 32,
}
DEFAULT_PROMPT = SWEEP_PROMPTS[256]

CHAT_TEMPLATE = (
    "<|im_start|>user\n{msg}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


# ---------------------------------------------------------------------------
# Monkey-patch: replace vLLM's ChunkGatedDeltaRule with our kernel
# ---------------------------------------------------------------------------

def _patch_custom_kernel():
    """
    Replace vLLM's ChunkGatedDeltaRule.forward_native (and forward_cuda) with
    our fast_chunk_gated_delta_rule.

    Interface contract (confirmed from qwen3_next.py):
      q, k, v    : [1, T, H, D]   — batch-first, single sequence for TTFT
      g, beta    : [1, T, H]      — pre-cumsum log-decay / gate
      initial_state: [1, H, Dk, Dv]
      cu_seqlens : None or [0, T]  — ignored (single-sequence path)
    Returns: (out [1, T, H, Dv], final_state [1, H, Dk, Dv])

    Our fast_chunk_gated_delta_rule accepts the same shapes.
    """
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qwen3_5_cutedsl_kernels import fast_chunk_gated_delta_rule

    backend = "triton"   # explicitly use Triton; CUTE-DSL kernel body is not yet complete
    print(f"[patch] Custom kernel backend: {backend}")

    from vllm.model_executor.models.qwen3_next import ChunkGatedDeltaRule

    def _custom_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens=None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        # cu_seqlens is not supported by our kernel.
        # For the TTFT benchmark (single request), cu_seqlens is either None
        # or [0, T], so we can safely ignore it.
        if cu_seqlens is not None and cu_seqlens.numel() > 2:
            # More than one sequence in the batch — fall back to FLA.
            from vllm.model_executor.layers.fla.ops import (
                chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
            )
            return fla_chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        return fast_chunk_gated_delta_rule(
            query=q, key=k, value=v, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            backend="triton",
        )

    ChunkGatedDeltaRule.forward_native = _custom_forward
    ChunkGatedDeltaRule.forward_cuda   = _custom_forward  # override SM90 path too
    print("[patch] ChunkGatedDeltaRule patched with custom kernel.")
    return backend


# ---------------------------------------------------------------------------
# vLLM setup
# ---------------------------------------------------------------------------

def _make_llm():
    from vllm import LLM
    return LLM(
        model=MODEL_DIR,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,   # disable torch.compile — ensures our patch runs
        # Force FLASH_ATTN backend — FlashInfer (default on B200) requires
        # ninja for JIT compilation which is not available in this environment.
        attention_config={"backend": "FLASH_ATTN"},
        # mamba_cache_mode="none",   # default; use "align" to enable prefix caching
    )


# ---------------------------------------------------------------------------
# TTFT measurement
# ---------------------------------------------------------------------------

def measure_ttft(llm, prompt: str, n_warmup: int = 2) -> float:
    """
    Returns TTFT in milliseconds.
    Uses max_tokens=1 so only the prefill pass runs.
    """
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    wrapped = CHAT_TEMPLATE.format(msg=prompt)

    # Warm-up (compile / JIT triton kernels)
    for _ in range(n_warmup):
        llm.generate([wrapped], sp)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate([wrapped], sp)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000   # ms


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_default(prompt: str):
    print("\n" + "=" * 60)
    print("Backend: DEFAULT vLLM (FLA / FlashInfer GDN kernels)")
    print("=" * 60)
    llm = _make_llm()
    ttft = measure_ttft(llm, prompt)
    print(f"TTFT: {ttft:.2f} ms  (prompt ≈ {len(prompt.split())} words)")
    return ttft


def run_custom(prompt: str):
    print("\n" + "=" * 60)
    print("Backend: CUSTOM kernel (Triton / CUTE-DSL)")
    print("=" * 60)
    backend = _patch_custom_kernel()
    llm = _make_llm()
    ttft = measure_ttft(llm, prompt)
    print(f"TTFT: {ttft:.2f} ms  (prompt ≈ {len(prompt.split())} words)  [{backend}]")
    return ttft, backend


def run_sweep():
    """
    Because monkey-patching is process-wide, we run each backend in a
    subprocess so they don't interfere.  Results are collected and printed.
    """
    import subprocess, json

    results = {}  # {seq_len: {"default": ms, "custom": ms}}

    for backend in ("default", "custom"):
        for seq_len, prompt in SWEEP_PROMPTS.items():
            cmd = [
                sys.executable, __file__,
                "--backend", backend,
                "--prompt_len", str(seq_len),
                "--json",
            ]
            out = subprocess.check_output(cmd, text=True)
            # last line is JSON: {"ttft_ms": ..., "backend": ...}
            data = json.loads(out.strip().split("\n")[-1])
            results.setdefault(seq_len, {})[backend] = data["ttft_ms"]

    print("\n" + "=" * 70)
    print(f"{'SeqLen':>8}  {'Default(ms)':>12}  {'Custom(ms)':>12}  {'Speedup':>8}")
    print("-" * 70)
    for seq_len in sorted(results):
        d = results[seq_len].get("default", float("nan"))
        c = results[seq_len].get("custom",  float("nan"))
        sp = d / c if c > 0 else float("nan")
        print(f"{seq_len:>8}  {d:>12.1f}  {c:>12.1f}  {sp:>8.2f}×")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["default", "custom", "both"], default="both")
    ap.add_argument("--sweep",   action="store_true", help="Sweep prompt lengths")
    ap.add_argument("--prompt_len", type=int, default=256)
    ap.add_argument("--json",    action="store_true", help="Print result as JSON")
    args = ap.parse_args()

    if args.sweep:
        run_sweep()
        return

    prompt = SWEEP_PROMPTS.get(args.prompt_len, DEFAULT_PROMPT)

    if args.backend == "both":
        # Run in subprocesses to isolate the monkey-patch
        import subprocess, json
        results = {}
        for b in ("default", "custom"):
            cmd = [sys.executable, __file__, "--backend", b,
                   "--prompt_len", str(args.prompt_len), "--json"]
            out = subprocess.check_output(cmd, text=True)
            data = json.loads(out.strip().split("\n")[-1])
            results[b] = data
        print("\n" + "=" * 60)
        print("TTFT Summary")
        print("=" * 60)
        d = results["default"]["ttft_ms"]
        c = results["custom"]["ttft_ms"]
        kb = results["custom"].get("kernel_backend", "?")
        print(f"  Default vLLM : {d:.2f} ms")
        print(f"  Custom kernel: {c:.2f} ms  [{kb}]")
        print(f"  Speedup      : {d/c:.2f}×" if c > 0 else "  Speedup: N/A")
        return

    if args.backend == "default":
        ttft = run_default(prompt)
        if args.json:
            import json
            print(json.dumps({"ttft_ms": ttft, "backend": "default"}))

    elif args.backend == "custom":
        ttft, kb = run_custom(prompt)
        if args.json:
            import json
            print(json.dumps({"ttft_ms": ttft, "backend": "custom",
                              "kernel_backend": kb}))


if __name__ == "__main__":
    main()
