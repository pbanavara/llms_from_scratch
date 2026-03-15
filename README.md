# Qwen3.5-0.8B — GatedDeltaNet Kernel Optimization

Custom Triton (and in-progress CUTE-DSL) kernels for the `GatedDeltaNet` linear-attention
layers in Qwen3.5-0.8B, benchmarked on **NVIDIA B200 (SM100, 178 GB)**.

## Environment

| | |
|---|---|
| GPU | NVIDIA B200 SM100 178 GB |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
| CUTLASS | 4.4.1 |
| Python | 3.12.13 |

---

## Kernel Micro-Benchmark (`benchmark.py kernel`)

Measures the `GatedDeltaNet` intra+inter chunk kernel in isolation (batch=1).

```
+---------+--------+-------+-----------+----------+---------+---------------+
| Backend | SeqLen | Batch | Mean (ms) | Std (ms) | Tok/s   | Peak Mem (MB) |
+---------+--------+-------+-----------+----------+---------+---------------+
| torch   | 256    | 1     | 8.23      | 2.09     | 31,107  | 170.3         |
| triton  | 256    | 1     | 1.04      | 0.02     | 247,333 | 182.3         |
|         |        |       |           |          | 8.0×    |               |
| torch   | 512    | 1     | 19.16     | 1.27     | 26,722  | 332.3         |
| triton  | 512    | 1     | 1.59      | 0.09     | 322,795 | 356.6         |
|         |        |       |           |          | 12.1×   |               |
| torch   | 1024   | 1     | 30.39     | 1.28     | 33,697  | 657.0         |
| triton  | 1024   | 1     | 2.37      | 0.08     | 431,878 | 705.0         |
|         |        |       |           |          | 12.8×   |               |
| torch   | 2048   | 1     | 57.46     | 3.23     | 35,640  | 1305.9        |
| triton  | 2048   | 1     | 4.19      | 0.06     | 489,286 | 1401.9        |
|         |        |       |           |          | 13.7×   |               |
| torch   | 4096   | 1     | 106.22    | 2.69     | 38,560  | 2601.6        |
| triton  | 4096   | 1     | 7.89      | 0.04     | 519,259 | 2795.6        |
|         |        |       |           |          | 13.5×   |               |
+---------+--------+-------+-----------+----------+---------+---------------+
```

**Triton speedup: 8–14× over PyTorch across sequence lengths.**

---

## Full Model Benchmark (`benchmark.py model`)

End-to-end Qwen3.5-0.8B generation, 200 tokens, batch=1.

```
+---------+------------+---------+----------+-------+---------------+
| Backend | Prompt tok | Gen tok | Time (s) | Tok/s | Peak Mem (GB) |
+---------+------------+---------+----------+-------+---------------+
| torch   | 26         | 200     | 28.65    | 7.0   | 1.77          |
| triton  | 26         | 200     | 8.25     | 24.3  | 1.77          |
+---------+------------+---------+----------+-------+---------------+
```

**3.5× end-to-end speedup** (7.0 → 24.3 tok/s) with no change in model weights or output quality.

---

## vLLM TTFT Benchmark (`vllm_inference.py`)

Time-To-First-Token via vLLM 0.17.1 (`enforce_eager=True`, `FLASH_ATTN` backend, prompt ≈ 41 tokens).

```
Backend          TTFT
─────────────────────
vLLM default     43.77 ms
```

Run commands:
```bash
# Default vLLM (FLA kernels)
VLLM_WORKER_MULTIPROC_METHOD=fork python vllm_inference.py --backend default --prompt_len 256

# Custom Triton kernel (patched into vLLM)
VLLM_WORKER_MULTIPROC_METHOD=fork python vllm_inference.py --backend custom --prompt_len 256
```

---

## Files

| File | Description |
|---|---|
| `qwen3_5.py` | Full Qwen3.5-0.8B model (from-scratch implementation) |
| `qwen3_5_transformers.py` | HuggingFace-compatible layer; auto-imports fast kernel |
| `qwen3_5_cutedsl_kernels.py` | Triton intra-chunk kernel + inter-chunk loop + dispatch |
| `benchmark.py` | Kernel micro-bench + full model tokens/sec |
| `vllm_inference.py` | vLLM TTFT benchmark (default vs custom kernel) |
| `cutlass_probe.py` | Probes CUTLASS 4.4.1 API for CUTE-DSL kernel rewrite |
| `CONTEXT.md` | Session context for ongoing CUTE-DSL kernel work |

---

## What's Next

- **CUTE-DSL kernel** (`_intra_chunk_sm100_kernel`): rewrite using confirmed
  CUTLASS 4.4.1 API (see `CONTEXT.md`). Expected additional ~20–30% over Triton
  by keeping the 64×64 `attn` accumulator in B200 Tensor Memory (TMEM) throughout
  all three chained MMA calls.
- **vLLM custom kernel TTFT**: resolve subprocess isolation so the Triton patch
  carries into the vLLM worker.
