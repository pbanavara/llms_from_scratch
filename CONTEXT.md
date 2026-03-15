# Qwen3.5 GatedDeltaNet Kernel Optimization — Session Context

## Goal
Replace the pure-PyTorch `torch_chunk_gated_delta_rule` in `qwen3_5_transformers.py`
with a faster kernel — Triton (working, 7-12× speedup) and CUTE-DSL / tcgen05 for B200
(in progress, targeting another ~20-30% on top of Triton).

---

## Repo
- **Remote**: `github.com:pbanavara/llms_from_scratch.git`
- **Branch**: `main`
- **Working dir on B200**: `~/llms_from_scratch/`

---

## Files written so far

| File | Status | Description |
|---|---|---|
| `qwen3_5.py` | ✅ done | Full Qwen3.5-0.8B model converted from notebook |
| `qwen3_5_transformers.py` | ✅ patched | Auto-imports fast kernel; falls back to PyTorch |
| `qwen3_5_cutedsl_kernels.py` | ✅ Triton done, CUTE pending | Triton intra-chunk kernel + inter-chunk loop + dispatch |
| `benchmark.py` | ✅ done | Kernel micro-bench + full model tokens/sec |
| `cutlass_probe.py` | ✅ done (v3) | Discovers correct CUTLASS 4.4.1 API — run this first |

---

## Benchmark results so far (B200, batch=1)

```
Backend  SeqLen  Mean(ms)  Tok/s
torch    256     7.23      35,427
triton   256     1.06      241,805   ← 6.8×
torch    512     16.57     30,902
triton   512     1.54      331,735   ← 10.7×
torch    1024    22.68     45,155
triton   1024    2.34      438,334   ← 9.7×
torch    2048    40.71     50,306
triton   2048    4.14      494,379   ← 9.8×
torch    4096    97.69     41,929
triton   4096    7.84      522,140   ← 12.5×
```

CUTE-DSL column is missing because `_CUTE_OK = False` — the kernel imports are
being rewritten using the correct CUTLASS 4.4.1 API (see below).

---

## Architecture: what the kernel does

The `GatedDeltaNet` forward pass has two phases:

### Intra-chunk phase (parallelisable — this is the kernel)
Each (batch × head × chunk) triple is independent:
```
attn[C,C] = -(k_beta @ key.T)         # C=64, Dk=128
attn       *= lower_tri_decay_mask     # exp(gc[i]-gc[j]) for i>j else 0
for i in 1..63:                        # sequential delta-correction
    attn[i,:i] += attn[i,:i] @ attn[:i,:i]
attn += I
V_out  = attn @ v_beta                 # [C, Dv=128]
KC_out = attn @ (k_beta * exp(gc))    # [C, Dk=128]
```

### Inter-chunk phase (sequential — stays in Python)
State propagation across chunks — must be sequential, inner matmuls are
already GPU-efficient PyTorch calls.

### Why B200 / CUTE-DSL wins over Triton
In the Triton kernel, the 64×64 `attn` accumulator uses shared memory,
causing SRAM round-trips on every correction iteration.
On B200 with CUTE-DSL:
- `alloc_tmem(64, smem_ptr)` puts attn in **Tensor Memory (TMEM)**
- All 3 chained MMA calls (`KB@K.T`, `attn@VB`, `attn@KB_decayed`) and the
  64 correction iterations operate on TMEM-resident attn — **zero SRAM spill**
- K/KB/VB loaded via **TMA** (async, hides latency)
- **`tcgen05.mma`** writes directly into TMEM (no store/reload between calls)

---

## Environment on B200

```
GPU     : NVIDIA B200  SM100  178 GB
Python  : 3.12.13
PyTorch : 2.10.0+cu128
Triton  : 3.6.0
CUTLASS : 4.4.1   (package: nvidia_cutlass_dsl)
CUDA cap: SM100
```

CUTLASS package path:
```
/mnt/sharefs/user11/.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/
```

---

## CUTLASS 4.4.1 API — confirmed correct imports

```python
# TMEM management
from cutlass.cute.arch import (
    alloc_tmem,               # (num_cols, smem_ptr, is_two_cta=None, arch='sm_100') -> None
    retrieve_tmem_ptr,        # (element_type, alignment, ptr_to_addr) -> Pointer
    dealloc_tmem,             # (tmem_ptr, num_cols, is_two_cta=None, arch='sm_100') -> None
    relinquish_tmem_alloc_permit,
    fence_view_async_tmem_load,
    fence_view_async_tmem_store,
)

# MMA (tcgen05 / Blackwell)
from cutlass.cute.nvgpu.tcgen05 import (
    MmaF16BF16Op,     # (ab_dtype, acc_dtype, instruction_shape, cta_group, a_src, a_major, b_major)
    MmaTF32Op,        # (instruction_shape, cta_group, a_src, a_major, b_major)
    make_s2t_copy,    # (atom, tmem_tensor) -> TiledCopy
    make_tmem_copy,   # (atom, tmem_tensor) -> TiledCopy
)
from cutlass.cute.nvgpu.tcgen05.mma import (
    CtaGroup,         # .One / .Two
    OperandSource,    # .SMEM / .RMEM
    OperandMajorMode, # .K / .MN
)

# TMA (global ↔ shared memory)
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkTensorTileG2SOp,   # TMA load  global → smem
    CopyBulkTensorTileS2GOp,   # TMA store smem → global
    make_tiled_tma_atom,       # (op, gmem_tensor, smem_layout, cta_tiler) -> (atom, tensor)
    tma_partition,             # (atom, cta_coord, cta_layout, smem, gmem) -> (smem_part, gmem_part)
)

# Layouts / tensors
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, BFloat16, Float16, Int32

# Key ops
cute.make_tiled_mma(op_or_atom, atom_layout_mnk=(1,1,1)) -> TiledMma
cute.gemm(atom, d, a, b, c)   # D = A @ B + C  (d and c are accumulators)
cute.copy(atom, src, dst)
cute.make_tensor(ptr, layout)
cute.make_rmem_tensor(shape_or_layout, dtype)
cute.make_fragment(shape_or_layout, dtype)
cute.local_tile(tensor, tiler, coord)
```

---

## API gaps — answered by cutlass_probe.py v3 (run it!)

The following are **unknown** and will be printed by `python cutlass_probe.py`:

1. **`@cute.kernel` launch syntax** — which of these works:
   ```python
   kernel[(BHC,1,1),(128,1,1)](args)   # subscript style
   kernel.launch((BHC,1,1),(128,1,1), args)
   kernel(args)                          # no grid (jit style)
   ```

2. **Shared memory declaration** — how to get a smem pointer inside a kernel
   (needed for `alloc_tmem(num_cols, smem_ptr)`). Probe looks for:
   `cute.make_smem_tensor`, `cute.arch.smem_alloc`, `extern_smem`, etc.

3. **MMA instruction shape** — which of `(64,128,16)`, `(128,128,16)`, `(64,256,16)`
   is valid for `MmaF16BF16Op` on SM100. Probe tests all variants.

4. **Example files** — probe searches `nvidia_cutlass_dsl/` for example `.py` files
   and prints the first 150 lines of the first GEMM/kernel example found.

---

## Current state of `qwen3_5_cutedsl_kernels.py`

The `_CUTE_OK` probe now correctly imports from CUTLASS 4.4.1 paths:
```python
from cutlass.cute.arch import alloc_tmem, retrieve_tmem_ptr, ...
from cutlass.cute.nvgpu.tcgen05 import MmaF16BF16Op, MmaTF32Op, ...
from cutlass.cute.nvgpu.cpasync import CopyBulkTensorTileG2SOp, ...
```

The `_intra_chunk_sm100_kernel` function body uses **incorrect fabricated API calls**
from the original session (things like `cute.Swizzle(3,4,3)`, `cute.make_smem_allocator()`,
`cute.tmem_row_view()` — none of which exist). This function needs a complete rewrite
using the correct API confirmed above.

---

## What to do next (in the new session)

### Step 1 — run the probe
```bash
cd ~/llms_from_scratch
python cutlass_probe.py 2>&1 | tee probe_out.txt
```
Paste the output to Claude. The probe will answer the 4 unknowns above.

### Step 2 — rewrite `_intra_chunk_sm100_kernel`
With the probe output, rewrite the `if _CUTE_OK:` block in
`qwen3_5_cutedsl_kernels.py` using the correct:
- `@cute.kernel` / `@cute.jit` decorator + correct launch
- `alloc_tmem` with the right smem pointer
- `MmaF16BF16Op` with the correct instruction shape
- TMA load/store via `make_tiled_tma_atom` + `tma_partition`
- `make_s2t_copy` for S→TMEM copies
- `cute.gemm` for all three MMA calls

### Step 3 — run benchmark with all 3 backends
```bash
python benchmark.py kernel --backends torch triton cute
python benchmark.py model  --backends torch triton cute
```

### Step 4 — full model tokens/sec comparison
```bash
python benchmark.py --backends torch triton cute --gen_tokens 200
```

---

## Key design decisions already made

- **Triton kernel** (`_intra_chunk_triton_kernel`): fuses 6 ops, keeps attn[64,64]
  in Triton register file (≈ SRAM), sequential correction as static loop.
  Already working and verified.

- **CUTE-DSL kernel** (`_intra_chunk_sm100_kernel`): same 6 ops but attn lives in
  TMEM throughout — no SRAM eviction during the 64-step correction loop.
  Body needs to be rewritten (see Step 2).

- **Inter-chunk loop** (`_inter_chunk_loop`): always PyTorch, no kernel needed.
  Computes per-chunk decay on-the-fly (no [B,H,NC,C,C] tensor alloc).

- **Dispatch** (`fast_chunk_gated_delta_rule`): auto picks cute > triton > torch.
  Drop-in for `torch_chunk_gated_delta_rule`. Already wired into
  `qwen3_5_transformers.py` at module import time.

- **Benchmark** (`benchmark.py`): two modes — `kernel` (micro) and `model` (e2e).
  Run with `--backends torch triton cute`.
