"""
Probe available CUTLASS 4.x Python DSL APIs for SM100 (B200).
Run this before re-writing the CUTE-DSL kernel:

    python cutlass_probe.py

Prints all importable atoms, MMA descriptors, TMA ops, TMEM ops,
and whether @cute.jit can compile a trivial SM100 kernel.
"""
import sys
import importlib
import inspect

SEP = "─" * 70

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def try_import(dotpath):
    try:
        mod = importlib.import_module(dotpath)
        print(f"  ✓  {dotpath}")
        return mod
    except Exception as e:
        print(f"  ✗  {dotpath}  ({e})")
        return None

def list_attrs(mod, prefix="", filter_fn=None):
    if mod is None:
        return
    attrs = sorted(dir(mod))
    if filter_fn:
        attrs = [a for a in attrs if filter_fn(a)]
    for a in attrs:
        try:
            val = getattr(mod, a)
            print(f"    {prefix}{a}  — {type(val).__name__}")
        except Exception:
            print(f"    {prefix}{a}  — <error>")

def grep_attrs(mod, *keywords, prefix=""):
    """List attributes whose name contains any keyword (case-insensitive)."""
    if mod is None:
        return
    kws = [k.lower() for k in keywords]
    for a in sorted(dir(mod)):
        if any(k in a.lower() for k in kws):
            try:
                val = getattr(mod, a)
                print(f"    {prefix}{a}  — {type(val).__name__}")
            except Exception:
                print(f"    {prefix}{a}  — <error>")


# ── 0. versions ─────────────────────────────────────────────────────────────
section("Versions")
import torch
print(f"  Python   : {sys.version.split()[0]}")
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA cap : SM{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}" if torch.cuda.is_available() else "  CUDA     : N/A")

try:
    import cutlass
    print(f"  CUTLASS  : {cutlass.__version__}")
except Exception as e:
    print(f"  CUTLASS  : FAILED — {e}")
    sys.exit(1)

try:
    import triton
    print(f"  Triton   : {triton.__version__}")
except Exception:
    print("  Triton   : not installed")


# ── 1. Top-level cutlass modules ────────────────────────────────────────────
section("Top-level cutlass submodules")
for sub in ["cute", "cute.arch", "cute.runtime", "cute.typing",
            "cute.nvgpu", "_mlcompute"]:
    try_import(f"cutlass.{sub}")


# ── 2. cutlass.cute.arch  (where atoms usually live) ────────────────────────
section("cutlass.cute.arch  — all SM100 / TMA / TMEM / MMA entries")
arch = try_import("cutlass.cute.arch")
if arch:
    grep_attrs(arch, "sm100", "tma", "tmem", "mma", "tcgen", "wgmma",
               "copy", "f32", "bf16", prefix="  arch.")


# ── 3. cutlass.cute.nvgpu  ───────────────────────────────────────────────────
section("cutlass.cute.nvgpu  — submodules & SM100 atoms")
nvgpu = try_import("cutlass.cute.nvgpu")
if nvgpu:
    print("  Submodules:")
    for sub in sorted(dir(nvgpu)):
        if not sub.startswith("_"):
            print(f"    nvgpu.{sub}")

    # Try common sub-paths
    for sub in ["sm100", "tcgen05", "cpasync", "tma"]:
        m = try_import(f"cutlass.cute.nvgpu.{sub}")
        if m:
            grep_attrs(m, "sm100", "tma", "tmem", "mma", "tcgen",
                       "copy", "f32", "bf16", prefix=f"  nvgpu.{sub}.")


# ── 4. cutlass.cute itself ──────────────────────────────────────────────────
section("cutlass.cute top-level — kernel-writing API")
cute = try_import("cutlass.cute")
if cute:
    print("\n  Kernel decorators / JIT:")
    grep_attrs(cute, "jit", "kernel", "compile", prefix="  cute.")
    print("\n  Tile / layout:")
    grep_attrs(cute, "layout", "make", "tile", "shape", prefix="  cute.")
    print("\n  Copy / MMA / TMEM ops:")
    grep_attrs(cute, "copy", "mma", "gemm", "tmem", "tma", "mbar",
               "smem", "swizzle", prefix="  cute.")


# ── 5. cutlass.cute.runtime ──────────────────────────────────────────────────
section("cutlass.cute.runtime")
rt = try_import("cutlass.cute.runtime")
if rt:
    grep_attrs(rt, "dlpack", "tensor", "launch", prefix="  rt.")


# ── 6. Try compiling a trivial SM100 @cute.jit kernel ───────────────────────
section("Smoke-test: compile trivial @cute.jit kernel on SM100")
try:
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    @cute.jit(arch="sm_100a")
    def _trivial(a: cute.Tensor, b: cute.Tensor):
        bidx = cute.arch.block_idx_x()
        val  = cute.load(a[bidx])
        cute.store(b[bidx], val)

    a = torch.ones(4, device="cuda")
    b = torch.zeros(4, device="cuda")
    _trivial[(4,)](from_dlpack(a), from_dlpack(b))
    torch.cuda.synchronize()
    ok = torch.allclose(a, b)
    print(f"  Kernel compiled & ran correctly: {ok}")
except Exception as e:
    print(f"  FAILED: {e}")


# ── 7. Try SM100 MMA atom names ─────────────────────────────────────────────
section("Probe specific SM100 atom names")
candidates = [
    "cutlass.cute.arch.SM100_MMA_F32BF16BF16F32_SS",
    "cutlass.cute.arch.SM100_MMA_F32BF16BF16F32_SS_1CTA",
    "cutlass.cute.arch.SM100_MMA_F32F16F16F32_SS",
    "cutlass.cute.arch.SM100_MMA_F32F32F32F32_SS",
    "cutlass.cute.arch.SM100_MMA_F32F32F32F32_SS_1CTA",
    "cutlass.cute.arch.SM100_TMA_LOAD",
    "cutlass.cute.arch.SM100_TMA_STORE",
    "cutlass.cute.arch.SM100_TMEM_ALLOC",
    "cutlass.cute.arch.SM100_TMEM_LOAD",
    "cutlass.cute.arch.SM100_TMEM_STORE",
    "cutlass.cute.nvgpu.tcgen05.MMA",
    "cutlass.cute.nvgpu.tcgen05.MMA_F32BF16BF16F32",
]
for dotpath in candidates:
    parts = dotpath.rsplit(".", 1)
    try:
        mod = importlib.import_module(parts[0])
        attr = getattr(mod, parts[1])
        print(f"  ✓  {dotpath}")
    except Exception as e:
        print(f"  ✗  {dotpath}  ({e})")

print(f"\n{SEP}")
print("  Done. Share this output to get the correct kernel imports.")
print(SEP)
