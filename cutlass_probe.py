"""
CUTLASS 4.4.1 Python DSL — targeted probe for launch syntax + smem.
Run:  python cutlass_probe.py 2>&1 | tee probe_out.txt
"""
import sys, importlib, inspect, os, pathlib

SEP = "─" * 70
def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def try_import(p):
    try: m = importlib.import_module(p); print(f"  ✓  {p}"); return m
    except Exception as e: print(f"  ✗  {p}  ({e})"); return None

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

PKG = pathlib.Path(cutlass.__file__).parent
DSL = pathlib.Path(cutlass.__file__).parent.parent.parent  # nvidia_cutlass_dsl root

# ── 1.  Find & read example .py files ────────────────────────────────────
section("Find example files (gemm / sm100 / blackwell / kernel)")
examples_found = []
for root, dirs, files in os.walk(DSL):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for f in files:
        if f.endswith(".py") and any(k in (root+f).lower() for k in
                ["example", "gemm", "sm100", "blackwell", "tutorial", "sample"]):
            examples_found.append(os.path.join(root, f))

for p in sorted(examples_found)[:20]:
    print(f"  {p}")

# Read first example we can find
for p in sorted(examples_found):
    try:
        lines = open(p).readlines()
        if any("cute.kernel" in l or "cute.jit" in l or "launch" in l.lower() for l in lines):
            print(f"\n── First 150 lines of: {p}\n")
            print("".join(lines[:150]))
            break
    except Exception:
        pass


# ── 2.  @cute.kernel  behaviour ──────────────────────────────────────────
section("@cute.kernel — decorator type and launch syntax probe")
print(f"  cute.kernel type: {type(cute.kernel)}")
try:
    print(f"  cute.kernel sig:  {inspect.signature(cute.kernel)}")
except Exception as e:
    print(f"  cute.kernel sig:  unavailable ({e})")
if cute.kernel.__doc__:
    print(f"  cute.kernel doc:\n    {cute.kernel.__doc__[:600]}")

# Define a real @cute.kernel that does work
try:
    from cutlass.cute.arch import (
        alloc_tmem, retrieve_tmem_ptr, dealloc_tmem,
        relinquish_tmem_alloc_permit,
    )
    from cutlass.cute.nvgpu.tcgen05 import MmaF16BF16Op, CtaGroup
    from cutlass.cute.nvgpu.tcgen05.mma import OperandSource, OperandMajorMode
    from cutlass.cute.nvgpu.cpasync import (
        CopyBulkTensorTileG2SOp, CopyBulkTensorTileS2GOp, make_tiled_tma_atom, tma_partition
    )
    from cutlass.cute.typing import Float32, BFloat16, Float16
    print("  All imports OK")
except Exception as e:
    print(f"  Import failed: {e}")

@cute.kernel
def _kernel_probe(a: cute.Tensor, b: cute.Tensor):
    bidx = cute.arch.block_idx_x()
    val  = cute.arch.load(a[bidx])
    cute.arch.store(b[bidx], val)

print(f"\n  @cute.kernel decorated fn type: {type(_kernel_probe)}")
try: print(f"  sig: {inspect.signature(_kernel_probe)}")
except Exception: pass
print(f"  dir: {[x for x in dir(_kernel_probe) if not x.startswith('__')]}")

# Try launch variants
a = torch.ones(4, device="cuda", dtype=torch.float32)
b = torch.zeros(4, device="cuda", dtype=torch.float32)
ac, bc = from_dlpack(a), from_dlpack(b)

attempts = [
    ("subscript [(4,1,1),(128,1,1)]",
     lambda: _kernel_probe[(4,1,1),(128,1,1)](ac, bc)),
    ("subscript [(4,),(128,)]",
     lambda: _kernel_probe[(4,),(128,)](ac, bc)),
    ("no grid",
     lambda: _kernel_probe(ac, bc)),
    (".launch(grid,block,args)",
     lambda: _kernel_probe.launch((4,1,1),(128,1,1), ac, bc)
     if hasattr(_kernel_probe, "launch") else (_ for _ in ()).throw(AttributeError("no .launch"))),
    ("cute.compile then call",
     lambda: cute.compile(_kernel_probe, ac, bc)(ac, bc)),
]
for name, fn in attempts:
    try:
        fn(); torch.cuda.synchronize()
        print(f"  ✓  '{name}' succeeded")
    except Exception as e:
        print(f"  ✗  '{name}' failed: {type(e).__name__}: {str(e)[:120]}")


# ── 3.  Shared memory declaration ────────────────────────────────────────
section("Shared-memory tensor creation")
smem_fns = ["make_smem_tensor", "make_smem_ptr", "smem_alloc",
            "declare_smem", "extern_smem", "SharedMemory"]
for fn_name in smem_fns:
    val = getattr(cute, fn_name, None) or getattr(cute.arch if hasattr(cute,'arch') else None, fn_name, None)
    print(f"  cute.{fn_name}: {'✓  ' + str(type(val)) if val else '✗  not found'}")

# Check cute.arch for smem helpers
import cutlass.cute.arch as _arch
for a in sorted(dir(_arch)):
    if "smem" in a.lower() or "shared" in a.lower() or "extern" in a.lower():
        print(f"  arch.{a}: {type(getattr(_arch, a))}")

# Also check cute module
for a in sorted(dir(cute)):
    if "smem" in a.lower() or "shared" in a.lower() or "extern" in a.lower():
        print(f"  cute.{a}: {type(getattr(cute, a))}")


# ── 4.  alloc_tmem  — minimal usage test ─────────────────────────────────
section("alloc_tmem minimal usage inside @cute.jit")

@cute.jit
def _tmem_test(out: cute.Tensor):
    """Try to allocate TMEM inside a jit function and write 1.0 to first element."""
    from cutlass.cute.arch import alloc_tmem, retrieve_tmem_ptr, dealloc_tmem
    tidx = cute.arch.thread_idx_x()
    # Attempt: use a register as the address holder
    # (In CUDA PTX, alloc_tmem writes address to smem — we need a smem ptr)
    # Probe: what happens if we pass a tensor element's address?
    # This will likely fail but error message will be informative
    addr_buf = cute.make_rmem_tensor((1,), cute.arch.Int32 if hasattr(cute.arch, 'Int32') else None)

try:
    out = torch.zeros(1, device="cuda", dtype=torch.float32)
    _tmem_test(from_dlpack(out))
    torch.cuda.synchronize()
    print("  _tmem_test succeeded")
except Exception as e:
    print(f"  _tmem_test failed: {type(e).__name__}: {str(e)[:300]}")


# ── 5.  cutlass.cute.typing — available types ─────────────────────────────
section("cutlass.cute.typing — available numeric types")
try:
    import cutlass.cute.typing as ctyping
    for a in sorted(dir(ctyping)):
        if not a.startswith("_"):
            print(f"  {a}: {type(getattr(ctyping, a)).__name__}")
except Exception as e:
    print(f"  {e}")


# ── 6.  Any CUTLASS Python DSL higher-level examples on disk ─────────────
section("Search nvidia_cutlass_dsl examples more broadly")
for root, dirs, files in os.walk(DSL.parent.parent.parent):  # widen search
    dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", "node_modules"]
               and not d.startswith(".")]
    for f in files:
        if f.endswith(".py") and "example" in root.lower():
            p = os.path.join(root, f)
            print(f"  {p}")
    if root.count(os.sep) - str(DSL.parent.parent.parent).count(os.sep) > 5:
        break


# ── 7.  tcgen05.mma  MmaF16BF16Op  minimal construction test ─────────────
section("MmaF16BF16Op construction + make_tiled_mma test")
try:
    from cutlass.cute.nvgpu.tcgen05 import MmaF16BF16Op
    from cutlass.cute.nvgpu.tcgen05.mma import CtaGroup, OperandSource, OperandMajorMode
    from cutlass.cute.typing import BFloat16, Float32

    print("  CtaGroup values:", list(CtaGroup))
    print("  OperandSource values:", list(OperandSource))
    print("  OperandMajorMode values:", list(OperandMajorMode))

    # Try to construct the MMA op
    for shape in [(64, 128, 16), (128, 128, 16), (64, 256, 16)]:
        try:
            op = MmaF16BF16Op(
                ab_dtype=BFloat16,
                acc_dtype=Float32,
                instruction_shape=shape,
                cta_group=CtaGroup.One,
                a_src=OperandSource.SMEM,
                a_major_mode=OperandMajorMode.K,
                b_major_mode=OperandMajorMode.K,
            )
            tiled = cute.make_tiled_mma(op)
            print(f"  ✓  MmaF16BF16Op{shape} + make_tiled_mma  OK  → {type(tiled)}")
            break
        except Exception as e:
            print(f"  ✗  MmaF16BF16Op{shape}: {str(e)[:100]}")

    # Try MmaTF32Op
    from cutlass.cute.nvgpu.tcgen05 import MmaTF32Op
    for shape in [(64, 128, 8), (128, 128, 8), (64, 256, 8)]:
        try:
            op = MmaTF32Op(
                instruction_shape=shape,
                cta_group=CtaGroup.One,
                a_src=OperandSource.SMEM,
                a_major_mode=OperandMajorMode.K,
                b_major_mode=OperandMajorMode.K,
            )
            tiled = cute.make_tiled_mma(op)
            print(f"  ✓  MmaTF32Op{shape} + make_tiled_mma  OK  → {type(tiled)}")
            break
        except Exception as e:
            print(f"  ✗  MmaTF32Op{shape}: {str(e)[:100]}")

except Exception as e:
    print(f"  ERROR: {e}")


print(f"\n{SEP}\n  Done.\n{SEP}")
