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
    from cutlass.cute.typing import Float32
    bidx, _, _ = cute.arch.block_idx()
    val  = cute.arch.load(a[bidx], Float32)
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
    """Probe alloc_smem + thread_idx inside a jit function (no alloc_tmem — needs @cute.kernel)."""
    from cutlass.cute.typing import Int32
    tidx, _, _ = cute.arch.thread_idx()
    smem_buf = cute.arch.alloc_smem(Int32, 1)  # just test that alloc_smem compiles

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
section("MmaF16BF16Op construction + make_tiled_mma test (inside @cute.jit)")
from cutlass.cute.nvgpu.tcgen05 import MmaF16BF16Op, MmaTF32Op
from cutlass.cute.nvgpu.tcgen05.mma import CtaGroup, OperandSource, OperandMajorMode
from cutlass.cute.typing import BFloat16, Float32 as F32

print("  CtaGroup values:", list(CtaGroup))
print("  OperandSource values:", list(OperandSource))
print("  OperandMajorMode values:", list(OperandMajorMode))

# MmaF16BF16Op must be constructed inside a DSL compilation context (@cute.jit or @cute.kernel)
for shape in [(64, 64, 16), (64, 128, 16), (128, 128, 16), (64, 256, 16)]:
    @cute.jit
    def _mma_probe_bf16(dummy: cute.Tensor):
        op = MmaF16BF16Op(
            ab_dtype=BFloat16,
            acc_dtype=F32,
            instruction_shape=shape,
            cta_group=CtaGroup.ONE,
            a_src=OperandSource.SMEM,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
        )
        tiled = cute.make_tiled_mma(op)
    try:
        _mma_probe_bf16(from_dlpack(torch.zeros(1, device="cuda", dtype=torch.float32)))
        torch.cuda.synchronize()
        print(f"  ✓  MmaF16BF16Op{shape} + make_tiled_mma  OK")
        break
    except Exception as e:
        print(f"  ✗  MmaF16BF16Op{shape}: {str(e)[:120]}")

for shape in [(64, 128, 8), (128, 128, 8), (64, 256, 8)]:
    @cute.jit
    def _mma_probe_tf32(dummy: cute.Tensor):
        op = MmaTF32Op(
            instruction_shape=shape,
            cta_group=CtaGroup.ONE,
            a_src=OperandSource.SMEM,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
        )
        tiled = cute.make_tiled_mma(op)
    try:
        _mma_probe_tf32(from_dlpack(torch.zeros(1, device="cuda", dtype=torch.float32)))
        torch.cuda.synchronize()
        print(f"  ✓  MmaTF32Op{shape} + make_tiled_mma  OK")
        break
    except Exception as e:
        print(f"  ✗  MmaTF32Op{shape}: {str(e)[:120]}")


# ── 8b. @cute.kernel via @cute.jit launch  (step-by-step isolation) ──────
section("@cute.kernel launched from @cute.jit  (step-by-step)")
import traceback as _tb

# Step 1: minimal no-op kernel via jit launcher
try:
    @cute.kernel
    def _k_noop(dummy: cute.Tensor):
        pass

    @cute.jit
    def _launch_noop(dummy: cute.Tensor):
        _k_noop(dummy).launch(grid=[1, 1, 1], block=[128, 1, 1])

    _launch_noop(from_dlpack(torch.zeros(1, device="cuda")))
    torch.cuda.synchronize()
    print("  step 1 (noop kernel via jit): ✓")
except Exception as e:
    print(f"  step 1 (noop kernel via jit): ✗  {type(e).__name__}: {str(e)[:200]}")
    _tb.print_exc()

# Step 2: SMEM alloc + thread_idx inside @cute.kernel via jit
try:
    @cute.kernel
    def _k_smem(dummy: cute.Tensor):
        from cutlass.cute.typing import Float32, Int32
        tidx, _, _ = cute.arch.thread_idx()
        s = cute.arch.alloc_smem(Float32, 128)

    @cute.jit
    def _launch_smem(dummy: cute.Tensor):
        _k_smem(dummy).launch(grid=[1, 1, 1], block=[128, 1, 1])

    _launch_smem(from_dlpack(torch.zeros(1, device="cuda")))
    torch.cuda.synchronize()
    print("  step 2 (SMEM alloc in kernel via jit): ✓")
except Exception as e:
    print(f"  step 2: ✗  {type(e).__name__}: {str(e)[:200]}")
    _tb.print_exc()

# Step 3: MMA op construction inside @cute.kernel via jit
try:
    @cute.kernel
    def _k_mma_op(dummy: cute.Tensor):
        from cutlass.cute.typing import Float32, BFloat16
        from cutlass.cute.nvgpu.tcgen05 import MmaF16BF16Op
        from cutlass.cute.nvgpu.tcgen05.mma import CtaGroup, OperandSource, OperandMajorMode
        tidx, _, _ = cute.arch.thread_idx()
        op = MmaF16BF16Op(
            ab_dtype=BFloat16, acc_dtype=Float32,
            instruction_shape=(64, 128, 16),
            cta_group=CtaGroup.ONE,
            a_src=OperandSource.SMEM,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        thr_mma = tiled_mma.get_slice(tidx)

    @cute.jit
    def _launch_mma_op(dummy: cute.Tensor):
        _k_mma_op(dummy).launch(grid=[1, 1, 1], block=[128, 1, 1])

    _launch_mma_op(from_dlpack(torch.zeros(1, device="cuda")))
    torch.cuda.synchronize()
    print("  step 3 (MMA op + tiled_mma in kernel via jit): ✓")
except Exception as e:
    print(f"  step 3: ✗  {type(e).__name__}: {str(e)[:200]}")
    _tb.print_exc()

# Step 3b: smem_layout_atom and tile_to_mma_shape (needed for correct SMEM layout)
try:
    from cutlass.cute.nvgpu.tcgen05 import (
        SmemLayoutAtomKind, make_smem_layout_atom, tile_to_mma_shape
    )
    from cutlass.cute.nvgpu.tcgen05.mma import CtaGroup, OperandSource, OperandMajorMode
    from cutlass.cute.nvgpu.tcgen05 import MmaF16BF16Op
    from cutlass.cute.typing import BFloat16, Float32
    import inspect
    print("  SmemLayoutAtomKind values:", list(SmemLayoutAtomKind))
    print("  make_smem_layout_atom sig:", inspect.signature(make_smem_layout_atom))
    print("  tile_to_mma_shape sig:", inspect.signature(tile_to_mma_shape))
    print("  step 3b (smem_layout_atom imports): ✓")
except Exception as e:
    print(f"  step 3b: ✗  {type(e).__name__}: {str(e)[:200]}")

# Step 4: partition with swizzled SMEM layout + TMEM (no fill — TMEM fill not supported)
try:
    @cute.kernel
    def _k_partition(dummy: cute.Tensor):
        from cutlass.cute.typing import Float32, BFloat16, Int32
        from cutlass.cute.nvgpu.tcgen05 import MmaF16BF16Op, make_smem_layout_atom, SmemLayoutAtomKind
        from cutlass.cute.nvgpu.tcgen05.mma import CtaGroup, OperandSource, OperandMajorMode
        from cutlass.cute.arch import alloc_tmem, retrieve_tmem_ptr, dealloc_tmem, relinquish_tmem_alloc_permit

        tidx, _, _ = cute.arch.thread_idx()
        M, K, N = 64, 64, 128

        # Swizzled SMEM layout atom for BF16 K-major access (matches tcgen05 requirements)
        atom_A = make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, BFloat16)
        atom_B = make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, BFloat16)
        layout_A = cute.tile_to_shape(atom_A, (M, K), (1, 0))   # M-outer, K-inner
        layout_B = cute.tile_to_shape(atom_B, (K, N), (0, 1))   # K-outer, N-inner

        sA = cute.arch.alloc_smem(BFloat16, M * K)
        sB = cute.arch.alloc_smem(BFloat16, K * N)
        smem_addr = cute.arch.alloc_smem(Int32, 1)
        alloc_tmem(N, smem_addr)
        tmem_ptr = retrieve_tmem_ptr(Float32, 128, smem_addr)

        op = MmaF16BF16Op(
            ab_dtype=BFloat16, acc_dtype=Float32,
            instruction_shape=(64, 128, 16),
            cta_group=CtaGroup.ONE,
            a_src=OperandSource.SMEM,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        thr_mma = tiled_mma.get_slice(tidx)

        sA_t  = cute.make_tensor(sA,       layout_A)
        sB_t  = cute.make_tensor(sB,       layout_B)
        tAcc  = cute.make_tensor(tmem_ptr, cute.make_layout((M, N)))

        tArA = thr_mma.partition_A(sA_t)
        tBrB = thr_mma.partition_B(sB_t)
        tCtC = thr_mma.partition_C(tAcc)
        # Note: TMEM doesn't support .fill() — zero-init requires S2T copy
        # relinquish_tmem_alloc_permit must precede dealloc_tmem
        cute.arch.sync_threads()
        relinquish_tmem_alloc_permit()
        dealloc_tmem(tmem_ptr, N)

    @cute.jit
    def _launch_partition(dummy: cute.Tensor):
        _k_partition(dummy).launch(grid=[1, 1, 1], block=[128, 1, 1])

    _launch_partition(from_dlpack(torch.zeros(1, device="cuda")))
    torch.cuda.synchronize()
    print("  step 4 (swizzled SMEM + TMEM partition — no fill): ✓")
except Exception as e:
    print(f"  step 4: ✗  {type(e).__name__}: {str(e)[:300]}")
    _tb.print_exc()

# Step 5: full cute.gemm call with swizzled SMEM (no zero-init — tests API only)
try:
    @cute.kernel
    def _k_gemm(mA: cute.Tensor, mB: cute.Tensor):
        from cutlass.cute.typing import Float32, BFloat16, Int32
        from cutlass.cute.nvgpu.tcgen05 import MmaF16BF16Op, make_smem_layout_atom, SmemLayoutAtomKind
        from cutlass.cute.nvgpu.tcgen05.mma import CtaGroup, OperandSource, OperandMajorMode
        from cutlass.cute.arch import alloc_tmem, retrieve_tmem_ptr, dealloc_tmem, relinquish_tmem_alloc_permit

        tidx, _, _ = cute.arch.thread_idx()
        M, K, N = 64, 64, 128

        atom_A = make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, BFloat16)
        atom_B = make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, BFloat16)
        layout_A = cute.tile_to_shape(atom_A, (M, K), (1, 0))
        layout_B = cute.tile_to_shape(atom_B, (K, N), (0, 1))

        sA = cute.arch.alloc_smem(BFloat16, M * K)
        sB = cute.arch.alloc_smem(BFloat16, K * N)
        for i in range(M * K // 128):
            idx = tidx + i * 128
            cute.arch.store(sA + idx, cute.arch.load(mA[idx], BFloat16))
        for i in range(K * N // 128):
            idx = tidx + i * 128
            cute.arch.store(sB + idx, cute.arch.load(mB[idx], BFloat16))
        cute.arch.sync_threads()

        smem_addr = cute.arch.alloc_smem(Int32, 1)
        alloc_tmem(N, smem_addr)
        tmem_ptr = retrieve_tmem_ptr(Float32, 128, smem_addr)

        op = MmaF16BF16Op(
            ab_dtype=BFloat16, acc_dtype=Float32,
            instruction_shape=(64, 128, 16),
            cta_group=CtaGroup.ONE,
            a_src=OperandSource.SMEM,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        # tcgen05 MMA requires explicitly setting ACCUMULATE field before gemm
        from cutlass.cute.nvgpu import tcgen05 as _t5
        tiled_mma.set(_t5.Field.ACCUMULATE, False)
        thr_mma = tiled_mma.get_slice(tidx)

        sA_t = cute.make_tensor(sA,       layout_A)
        sB_t = cute.make_tensor(sB,       layout_B)
        tAcc = cute.make_tensor(tmem_ptr, cute.make_layout((M, N)))

        tArA = thr_mma.partition_A(sA_t)
        tBrB = thr_mma.partition_B(sB_t)
        tCtC = thr_mma.partition_C(tAcc)
        # gemm: tAcc = sA @ sB (ACCUMULATE=False means no addition of prior tAcc)
        cute.gemm(tiled_mma, tCtC, tArA, tBrB, tCtC)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.sync_threads()
        relinquish_tmem_alloc_permit()
        dealloc_tmem(tmem_ptr, N)

    @cute.jit
    def _launch_gemm(mA: cute.Tensor, mB: cute.Tensor):
        _k_gemm(mA, mB).launch(grid=[1, 1, 1], block=[128, 1, 1])

    a_flat = torch.ones(64 * 64,  device="cuda", dtype=torch.bfloat16)
    b_flat = torch.ones(64 * 128, device="cuda", dtype=torch.bfloat16)
    _launch_gemm(from_dlpack(a_flat), from_dlpack(b_flat))
    torch.cuda.synchronize()
    print("  step 5 (cute.gemm swizzled SMEM→TMEM): ✓")
except Exception as e:
    print(f"  step 5: ✗  {type(e).__name__}: {str(e)[:300]}")
    _tb.print_exc()


# ── 8.  alloc_tmem inside @cute.kernel  (last — may crash) ───────────────
section("alloc_tmem inside @cute.kernel (last test — crash here is OK)")

@cute.kernel
def _tmem_kernel_probe(dummy: cute.Tensor):
    """Test alloc_tmem inside @cute.kernel — the correct scope for CTA-level TMEM."""
    from cutlass.cute.arch import alloc_tmem, retrieve_tmem_ptr, dealloc_tmem, relinquish_tmem_alloc_permit
    from cutlass.cute.typing import Float32, Int32
    smem_buf = cute.arch.alloc_smem(Int32, 1)
    alloc_tmem(64, smem_buf)
    tmem_ptr = retrieve_tmem_ptr(Float32, 128, smem_buf)
    cute.arch.sync_threads()
    relinquish_tmem_alloc_permit()
    dealloc_tmem(tmem_ptr, 64)

try:
    dummy = from_dlpack(torch.zeros(1, device="cuda", dtype=torch.float32))
    _tmem_kernel_probe(dummy)
    torch.cuda.synchronize()
    print("  alloc_tmem inside @cute.kernel: ✓ succeeded")
except Exception as e:
    print(f"  alloc_tmem inside @cute.kernel: ✗  {type(e).__name__}: {str(e)[:300]}")

print(f"\n{SEP}\n  Done.\n{SEP}")
