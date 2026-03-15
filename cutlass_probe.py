"""
Probe available CUTLASS 4.x Python DSL APIs for SM100 (B200).
Run:  python cutlass_probe.py
"""
import sys
import importlib
import inspect

SEP = "─" * 70

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def try_import(dotpath):
    try:
        mod = importlib.import_module(dotpath)
        print(f"  ✓  {dotpath}")
        return mod
    except Exception as e:
        print(f"  ✗  {dotpath}  ({e})")
        return None

def grep_attrs(mod, *keywords, prefix="", show_sig=False):
    if mod is None:
        return
    kws = [k.lower() for k in keywords]
    for a in sorted(dir(mod)):
        if any(k in a.lower() for k in kws):
            try:
                val = getattr(mod, a)
                sig = ""
                if show_sig:
                    try:
                        sig = f"  sig={inspect.signature(val)}"
                    except Exception:
                        pass
                print(f"    {prefix}{a}  — {type(val).__name__}{sig}")
            except Exception:
                print(f"    {prefix}{a}  — <error>")

def all_attrs(mod, prefix=""):
    if mod is None:
        return
    for a in sorted(dir(mod)):
        if a.startswith("_"):
            continue
        try:
            val = getattr(mod, a)
            print(f"    {prefix}{a}  — {type(val).__name__}")
        except Exception:
            print(f"    {prefix}{a}  — <error>")


# ── versions ──────────────────────────────────────────────────────────────
section("Versions")
import torch
print(f"  Python  : {sys.version.split()[0]}")
print(f"  PyTorch : {torch.__version__}")
if torch.cuda.is_available():
    maj, minor = torch.cuda.get_device_capability()
    print(f"  GPU     : {torch.cuda.get_device_name()}  SM{maj}{minor}")
try:
    import cutlass; print(f"  CUTLASS : {cutlass.__version__}")
except Exception as e:
    print(f"  CUTLASS : FAILED — {e}"); sys.exit(1)
try:
    import triton; print(f"  Triton  : {triton.__version__}")
except Exception:
    print("  Triton  : not installed")


# ── tcgen05.mma submodule ──────────────────────────────────────────────────
section("cutlass.cute.nvgpu.tcgen05.mma — full contents")
tcgen05_mma = try_import("cutlass.cute.nvgpu.tcgen05.mma")
all_attrs(tcgen05_mma, prefix="  mma.")


# ── tcgen05.copy submodule ─────────────────────────────────────────────────
section("cutlass.cute.nvgpu.tcgen05.copy — full contents")
tcgen05_copy = try_import("cutlass.cute.nvgpu.tcgen05.copy")
all_attrs(tcgen05_copy, prefix="  copy.")


# ── arch.tmem submodule ────────────────────────────────────────────────────
section("cutlass.cute.arch.tmem — full contents")
arch_tmem = try_import("cutlass.cute.arch.tmem")
all_attrs(arch_tmem, prefix="  tmem.")


# ── MmaF16BF16Op / MmaTF32Op constructors ────────────────────────────────
section("tcgen05 MMA Op constructors and subclasses")
try:
    from cutlass.cute.nvgpu import tcgen05

    for name in ["MmaF16BF16Op", "MmaTF32Op", "MmaFP8Op"]:
        cls = getattr(tcgen05, name, None)
        if cls is None:
            print(f"  ✗  {name} not found")
            continue
        print(f"\n  {name}:")
        # list concrete subclasses if it's ABCMeta
        try:
            subs = cls.__subclasses__()
            if subs:
                for s in subs:
                    print(f"    subclass: {s.__name__}  module={s.__module__}")
                    try:
                        print(f"      sig: {inspect.signature(s)}")
                    except Exception:
                        pass
            else:
                print(f"    no subclasses found")
                try:
                    print(f"    sig: {inspect.signature(cls)}")
                except Exception:
                    pass
        except Exception as e:
            print(f"    error: {e}")

    # Also list all concrete MmaOp subclasses from tcgen05.mma module
    print("\n  Concrete Tcgen05MmaOp subclasses (from tcgen05.mma):")
    import cutlass.cute.nvgpu.tcgen05.mma as _mma_mod
    for a in sorted(dir(_mma_mod)):
        val = getattr(_mma_mod, a)
        if isinstance(val, type):
            print(f"    {a}")
            try:
                print(f"      sig: {inspect.signature(val)}")
            except Exception:
                pass

except Exception as e:
    print(f"  ERROR: {e}")


# ── arch.alloc_tmem signature ─────────────────────────────────────────────
section("arch.alloc_tmem / retrieve_tmem_ptr signatures")
try:
    from cutlass.cute.arch import alloc_tmem, retrieve_tmem_ptr, dealloc_tmem
    for fn in [alloc_tmem, retrieve_tmem_ptr, dealloc_tmem]:
        try:
            print(f"  {fn.__name__}: {inspect.signature(fn)}")
        except Exception as e:
            print(f"  {fn.__name__}: sig unavailable ({e})")
            # Try docstring
            if fn.__doc__:
                print(f"    doc: {fn.__doc__[:200]}")
except Exception as e:
    print(f"  ERROR: {e}")


# ── cpasync TMA make_tiled_tma_atom signature ─────────────────────────────
section("cpasync.make_tiled_tma_atom signature")
try:
    from cutlass.cute.nvgpu.cpasync import make_tiled_tma_atom, tma_partition
    for fn in [make_tiled_tma_atom, tma_partition]:
        try:
            print(f"  {fn.__name__}: {inspect.signature(fn)}")
        except Exception as e:
            print(f"  {fn.__name__}: sig unavailable ({e})")
            if fn.__doc__:
                print(f"    doc: {fn.__doc__[:300]}")
except Exception as e:
    print(f"  ERROR: {e}")


# ── tcgen05.make_s2t_copy / make_tmem_copy ────────────────────────────────
section("tcgen05 copy helpers")
try:
    from cutlass.cute.nvgpu import tcgen05
    for fn_name in ["make_s2t_copy", "make_tmem_copy", "get_tmem_copy_properties",
                    "make_umma_smem_desc", "tile_to_mma_shape"]:
        fn = getattr(tcgen05, fn_name, None)
        if fn is None:
            print(f"  ✗  {fn_name}")
            continue
        try:
            print(f"  {fn_name}: {inspect.signature(fn)}")
        except Exception as e:
            print(f"  {fn_name}: sig unavailable ({e})")
            if fn.__doc__:
                print(f"    doc: {fn.__doc__[:300]}")
except Exception as e:
    print(f"  ERROR: {e}")


# ── cute.make_tiled_mma signature ─────────────────────────────────────────
section("cute.make_tiled_mma / cute.gemm signatures")
try:
    import cutlass.cute as cute
    for fn_name in ["make_tiled_mma", "gemm", "make_tiled_copy", "copy",
                    "make_rmem_tensor", "make_fragment", "local_tile"]:
        fn = getattr(cute, fn_name, None)
        if fn is None:
            print(f"  ✗  {fn_name}")
            continue
        try:
            print(f"  {fn_name}: {inspect.signature(fn)}")
        except Exception as e:
            print(f"  {fn_name}: sig unavailable ({e})")
            if fn.__doc__:
                print(f"    doc: {fn.__doc__[:300]}")
except Exception as e:
    print(f"  ERROR: {e}")


# ── @cute.jit launch syntax ───────────────────────────────────────────────
section("@cute.jit launch syntax probe")
try:
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import inspect

    # Inspect jit decorator
    print(f"  cute.jit type: {type(cute.jit)}")
    try:
        print(f"  cute.jit sig: {inspect.signature(cute.jit)}")
    except Exception as e:
        print(f"  cute.jit sig: unavailable ({e})")
    if cute.jit.__doc__:
        print(f"  cute.jit doc (first 400 chars):\n    {cute.jit.__doc__[:400]}")

    print(f"\n  cute.compile type: {type(cute.compile)}")
    try:
        print(f"  cute.compile sig: {inspect.signature(cute.compile)}")
    except Exception as e:
        print(f"  cute.compile sig: unavailable ({e})")
    if hasattr(cute.compile, '__doc__') and cute.compile.__doc__:
        print(f"  cute.compile doc (first 400 chars):\n    {cute.compile.__doc__[:400]}")

    # Define trivial JIT kernel
    @cute.jit
    def _trivial(a: cute.Tensor, b: cute.Tensor):
        pass  # just test compilation / launch syntax

    print(f"\n  Decorated fn type: {type(_trivial)}")
    try:
        print(f"  Decorated fn sig: {inspect.signature(_trivial)}")
    except Exception:
        pass

    a = torch.ones(128, device="cuda", dtype=torch.float32)
    b = torch.zeros(128, device="cuda", dtype=torch.float32)
    a_c = from_dlpack(a)
    b_c = from_dlpack(b)

    launch_attempts = [
        ("grid kwarg",         lambda: _trivial(a_c, b_c, grid=(1,1,1), block=(128,1,1))),
        ("grid_dim kwarg",     lambda: _trivial(a_c, b_c, grid_dim=(1,1,1), block_dim=(128,1,1))),
        ("no grid",            lambda: _trivial(a_c, b_c)),
        ("compile+call",       lambda: cute.compile(_trivial)(a_c, b_c, grid=(1,1,1), block=(128,1,1))),
        ("compile no grid",    lambda: cute.compile(_trivial)(a_c, b_c)),
    ]
    for name, fn in launch_attempts:
        try:
            fn()
            torch.cuda.synchronize()
            print(f"  ✓  Launch '{name}' succeeded")
        except Exception as e:
            print(f"  ✗  Launch '{name}' failed: {e}")

except Exception as e:
    print(f"  ERROR in launch probe: {e}")


# ── Look for CUTLASS Python DSL examples ─────────────────────────────────
section("cutlass package examples / source location")
try:
    import cutlass, os
    pkg_dir = os.path.dirname(cutlass.__file__)
    print(f"  Package dir: {pkg_dir}")
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = [d for d in dirs if not d.startswith("__")]
        for f in files:
            if f.endswith(".py") and any(k in f.lower() for k in
                                         ["gemm", "example", "sm100", "blackwell", "cute"]):
                print(f"  {os.path.join(root, f)}")
        if root.count(os.sep) - pkg_dir.count(os.sep) > 3:
            break  # don't recurse too deep
except Exception as e:
    print(f"  {e}")


print(f"\n{SEP}\n  Done.\n{SEP}")
