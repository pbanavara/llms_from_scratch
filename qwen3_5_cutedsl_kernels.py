"""Fast GatedDeltaNet kernels for Qwen3.5 — B200 (SM100) CUTE-DSL + Triton fallback.

Kernel hierarchy (auto-selected at runtime):
  1. CUTE-DSL   — SM100 / B200: TMEM-resident attn accumulator, TMA loads,
                  tcgen05.mma with direct TMEM accumulation.
                  Requires: CUTLASS ≥ 3.6, CUDA ≥ 12.8, Blackwell GPU.
  2. Triton     — Any CUDA GPU: fuses decay, KB@K.T, delta-correction,
                  V_out and KC_out matmuls into one kernel.
                  Requires: triton ≥ 3.0.
  3. PyTorch    — Pure-PyTorch fallback from qwen3_5_transformers.py.

Drop-in replacement:
    from qwen3_5_cutedsl_kernels import fast_chunk_gated_delta_rule
    # pass as chunk_gated_delta_rule= to Qwen3_5GatedDeltaNet

B200 design notes
-----------------
The intra-chunk phase is embarrassingly parallel over (batch × head × chunk).
Each CTA owns one chunk and does:

    attn[C,C] = -(KB @ K.T)          # tcgen05.mma  → TMEM
    attn      *= lower_tri_decay      # in-place on TMEM
    for i in 1..C-1:                  # sequential, TMEM-resident
        attn[i,:i] += attn[i,:i] @ attn[:i,:i]
    attn += I
    V_out  = attn @ VB                # tcgen05.mma  TMEM × SRAM → SRAM
    KC_out = attn @ (KB * exp(gc))    # tcgen05.mma  TMEM × SRAM → SRAM

Key B200 advantages:
  • TMEM (256 KB/SM): the 64×64 attn matrix (16 KB) stays in Tensor Memory
    for all three chained MMA calls — zero SRAM spill, zero store/reload.
  • TMA: K, KB, VB are bulk-copied asynchronously; hides load latency.
  • tcgen05.mma: direct TMEM accumulation — no explicit store between
    KB@K.T, the correction loop, and the two output matmuls.

The inter-chunk loop (sequential state propagation) remains in Python/PyTorch;
its inner matmuls are already GPU-efficient.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# ── capability probes ──────────────────────────────────────────────────────

def _gpu_sm() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor

_SM = _gpu_sm()
_IS_B200 = _SM >= 100       # SM100 = Blackwell (B200)
_IS_HOPPER = _SM >= 90      # SM90  = Hopper  (H100) — Triton works well here too

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False

# CUTE-DSL: requires nvidia-cutlass with SM100 support.
# Atom locations confirmed by cutlass_probe.py on CUTLASS 4.4.1 / SM100:
#   TMEM  → cutlass.cute.arch   (alloc_tmem, retrieve_tmem_ptr, dealloc_tmem)
#   MMA   → cutlass.cute.nvgpu.tcgen05  (MmaF16BF16Op / MmaTF32Op)
#   TMA   → cutlass.cute.nvgpu.cpasync  (CopyBulkTensorTileG2SOp)
#   S2T   → cutlass.cute.nvgpu.tcgen05  (make_s2t_copy)
_CUTE_IMPORT_ERROR: str = ""
try:
    import cutlass.cute as cute                                          # noqa: F401
    from cutlass.cute.runtime import from_dlpack as cute_from_dlpack    # noqa: F401
    from cutlass.cute.arch import (                                      # noqa: F401
        alloc_tmem, retrieve_tmem_ptr, dealloc_tmem,
        relinquish_tmem_alloc_permit,
        fence_view_async_tmem_load, fence_view_async_tmem_store,
    )
    from cutlass.cute.nvgpu.tcgen05 import (                            # noqa: F401
        MmaF16BF16Op, MmaTF32Op,
        make_s2t_copy, make_tmem_copy,
    )
    from cutlass.cute.nvgpu.cpasync import (                            # noqa: F401
        CopyBulkTensorTileG2SOp, CopyBulkTensorTileS2GOp,
        make_tiled_tma_atom, tma_partition,
    )
    _CUTE_OK = _IS_B200
except Exception as _e:
    _CUTE_OK = False
    _CUTE_IMPORT_ERROR = str(_e)
    if _IS_B200:
        print(f"[qwen3_5_cutedsl] CUTE-DSL unavailable: {_e}")
        print("[qwen3_5_cutedsl] Run `python cutlass_probe.py` for details.")


# ── utilities ──────────────────────────────────────────────────────────────

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def get_backend() -> str:
    if _CUTE_OK:
        return "cute"
    if _TRITON_OK:
        return "triton"
    return "torch"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TRITON intra-chunk kernel
#
#  Grid : (B * H * n_chunks,)  — one program per chunk, fully independent.
#  Each program loads K/KB/VB/GC for its chunk, keeps attn[C,C] in the
#  register file (64×64×4 = 16 KB), and writes V_out/KC_out to GMEM.
#
#  Fused ops (vs. PyTorch baseline):
#    cumsum(g)  →  build decay mask  →  -(KB @ K.T)  →  mask+apply decay  →
#    sequential delta-correction  →  attn @ VB  →  attn @ KB_decayed
#
#  No intermediate [B,H,n_chunks,C,C] decay tensor is ever materialised.
# ══════════════════════════════════════════════════════════════════════════════

if _TRITON_OK:
    @triton.jit
    def _intra_chunk_triton_kernel(
        K_ptr, KB_ptr, VB_ptr, GC_ptr,
        Vout_ptr, KCout_ptr,
        C:  tl.constexpr,    # chunk_size  = 64
        Dk: tl.constexpr,    # key head dim = 128
        Dv: tl.constexpr,    # val head dim = 128
    ):
        """One Triton program owns one (batch, head, chunk) triple."""
        bhc = tl.program_id(0)

        r  = tl.arange(0, C)    # [C]  row / col indices
        dk = tl.arange(0, Dk)   # [Dk] key-dim indices
        dv = tl.arange(0, Dv)   # [Dv] val-dim indices

        # ── Load chunk data ────────────────────────────────────────────────
        base_kdk = bhc * C * Dk
        base_vdv = bhc * C * Dv
        base_gc  = bhc * C

        # [C, Dk]
        K  = tl.load(K_ptr  + base_kdk + r[:, None] * Dk + dk[None, :])
        KB = tl.load(KB_ptr + base_kdk + r[:, None] * Dk + dk[None, :])
        # [C, Dv]
        VB = tl.load(VB_ptr + base_vdv + r[:, None] * Dv + dv[None, :])
        # [C] — g already cumsum'd by the caller
        gc = tl.load(GC_ptr + base_gc + r)

        # ── attn = -(KB @ K.T)  shape [C, C] ─────────────────────────────
        # tl.dot([C,Dk], [Dk,C]) — M=64, K=128, N=64  ✓
        attn = -tl.dot(KB, tl.trans(K))   # fp32 accumulation

        # ── Lower-triangular decay mask ───────────────────────────────────
        # attn[i,j] *= exp(gc[i] - gc[j])  for i > j,  else 0
        gi   = gc[:, None]                        # [C, 1]
        gj   = gc[None, :]                        # [1, C]
        mask = r[:, None] > r[None, :]            # [C, C]  lower-tri (strict)
        attn = tl.where(mask, attn * tl.exp(gi - gj), 0.0)

        # ── Sequential delta-correction ───────────────────────────────────
        # For i = 1..C-1:
        #   attn[i, :i] += attn[i, :i] @ attn[:i, :i]
        #
        # In Triton we cannot dynamically index rows, so we use masked
        # reductions. All 64 iterations run inside one GPU program
        # (no Python overhead), and the 64×64 attn lives in the register
        # file throughout — equivalent to TMEM on Hopper/Ampere.
        for i in tl.static_range(1, C):
            # row_i[k] = attn[i, k]  if k < i  else  0   — shape [C]
            row_i = tl.sum(
                tl.where((r[:, None] == i) & (r[None, :] < i), attn, 0.0),
                axis=0,
            )  # [C]

            # correction[j] = sum_k row_i[k] * attn[k, j]  (vec @ mat)
            correction = tl.sum(row_i[:, None] * attn, axis=0)  # [C]

            # Write correction into row i, columns < i only
            row_sel = (r == i).to(tl.float32)          # [C]
            col_sel = (r  < i).to(tl.float32)          # [C]
            attn = attn + row_sel[:, None] * (col_sel * correction)[None, :]

        # Add identity diagonal
        attn = attn + tl.where(r[:, None] == r[None, :], 1.0, 0.0)

        # ── V_out = attn @ VB   [C, Dv] ─────────────────────────────────
        # tl.dot([C,C], [C,Dv]) — M=64, K=64, N=128  ✓
        Vout = tl.dot(attn.to(VB.dtype), VB)

        # ── KC_out = attn @ (KB * exp(gc))  [C, Dk] ─────────────────────
        KB_dec = KB * tl.exp(gc[:, None])   # scale row k by exp(gc[k])
        KCout  = tl.dot(attn.to(KB_dec.dtype), KB_dec)

        # ── Store ─────────────────────────────────────────────────────────
        tl.store(Vout_ptr  + base_vdv + r[:, None] * Dv + dv[None, :], Vout)
        tl.store(KCout_ptr + base_kdk + r[:, None] * Dk + dk[None, :], KCout)


    def triton_intra_chunk_fwd(
        K: torch.Tensor,
        KB: torch.Tensor,
        VB: torch.Tensor,
        G_cum: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args (all float32, contiguous):
            K, KB : [BHC, C, Dk]
            VB    : [BHC, C, Dv]
            G_cum : [BHC, C]   cumsum(g) along chunk dim
        Returns:
            V_out  : [BHC, C, Dv]  delta-corrected values
            KC_out : [BHC, C, Dk]  k_cumdecay
        """
        BHC, C, Dk = K.shape
        Dv = VB.shape[-1]
        K, KB, VB, G_cum = [t.contiguous().float() for t in (K, KB, VB, G_cum)]
        V_out  = torch.empty(BHC, C, Dv, device=K.device, dtype=K.dtype)
        KC_out = torch.empty(BHC, C, Dk, device=K.device, dtype=K.dtype)
        _intra_chunk_triton_kernel[(BHC,)](K, KB, VB, G_cum, V_out, KC_out,
                                           C=C, Dk=Dk, Dv=Dv)
        return V_out, KC_out


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CUTE-DSL intra-chunk kernel  (SM100 / B200)
#
#  Requires: nvidia-cutlass >= 3.6, CUDA >= 12.8, Blackwell GPU (SM100).
#
#  Versus Triton, the key difference is that the 64×64 `attn` accumulator
#  lives in **TMEM** (Tensor Memory) for the entire kernel:
#
#    • tcgen05.mma writes KB@K.T directly into TMEM — no SRAM store.
#    • The 64 correction iterations read/write TMEM in-place without any
#      SRAM round-trip — this is the primary win over Triton on B200.
#    • Both output matmuls (attn@VB, attn@KB_dec) use TMEM as the left
#      operand; results land in SRAM and are flushed via TMA store.
#    • K, KB, VB are loaded via TMA with a pipeline barrier, hiding
#      global-memory latency behind the prior chunk's correction.
#
#  SRAM budget per CTA (f32, C=64, Dk=Dv=128):
#    sK, sKB : 2 × 64×128×4 = 64 KB
#    sVB     : 64×128×4     = 32 KB
#    sVout   : 64×128×4     = 32 KB
#    sKCout  : 64×128×4     = 32 KB
#    Total   ≈ 160 KB  (B200 has 256 KB shared mem / CTA → fits)
#
#  TMEM budget: 64×64×4 = 16 KB  (B200 has 256 KB TMEM / SM → trivial)
# ══════════════════════════════════════════════════════════════════════════════

if _CUTE_OK:
    @cute.jit(arch="sm_100a")
    def _intra_chunk_sm100_kernel(
        mK:     cute.Tensor,    # global [BHC, C, Dk]
        mKB:    cute.Tensor,    # global [BHC, C, Dk]
        mVB:    cute.Tensor,    # global [BHC, C, Dv]
        mGC:    cute.Tensor,    # global [BHC, C]
        mVout:  cute.Tensor,    # global [BHC, C, Dv]  (output)
        mKCout: cute.Tensor,    # global [BHC, C, Dk]  (output)
        C:  cute.Constexpr,     # 64
        Dk: cute.Constexpr,     # 128
        Dv: cute.Constexpr,     # 128
    ):
        """One CTA per (batch × head × chunk).  Grid = (BHC,)."""
        bhc = cute.arch.block_idx_x()

        # ── Shared memory ─────────────────────────────────────────────────
        # Swizzle(3,4,3): 128-byte swizzle, optimal for tcgen05.mma SRAM reads.
        smem   = cute.make_smem_allocator()
        swz    = cute.Swizzle(3, 4, 3)
        layout_kd = cute.make_layout((C, Dk), swizzle=swz)
        layout_vd = cute.make_layout((C, Dv), swizzle=swz)

        sK     = smem.allocate(layout_kd, dtype=cute.Float32)
        sKB    = smem.allocate(layout_kd, dtype=cute.Float32)
        sVB    = smem.allocate(layout_vd, dtype=cute.Float32)
        sVout  = smem.allocate(layout_vd, dtype=cute.Float32)
        sKCout = smem.allocate(layout_kd, dtype=cute.Float32)

        # ── TMA async loads ───────────────────────────────────────────────
        # Three simultaneous TMA transactions; a single mbarrier with
        # arrive_count=3 waits for all three before proceeding.
        tma_K  = cute.make_tma_copy(SM100_TMA_LOAD(), mK)
        tma_KB = cute.make_tma_copy(SM100_TMA_LOAD(), mKB)
        tma_VB = cute.make_tma_copy(SM100_TMA_LOAD(), mVB)

        mbar = cute.make_mbarrier(arrive_count=3)
        cute.tma_load(tma_K,  mK [bhc], sK,  mbar)
        cute.tma_load(tma_KB, mKB[bhc], sKB, mbar)
        cute.tma_load(tma_VB, mVB[bhc], sVB, mbar)
        cute.mbarrier_wait(mbar, phase=0)

        # ── TMEM allocation ───────────────────────────────────────────────
        # tAttn[C, C] lives in Tensor Memory for the entire kernel body.
        # tcgen05.mma accumulates into it without any SRAM store/reload.
        tmem_alloc = cute.make_tmem_allocator()
        tAttn = tmem_alloc.allocate(
            cute.make_layout((C, C)), dtype=cute.Float32
        )

        # ── Step 1: tAttn = -(sKB @ sK.T)  via tcgen05.mma ───────────────
        # SM100_MMA_F32F32F32F32_SS_1CTA: SRAM operands → TMEM accumulator.
        mma = cute.make_tiled_mma(
            SM100_MMA_F32F32F32F32_SS_1CTA(),
            cute.make_layout((1, 1, 1)),   # 1 CTA-group
        )
        cute.gemm(mma, sKB, cute.transpose(sK), tAttn)   # TMEM ← KB @ K.T
        cute.transform_inplace(tAttn, lambda x: -x)       # negate

        # ── Step 2: lower-triangular decay mask ──────────────────────────
        # Load g_cum for this chunk into registers (only 64 fp32 scalars).
        gc = cute.load_to_register(mGC[bhc])              # [C] register tensor

        # Apply in-place: attn[i,j] *= exp(gc[i]-gc[j]) if i>j else 0.
        @cute.elementwise_inplace(tAttn)
        def _apply_decay(val, i, j):
            return cute.select(i > j, val * cute.exp(gc[i] - gc[j]), 0.0)

        # ── Step 3: sequential delta-correction  (TMEM-resident) ─────────
        # attn[i, :i] += attn[i, :i] @ attn[:i, :i]   for i = 1 .. C-1
        #
        # Operates entirely within TMEM — no SRAM traffic.
        # cute.tmem_matvec_acc(row, mat, row) computes row += row @ mat
        # using warp-level TMEM read-modify-write.
        for i in cute.static_range(1, C):
            row_i = cute.tmem_row_view(tAttn, row=i, cols=slice(0, i))
            sub_i = cute.tmem_submatrix_view(tAttn, rows=slice(0, i),
                                              cols=slice(0, i))
            cute.tmem_matvec_acc(row_i, sub_i, row_i)  # row_i += row_i @ sub_i

        # Add identity diagonal
        @cute.elementwise_inplace(tAttn)
        def _add_eye(val, i, j):
            return val + cute.select(i == j, 1.0, 0.0)

        # ── Step 4: sVout = tAttn @ sVB  (TMEM × SRAM → SRAM) ────────────
        cute.gemm(mma, tAttn, sVB, sVout)

        # ── Step 5: sKCout = tAttn @ (sKB * exp(gc))  ────────────────────
        # Scale KB rows in-place: sKB[k, :] *= exp(gc[k]).
        @cute.elementwise_inplace(sKB)
        def _scale_kb(val, i, j):
            return val * cute.exp(gc[i])

        cute.gemm(mma, tAttn, sKB, sKCout)

        # ── TMA stores ────────────────────────────────────────────────────
        tma_Vout  = cute.make_tma_copy(SM100_TMA_STORE(), mVout)
        tma_KCout = cute.make_tma_copy(SM100_TMA_STORE(), mKCout)
        cute.tma_store(tma_Vout,  sVout,  mVout [bhc])
        cute.tma_store(tma_KCout, sKCout, mKCout[bhc])
        cute.tma_store_fence()


    def cute_intra_chunk_fwd(
        K: torch.Tensor,
        KB: torch.Tensor,
        VB: torch.Tensor,
        G_cum: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args (float32, contiguous):
            K, KB  : [BHC, C, Dk]
            VB     : [BHC, C, Dv]
            G_cum  : [BHC, C]
        Returns:
            V_out  : [BHC, C, Dv]
            KC_out : [BHC, C, Dk]
        """
        BHC, C, Dk = K.shape
        Dv = VB.shape[-1]
        K, KB, VB, G_cum = [t.contiguous().float() for t in (K, KB, VB, G_cum)]
        V_out  = torch.empty(BHC, C, Dv, device=K.device, dtype=K.dtype)
        KC_out = torch.empty(BHC, C, Dk, device=K.device, dtype=K.dtype)
        _intra_chunk_sm100_kernel[(BHC,)](
            cute_from_dlpack(K),
            cute_from_dlpack(KB),
            cute_from_dlpack(VB),
            cute_from_dlpack(G_cum),
            cute_from_dlpack(V_out),
            cute_from_dlpack(KC_out),
            C=C, Dk=Dk, Dv=Dv,
        )
        return V_out, KC_out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Inter-chunk state propagation  (inherently sequential)
#
#  This loop cannot be parallelised: chunk i+1's initial state comes from
#  chunk i.  The inner matmuls are standard BLAS calls — already efficient.
#  We only recompute the per-chunk decay_mask here (no full [B,H,NC,C,C]
#  tensor allocation).
# ══════════════════════════════════════════════════════════════════════════════

def _inter_chunk_loop(
    query:        torch.Tensor,    # [B, H, NC, C, Dk]  l2-normed & scaled
    key:          torch.Tensor,    # [B, H, NC, C, Dk]  l2-normed
    V_corrected:  torch.Tensor,    # [B, H, NC, C, Dv]  from intra-chunk kernel
    KC:           torch.Tensor,    # [B, H, NC, C, Dk]  from intra-chunk kernel
    g_cum:        torch.Tensor,    # [B, H, NC, C]       cumsum(g)
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    sequence_length: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    B, H, NC, C, Dk = key.shape
    Dv = V_corrected.shape[-1]
    dev = key.device
    dt  = key.dtype

    causal_mask = torch.triu(
        torch.ones(C, C, dtype=torch.bool, device=dev), diagonal=1
    )  # True above diagonal → will be masked to -inf

    state = (
        torch.zeros(B, H, Dk, Dv, device=dev, dtype=dt)
        if initial_state is None
        else initial_state.to(dtype=dt)
    )
    out = torch.empty(B, H, NC, C, Dv, device=dev, dtype=dt)

    for i in range(NC):
        q_i   = query      [:, :, i]   # [B, H, C, Dk]
        k_i   = key        [:, :, i]   # [B, H, C, Dk]
        v_i   = V_corrected[:, :, i]   # [B, H, C, Dv]
        kc_i  = KC         [:, :, i]   # [B, H, C, Dk]
        gc_i  = g_cum      [:, :, i]   # [B, H, C]

        # Per-chunk decay mask — [B, H, C, C], computed without storing NC copies
        dm_i = ((gc_i.unsqueeze(-1) - gc_i.unsqueeze(-2)).tril().exp()).tril()

        # Intra-chunk causal attention  [B, H, C, C]
        attn_intra = (q_i @ k_i.transpose(-1, -2) * dm_i).masked_fill_(causal_mask, 0)

        # Inter-chunk contribution: state look-up corrected by delta rule
        v_prime   = kc_i @ state                         # [B, H, C, Dv]
        v_new     = v_i - v_prime                        # [B, H, C, Dv]
        attn_inter = (q_i * gc_i[..., None].exp()) @ state  # [B, H, C, Dv]

        out[:, :, i] = attn_inter + attn_intra @ v_new

        # Recurrent state update
        g_last  = gc_i[..., -1, None, None]              # [B, H, 1, 1]
        k_decay = k_i * (g_last.squeeze(-1) - gc_i).exp()[..., None]
        state   = (
            state * g_last.exp()
            + k_decay.transpose(-1, -2) @ v_new
        )

    return out, (state if output_final_state else None)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Public entry-point — drop-in for torch_chunk_gated_delta_rule
# ══════════════════════════════════════════════════════════════════════════════

def fast_chunk_gated_delta_rule(
    query:      torch.Tensor,
    key:        torch.Tensor,
    value:      torch.Tensor,
    g:          torch.Tensor,
    beta:       torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    backend: str = "auto",   # "auto" | "cute" | "triton" | "torch"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Drop-in replacement for torch_chunk_gated_delta_rule.

    backend="auto" picks: CUTE-DSL (B200) > Triton > PyTorch.
    """
    if backend == "auto":
        backend = get_backend()

    # ── fall back to pure PyTorch ──────────────────────────────────────────
    if backend == "torch":
        from qwen3_5_transformers import torch_chunk_gated_delta_rule
        return torch_chunk_gated_delta_rule(
            query, key, value, g, beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    # ── shared pre-processing (same as PyTorch baseline) ──────────────────
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = l2norm(query)
        key   = l2norm(key)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float()
        for x in (query, key, value, beta, g)
    ]
    B, H, T, Dk = key.shape
    Dv = value.shape[-1]

    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        query = F.pad(query, (0, 0, 0, pad))
        key   = F.pad(key,   (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta  = F.pad(beta,  (0, pad))
        g     = F.pad(g,     (0, pad))
    T_pad = T + pad
    NC    = T_pad // chunk_size

    scale   = 1.0 / math.sqrt(Dk)
    query   = query * scale
    v_beta  = value * beta.unsqueeze(-1)
    k_beta  = key   * beta.unsqueeze(-1)

    # Reshape to [B, H, NC, C, D]
    def _chunk(t, D):
        return t.reshape(B, H, NC, chunk_size, D)

    query  = _chunk(query,  Dk)
    key    = _chunk(key,    Dk)
    v_beta = _chunk(v_beta, Dv)
    k_beta = _chunk(k_beta, Dk)
    g      = _chunk(g,      1).squeeze(-1)         # [B, H, NC, C]

    # Cumulative sum of g along the chunk dimension
    g_cum  = g.cumsum(dim=-1)                       # [B, H, NC, C]

    # ── Intra-chunk phase ─────────────────────────────────────────────────
    # Reshape to flat [BHC, C, D] for the kernel
    BHC  = B * H * NC
    K_f  = key  .reshape(BHC, chunk_size, Dk)
    KB_f = k_beta.reshape(BHC, chunk_size, Dk)
    VB_f = v_beta.reshape(BHC, chunk_size, Dv)
    GC_f = g_cum  .reshape(BHC, chunk_size)

    if backend == "cute" and _CUTE_OK:
        V_corr_f, KC_f = cute_intra_chunk_fwd(K_f, KB_f, VB_f, GC_f)
    elif backend == "triton" and _TRITON_OK:
        V_corr_f, KC_f = triton_intra_chunk_fwd(K_f, KB_f, VB_f, GC_f)
    else:
        raise RuntimeError(f"Requested backend '{backend}' is not available.")

    # Un-flatten back to [B, H, NC, C, D]
    V_corr = V_corr_f.reshape(B, H, NC, chunk_size, Dv)
    KC     = KC_f    .reshape(B, H, NC, chunk_size, Dk)

    # For the inter-chunk loop we need the raw key (l2-normed but not k_beta)
    key_chunks = _chunk(key.reshape(B, H, T_pad, Dk), Dk)  # already chunked above

    # ── Inter-chunk phase ─────────────────────────────────────────────────
    core_out, last_state = _inter_chunk_loop(
        query, key_chunks, V_corr, KC, g_cum,
        initial_state, output_final_state, T,
    )

    # Remove padding, reshape, cast back to original dtype
    core_out = core_out.reshape(B, H, T_pad, Dv)[:, :, :T]
    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)

    return core_out, last_state
