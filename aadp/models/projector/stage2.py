"""Stage 2 of the A-ADP projector: InterSliceAggregator.

Aggregates D×K slice latents (from Stage 1) into a fixed M-token budget using
FiLM-conditioned depth queries and cross-attention. Depth positional encodings
make the model aware of slice position; FiLM modulation steers which slices
the M queries attend to based on the clinical instruction.
"""

import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from aadp.models.film import FiLMLayer, NullFiLMLayer
from aadp.models.projector.pos_encoding import LearnableDepthEnc1D


class InterSliceAggregator(nn.Module):
    """Cross-attention aggregator that collapses D×K slice latents to M tokens.

    Learnable depth queries Qd (shape M×C) are first FiLM-modulated by the
    instruction embedding ``etext``, then attend over the full D×K latent
    sequence.  Attention weights (stored after each forward pass) expose
    per-slice attention mass for metric computation.

    Args:
        embed_dim:   Hidden dimension C. Must match Stage 1's output dim.
        num_tokens:  M — number of output tokens consumed by the LLM. Default 64.
        num_heads:   Attention heads. Default 8.
        cond_dim:    Dimensionality of ``etext`` from InstructionEncoder.
        dropout:     Attention dropout. Default 0.0.
        use_film:    If True use FiLMLayer; if False use NullFiLMLayer (ablation).
                     Default True.
        max_depth:   Maximum depth passed to LearnableDepthEnc1D. Default 512.
        top_k:       Number of visual positions each query attends to (sparse
                     top-K cross-attention), out of N = D*K total positions at
                     forward time. Default 128 — chosen for this project's
                     actual CT-CLIP configuration (D=24, K=576 → N=13,824;
                     min(128, N//4)=128). Diagnostic finding (see
                     DIAGNOSTIC_REPORT.md, "Test 3 Rerun"): CT-CLIP features
                     are highly similar across spatial positions (raw pairwise
                     cosine ~0.87), so full softmax over all N keys produces a
                     near-uniform attention map regardless of query — the
                     cross-attention output collapses toward mean(V) for every
                     query, erasing FiLM's instruction-specific query
                     modulation before it can matter. Restricting softmax to
                     each query's top-K positions forces it to commit to a
                     small, query-specific subset instead of averaging
                     everything away. If N <= top_k at forward time (e.g. a
                     different D/K than this default was tuned for), falls
                     back to standard full-softmax attention for that pass —
                     mathematically identical to top-K when top_k >= N, and
                     avoids ever masking every position of a row to -inf.
        device:      Device to place the module on. Default ``"cuda"``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 64,
        num_heads: int = 8,
        cond_dim: int = 768,
        dropout: float = 0.0,
        use_film: bool = True,
        max_depth: int = 512,
        top_k: int = 128,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self._embed_dim = embed_dim
        self._num_tokens = num_tokens
        self._num_heads = num_heads
        self.top_k = top_k

        # Cosine attention fix (DIAGNOSTIC_REPORT.md, "Test 3 Rerun 2"): top-K
        # sparse attention alone didn't help because QK^T score ranking was
        # dominated by K's own magnitude structure (CT-CLIP features are
        # homogeneous in direction but not in norm), not by query-key
        # directional alignment — so the same "loud" positions won top-K
        # regardless of instruction. L2-normalizing both Q and K before the
        # dot product makes scores pure cosine similarity, restoring
        # direction-dependent (hence instruction-dependent) ranking. A
        # learnable temperature recovers the scale control that normalizing
        # away |Q| and |K| removes.
        self.attn_temperature = nn.Parameter(torch.tensor(0.07))

        # V-FiLM fix (DIAGNOSTIC_REPORT.md, "Test 3 Rerun 3"): cosine
        # attention fixed WHICH positions get selected (top-K overlap
        # 95%→75%) but not the fact that the VALUES being aggregated are
        # themselves homogeneous (CT-CLIP raw feature cosine ~0.87) — a
        # ~25%-different weighted combination of very-similar vectors still
        # lands close in vector space regardless of attention weights.
        # gamma_v_proj/beta_v_proj let the instruction directly modulate
        # which feature dimensions of V are amplified/shifted, independent
        # of which positions attention selects. Zero-initialized so
        # gamma_v=1, beta_v=0 at start of training — identical to no V-FiLM
        # until the model learns otherwise.
        self.gamma_v_proj = nn.Linear(cond_dim, embed_dim)
        self.beta_v_proj = nn.Linear(cond_dim, embed_dim)
        nn.init.zeros_(self.gamma_v_proj.weight)
        nn.init.zeros_(self.gamma_v_proj.bias)   # gamma_v = 1.0 + 0 = 1.0 at init
        nn.init.zeros_(self.beta_v_proj.weight)
        nn.init.zeros_(self.beta_v_proj.bias)    # beta_v = 0 at init

        # Learnable depth queries: (M, C), shared across volumes in a batch
        self.depth_queries = nn.Parameter(torch.empty(num_tokens, embed_dim))
        nn.init.trunc_normal_(self.depth_queries, std=0.02)

        film_cls = FiLMLayer if use_film else NullFiLMLayer
        self.film: Union[FiLMLayer, NullFiLMLayer] = film_cls(
            cond_dim, embed_dim, device=device
        )

        self.depth_pos_enc = LearnableDepthEnc1D(max_depth, embed_dim, device=device)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # No norm_q: diagnostics (scripts/diagnose_mode_collapse.py) found its
        # weight essentially untrained (mean~0.996, std~0.008 after training)
        # yet still the smallest-gradient submodule of the aggregator (grad
        # norm ~0.0009 vs cross_attn's ~0.28), i.e. not collapsed but a
        # bottleneck on query variation reaching cross_attn. Queries entering
        # cross-attention unnormalized is standard in Q-Former-style designs.
        self.norm_kv = nn.LayerNorm(embed_dim)

        # Attention weights buffer for visualisation / recall@k evaluation
        self._last_attn_weights: Optional[torch.Tensor] = None

        self.to(torch.device(device))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_tokens(self) -> int:
        """Number of output tokens M delivered to the LLM."""
        return self._num_tokens

    @property
    def embed_dim(self) -> int:
        """Hidden dimension C."""
        return self._embed_dim

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        slice_latents: torch.Tensor,
        etext: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate slice latents into M output tokens.

        Args:
            slice_latents: ``(B, D, K, C)`` — Stage 1 outputs reshaped from
                           ``(B*D, K, C)``.
            etext:         ``(B, cond_dim)`` — instruction embedding.

        Returns:
            ``(B, M, C)`` — M aggregated tokens ready for the LLM.
        """
        B, D, K, C = slice_latents.shape

        # Step 1-2: depth positional encoding (D, C)
        depth_pe = self.depth_pos_enc(D)

        # Step 3: broadcast (D, C) over B and K → add same depth emb to every
        # one of the K latents in each slice
        slice_latents = slice_latents + depth_pe.unsqueeze(0).unsqueeze(2)
        # shape: (B, D, K, C)

        # Step 4: flatten D×K into sequence length
        kv = slice_latents.reshape(B, D * K, C)    # (B, D*K, C)

        # Step 5: expand depth queries to batch
        q = self.depth_queries.unsqueeze(0).expand(B, -1, -1)  # (B, M, C)

        # Step 6: FiLM-modulate queries with instruction embedding
        q = self.film(q, etext)                    # (B, M, C)

        # Step 7: pre-norm (kv only — see __init__ comment on norm_q removal)
        kv = self.norm_kv(kv)

        # Step 8: top-K sparse cross-attention (see __init__'s top_k docstring
        # for why full softmax attention collapsed FiLM's instruction signal).
        # Manual QKV split reusing nn.MultiheadAttention's own parameters —
        # not its forward() — so a top-K mask can be applied to the scores
        # before softmax (nn.MultiheadAttention has no clean hook for this),
        # while checkpoint state_dict keys/shapes stay byte-identical to the
        # previous full-attention version.
        N = kv.shape[1]
        head_dim = self._embed_dim // self._num_heads

        in_proj_weight = self.cross_attn.in_proj_weight   # (3*C, C)
        in_proj_bias = self.cross_attn.in_proj_bias       # (3*C,)
        Wq, Wk, Wv = in_proj_weight.chunk(3, dim=0)
        bq, bk, bv = in_proj_bias.chunk(3, dim=0)

        Qp = F.linear(q, Wq, bq)    # (B, M, C)
        Kp = F.linear(kv, Wk, bk)   # (B, N, C)
        Vp = F.linear(kv, Wv, bv)   # (B, N, C)

        # V-FiLM: instruction-conditioned modulation of the value content
        # itself (which feature dimensions matter), independent of Q/K's
        # attention-weight conditioning. Applied pre-head-split so
        # gamma_v/beta_v (B, C) broadcast over the full N sequence in one
        # shot, matching Qp/Kp/Vp's shape convention.
        gamma_v = 1.0 + self.gamma_v_proj(etext)   # (B, C)
        beta_v = self.beta_v_proj(etext)            # (B, C)
        Vp = gamma_v.unsqueeze(1) * Vp + beta_v.unsqueeze(1)   # (B, N, C)

        if os.environ.get("ICTC_DEBUG_ATTN"):
            print(f"[DEBUG] gamma_v mean: {gamma_v[0].mean():.4f}, beta_v mean: {beta_v[0].mean():.4f}")

        Qh = Qp.view(B, self._num_tokens, self._num_heads, head_dim).transpose(1, 2)  # (B,H,M,hd)
        Kh = Kp.view(B, N, self._num_heads, head_dim).transpose(1, 2)                 # (B,H,N,hd)
        Vh = Vp.view(B, N, self._num_heads, head_dim).transpose(1, 2)                 # (B,H,N,hd)

        # Cosine attention: normalize Q and K (per head, last-dim=head_dim)
        # so scores reflect directional alignment only, not |K|'s magnitude
        # structure. V is left unnormalized — it still carries real feature
        # content into the weighted sum. +1e-8 guards the zero-vector edge
        # case before normalize (F.normalize's own eps=1e-12 already covers
        # this, but the guard is added explicitly per spec for clarity).
        Qh = F.normalize(Qh + 1e-8, dim=-1)
        Kh = F.normalize(Kh, dim=-1)

        tau = self.attn_temperature.clamp(min=1e-4)
        scores = (Qh @ Kh.transpose(-2, -1)) / tau   # (B,H,M,N) — cosine sim / temperature

        if os.environ.get("ICTC_DEBUG_ATTN"):
            print(f"[DEBUG] Q norm after normalize: {Qh.norm(dim=-1).mean():.4f}")
            print(f"[DEBUG] K norm after normalize: {Kh.norm(dim=-1).mean():.4f}")
            print(f"[DEBUG] score std: {scores.std():.4f}, range: [{scores.min():.1f}, {scores.max():.1f}]")

        # NaN guard: top_k must never exceed N (masking every position of a
        # row to -inf would make softmax produce NaN). When top_k >= N,
        # attending over all N positions IS standard full-softmax attention,
        # so this fallback is exact, not an approximation.
        effective_top_k = min(self.top_k, N)
        if effective_top_k < N:
            topk_idx = scores.topk(effective_top_k, dim=-1).indices        # (B,H,M,k)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(-1, topk_idx, 0.0)
            scores = scores + mask                                        # -inf for non-top-K

        attn_weights = F.softmax(scores, dim=-1)   # (B,H,M,N) — sparse when top_k < N
        attn_weights = F.dropout(attn_weights, p=self.cross_attn.dropout, training=self.training)

        out_h = attn_weights @ Vh                                    # (B,H,M,hd)
        out = out_h.transpose(1, 2).reshape(B, self._num_tokens, self._embed_dim)  # (B,M,C)
        out = F.linear(out, self.cross_attn.out_proj.weight, self.cross_attn.out_proj.bias)

        # Average over heads to match the previous nn.MultiheadAttention
        # contract (average_attn_weights=True default) — get_slice_attention()
        # and tests/metrics depending on (B, M, D*K) shape are unaffected.
        attn_weights_avg = attn_weights.mean(dim=1)   # (B, M, N) == (B, M, D*K)

        # Store weights detached (no grad accumulation in buffer)
        self._last_attn_weights = attn_weights_avg.detach()

        return out

    # ── Slice attention helper ────────────────────────────────────────────────

    def get_slice_attention(self, D: int, K: int) -> torch.Tensor:
        """Return per-slice attention mass averaged over M and K.

        Must be called after at least one forward pass.

        Args:
            D: Number of depth slices.
            K: Number of latents per slice (from Stage 1).

        Returns:
            ``(B, D)`` tensor of per-slice attention mass.

        Raises:
            RuntimeError: If called before any forward pass.
        """
        if self._last_attn_weights is None:
            raise RuntimeError(
                "get_slice_attention() called before any forward pass. "
                "Run forward() first."
            )
        B, M, _ = self._last_attn_weights.shape
        # Reshape to (B, M, D, K) then average over M and K → (B, D)
        weights = self._last_attn_weights.reshape(B, M, D, K)
        return weights.mean(dim=(1, 3))             # (B, D)
