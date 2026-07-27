"""AttentionConditionedInterSliceAggregator — CT-CLIP Stage 2 ablation.

Drop-in replacement for ``InterSliceAggregator``
(aadp/models/projector/stage2.py) that replaces FiLM conditioning —
both the query modulation (Q-FiLM) and the value modulation (V-FiLM) added
this session — with a second cross-attention operation, mirroring
``AttentionConditionedStage2`` (aadp/ablations/attention_conditioned_stage2.py):
depth queries first attend over a single projected instruction token, then
the resulting instruction-conditioned queries attend over the visual
D*K sequence.

The visual cross-attention itself (manual QKV split + cosine-normalized
scores + top-K sparse masking) is left IDENTICAL to InterSliceAggregator's
already-validated mechanism (see docs/DIAGNOSTIC_REPORT.md, "Test 3 Rerun 2")
— that fix addresses a separate collapse cause (CT-CLIP features being
highly homogeneous across spatial positions, ~0.87 raw cosine, so full
softmax averages toward mean(V) regardless of query) that is orthogonal to
the FiLM-vs-attention conditioning question this ablation tests. Reusing it
here means a training run isolates the conditioning-mechanism variable
instead of reintroducing an already-diagnosed, unrelated failure mode.
"""

import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from aadp.models.projector.pos_encoding import LearnableDepthEnc1D


class AttentionConditionedInterSliceAggregator(nn.Module):
    """Cross-attention aggregator with attention-based (not FiLM) conditioning.

    Args:
        embed_dim:   Hidden dimension C. Must match Stage 1's output dim.
        num_tokens:  M — number of output tokens consumed by the LLM. Default 64.
        num_heads:   Attention heads (shared by both cross-attention ops). Default 8.
        cond_dim:    Dimensionality of ``etext`` from InstructionEncoder.
        dropout:     Attention dropout. Default 0.0.
        max_depth:   Maximum depth passed to LearnableDepthEnc1D. Default 512.
        top_k:       Visual cross-attention top-K, identical semantics to
                     InterSliceAggregator's ``top_k``. Default 128.
        device:      Device to place the module on. Default ``"cuda"``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 64,
        num_heads: int = 8,
        cond_dim: int = 768,
        dropout: float = 0.0,
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

        # Visual cross-attention scale — identical role to InterSliceAggregator's
        # attn_temperature (cosine-attention fix).
        self.attn_temperature = nn.Parameter(torch.tensor(0.07))

        # Learnable depth queries: (M, C), shared across volumes in a batch
        self.depth_queries = nn.Parameter(torch.empty(num_tokens, embed_dim))
        nn.init.trunc_normal_(self.depth_queries, std=0.02)

        # Attention-based conditioning (replaces FiLM entirely): depth
        # queries attend over a single projected instruction token before
        # attending over the visual sequence. Mirrors
        # AttentionConditionedStage2's cond_cross_attn design.
        self.text_proj = nn.Linear(cond_dim, embed_dim)
        self.cond_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_text = nn.LayerNorm(embed_dim)

        self.depth_pos_enc = LearnableDepthEnc1D(max_depth, embed_dim, device=device)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
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

        depth_pe = self.depth_pos_enc(D)
        slice_latents = slice_latents + depth_pe.unsqueeze(0).unsqueeze(2)
        kv = slice_latents.reshape(B, D * K, C)    # (B, D*K, C)

        q = self.depth_queries.unsqueeze(0).expand(B, -1, -1)  # (B, M, C)
        kv = self.norm_kv(kv)

        # Attention-based conditioning (replaces Q-FiLM/V-FiLM): queries
        # attend over the single projected instruction token.
        text_kv = self.text_proj(etext).unsqueeze(1)   # (B, 1, C)
        text_kv = self.norm_text(text_kv)
        q_n = self.norm_q(q)
        q, _ = self.cond_cross_attn(q_n, text_kv, text_kv, need_weights=False)  # (B, M, C)

        # Visual cross-attention: manual QKV split reusing nn.MultiheadAttention's
        # own parameters, identical mechanism to InterSliceAggregator (top-K
        # sparse + cosine-normalized scores) — see module docstring.
        N = kv.shape[1]
        head_dim = self._embed_dim // self._num_heads

        in_proj_weight = self.cross_attn.in_proj_weight   # (3*C, C)
        in_proj_bias = self.cross_attn.in_proj_bias       # (3*C,)
        Wq, Wk, Wv = in_proj_weight.chunk(3, dim=0)
        bq, bk, bv = in_proj_bias.chunk(3, dim=0)

        Qp = F.linear(q, Wq, bq)    # (B, M, C)
        Kp = F.linear(kv, Wk, bk)   # (B, N, C)
        Vp = F.linear(kv, Wv, bv)   # (B, N, C)

        Qh = Qp.view(B, self._num_tokens, self._num_heads, head_dim).transpose(1, 2)  # (B,H,M,hd)
        Kh = Kp.view(B, N, self._num_heads, head_dim).transpose(1, 2)                 # (B,H,N,hd)
        Vh = Vp.view(B, N, self._num_heads, head_dim).transpose(1, 2)                 # (B,H,N,hd)

        Qh = F.normalize(Qh + 1e-8, dim=-1)
        Kh = F.normalize(Kh, dim=-1)

        tau = self.attn_temperature.clamp(min=1e-4)
        scores = (Qh @ Kh.transpose(-2, -1)) / tau   # (B,H,M,N)

        if os.environ.get("ICTC_DEBUG_ATTN"):
            print(f"[DEBUG] Q norm after normalize: {Qh.norm(dim=-1).mean():.4f}")
            print(f"[DEBUG] K norm after normalize: {Kh.norm(dim=-1).mean():.4f}")
            print(f"[DEBUG] score std: {scores.std():.4f}, range: [{scores.min():.1f}, {scores.max():.1f}]")

        effective_top_k = min(self.top_k, N)
        if effective_top_k < N:
            topk_idx = scores.topk(effective_top_k, dim=-1).indices        # (B,H,M,k)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(-1, topk_idx, 0.0)
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)   # (B,H,M,N)
        attn_weights = F.dropout(attn_weights, p=self.cross_attn.dropout, training=self.training)

        out_h = attn_weights @ Vh                                    # (B,H,M,hd)
        out = out_h.transpose(1, 2).reshape(B, self._num_tokens, self._embed_dim)  # (B,M,C)
        out = F.linear(out, self.cross_attn.out_proj.weight, self.cross_attn.out_proj.bias)

        attn_weights_avg = attn_weights.mean(dim=1)   # (B, M, N) == (B, M, D*K)
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
        weights = self._last_attn_weights.reshape(B, M, D, K)
        return weights.mean(dim=(1, 3))             # (B, D)
