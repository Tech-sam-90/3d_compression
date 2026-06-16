"""AttentionConditionedStage2 and AttentionConditionedAADP ablations.

These are drop-in replacements for ``InterSliceAggregator`` (Stage 2) and
``AADPProjector`` respectively.  The ablation replaces FiLM's lightweight
channel-wise affine conditioning with a second cross-attention operation that
attends the depth queries over the projected instruction embedding.

Purpose:
    Directly test whether FiLM's O(C) projection is sufficient or whether the
    heavier O(M × L_text × C) cross-attention conditioning yields measurably
    better recall@k and dice_overlap on the VTCB benchmark.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from aadp.models.projector.pos_encoding import LearnableDepthEnc1D
from aadp.models.projector.stage1 import IntraSliceDistiller


class AttentionConditionedStage2(nn.Module):
    """Stage 2 variant: depth queries are conditioned via cross-attention over etext.

    Unlike ``InterSliceAggregator`` which uses FiLM (a single linear projection
    per channel), this module attends each depth query over a projected
    representation of the instruction embedding.  This is a strictly more
    expressive conditioning mechanism at the cost of additional compute and
    parameters.

    Args:
        embed_dim:  Hidden dimension C.
        num_tokens: M — number of output tokens. Default 64.
        num_heads:  Attention heads for both cross-attention operations.
                    Default 8.
        cond_dim:   Dimensionality of ``etext`` from InstructionEncoder.
        dropout:    Attention dropout. Default 0.0.
        max_depth:  Max depth for the learnable depth encoder. Default 512.
        device:     Target device. Default ``"cuda"``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 64,
        num_heads: int = 8,
        cond_dim: int = 768,
        dropout: float = 0.0,
        max_depth: int = 512,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self._embed_dim = embed_dim
        self._num_tokens = num_tokens

        # Learnable depth queries (M, C)
        self.depth_queries = nn.Parameter(torch.empty(num_tokens, embed_dim))
        nn.init.trunc_normal_(self.depth_queries, std=0.02)

        # Project etext from cond_dim → embed_dim for use as K/V in cond_cross_attn
        self.text_proj = nn.Linear(cond_dim, embed_dim)

        # Conditioning: depth queries attend over the projected instruction token
        self.cond_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Visual: conditioned queries attend over the D×K slice latent sequence
        self.visual_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_text = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.depth_pos_enc = LearnableDepthEnc1D(max_depth, embed_dim, device=device)

        self._last_attn_weights: Optional[torch.Tensor] = None

        self.to(torch.device(device))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        slice_latents: torch.Tensor,
        etext: torch.Tensor,
        depth_spacing_mm: Optional[float] = None,
    ) -> torch.Tensor:
        """Aggregate slice latents into M tokens via attention-conditioned queries.

        Args:
            slice_latents:    ``(B, D, K, C)`` Stage 1 outputs.
            etext:            ``(B, cond_dim)`` instruction embedding.
            depth_spacing_mm: Physical slice spacing in mm; passed to depth enc.

        Returns:
            ``(B, M, C)`` aggregated tokens.
        """
        B, D, K, C = slice_latents.shape

        # Step 1: depth positional encoding → add to slice latents
        if depth_spacing_mm is not None:
            depth_pe = self.depth_pos_enc.with_spacing(D, depth_spacing_mm)
        else:
            depth_pe = self.depth_pos_enc(D)
        slice_latents = slice_latents + depth_pe.unsqueeze(0).unsqueeze(2)

        # Step 2: flatten D×K into a single sequence
        kv = slice_latents.reshape(B, D * K, C)

        # Step 3: expand depth queries to batch
        q = self.depth_queries.unsqueeze(0).expand(B, -1, -1)  # (B, M, C)

        # Step 4: condition queries via cross-attention over projected etext
        text_kv = self.text_proj(etext).unsqueeze(1)   # (B, 1, C)
        text_kv = self.norm_text(text_kv)
        q = self.norm_q(q)
        q_cond, _ = self.cond_cross_attn(
            q, text_kv, text_kv, need_weights=False
        )                                               # (B, M, C)

        # Step 5-6: attend conditioned queries over visual tokens
        kv = self.norm_kv(kv)
        out, attn_weights = self.visual_cross_attn(
            q_cond, kv, kv, need_weights=True
        )                                               # (B, M, C), (B, M, D*K)

        # Step 7: store for metric computation
        self._last_attn_weights = attn_weights.detach()

        return out

    # ── Slice attention helper ─────────────────────────────────────────────────

    def get_slice_attention(self, D: int, K: int) -> torch.Tensor:
        """Per-slice attention mass ``(B, D)`` averaged over M and K queries.

        Raises:
            RuntimeError: If called before any forward pass.
        """
        if self._last_attn_weights is None:
            raise RuntimeError(
                "get_slice_attention() called before any forward pass."
            )
        B, M, _ = self._last_attn_weights.shape
        weights = self._last_attn_weights.reshape(B, M, D, K)
        return weights.mean(dim=(1, 3))  # (B, D)


# ── Drop-in AADPProjector replacement ────────────────────────────────────────


class AttentionConditionedAADP(nn.Module):
    """Full A-ADP pipeline with attention-conditioned Stage 2 (ablation).

    Drop-in replacement for ``AADPProjector``.  Stage 1 is identical;
    Stage 2 replaces FiLM with cross-attention conditioning.

    Constructor signature is identical to ``AADPProjector``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_latents: int = 32,
        num_tokens: int = 64,
        num_heads_stage1: int = 8,
        num_heads_stage2: int = 8,
        cond_dim: int = 768,
        dropout: float = 0.0,
        use_film: bool = True,   # accepted for signature parity; ignored here
        max_depth: int = 512,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()
        _ = use_film  # this ablation always uses cross-attention conditioning

        self._embed_dim = embed_dim
        self._num_latents = num_latents
        self._num_tokens = num_tokens

        self.stage1 = IntraSliceDistiller(
            embed_dim=embed_dim,
            num_latents=num_latents,
            num_heads=num_heads_stage1,
            dropout=dropout,
            device=device,
        )
        self.stage2 = AttentionConditionedStage2(
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            num_heads=num_heads_stage2,
            cond_dim=cond_dim,
            dropout=dropout,
            max_depth=max_depth,
            device=device,
        )

        self._last_D: Optional[int] = None

        self.to(torch.device(device))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def num_latents(self) -> int:
        return self._num_latents

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        patch_tokens: torch.Tensor,
        etext: torch.Tensor,
        H_patches: int,
        W_patches: int,
        depth_spacing_mm: Optional[float] = None,
    ) -> torch.Tensor:
        """Run the two-stage pipeline with attention-conditioned Stage 2.

        Args:
            patch_tokens: ``(B, D, N, C)`` ViT patch tokens.
            etext:        ``(B, cond_dim)`` instruction embedding.
            H_patches:    Patch grid height.
            W_patches:    Patch grid width.
            depth_spacing_mm: Physical slice spacing in mm.

        Returns:
            ``(B, M, C)`` tokens.
        """
        B, D, N, C = patch_tokens.shape
        self._last_D = D

        x = patch_tokens.reshape(B * D, N, C)
        x = self.stage1(x, H_patches, W_patches)   # (B*D, K, C)

        K = x.shape[1]
        x = x.reshape(B, D, K, C)
        return self.stage2(x, etext, depth_spacing_mm)  # (B, M, C)

    # ── Additional methods ─────────────────────────────────────────────────────

    def get_slice_attention(self) -> torch.Tensor:
        """Delegate to Stage 2's attention extractor.

        Raises:
            RuntimeError: If called before any forward pass.
        """
        if self._last_D is None:
            raise RuntimeError(
                "get_slice_attention() called before any forward pass."
            )
        return self.stage2.get_slice_attention(self._last_D, self._num_latents)

    def num_parameters(self) -> Dict[str, int]:
        """Trainable parameter counts by stage."""

        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        s1 = _count(self.stage1)
        s2 = _count(self.stage2)
        return {"stage1": s1, "stage2": s2, "total": s1 + s2}
