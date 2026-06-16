"""AADPProjector — the complete two-stage A-ADP compression pipeline.

Combines IntraSliceDistiller (Stage 1) and InterSliceAggregator (Stage 2)
into a single module that maps ViT patch tokens + an instruction embedding
to M tokens ready for a large language model.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from aadp.models.projector.stage1 import IntraSliceDistiller
from aadp.models.projector.stage2 import InterSliceAggregator


class AADPProjector(nn.Module):
    """Two-stage Anatomy-Aware Dynamic Projector.

    Stage 1 (``IntraSliceDistiller``) compresses each CT slice's N patch
    tokens down to K latents independently using Perceiver cross-attention.

    Stage 2 (``InterSliceAggregator``) aggregates the resulting D×K latents
    into M tokens for the LLM, guided by FiLM-modulated depth queries that
    focus attention on slices relevant to the clinical instruction.

    Args:
        embed_dim:         Hidden dimension C. Must match ViT ``output_dim``.
        num_latents:       K — per-slice latents from Stage 1. Default 32.
        num_tokens:        M — final tokens for the LLM from Stage 2. Default 64.
        num_heads_stage1:  Attention heads for Stage 1. Default 8.
        num_heads_stage2:  Attention heads for Stage 2. Default 8.
        cond_dim:          Instruction embedding dimension.
        dropout:           Dropout applied to both stages. Default 0.0.
        use_film:          If True use FiLMLayer in Stage 2; False for ablation.
                           Default True.
        max_depth:         Max depth passed to Stage 2's depth encoder. Default 512.
        device:            Device to place the module on. Default ``"cuda"``.
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
        use_film: bool = True,
        max_depth: int = 512,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

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
        self.stage2 = InterSliceAggregator(
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            num_heads=num_heads_stage2,
            cond_dim=cond_dim,
            dropout=dropout,
            use_film=use_film,
            max_depth=max_depth,
            device=device,
        )

        # Stored for rebuild_at_budget
        self._stage2_kwargs: Dict = dict(
            embed_dim=embed_dim,
            num_heads=num_heads_stage2,
            cond_dim=cond_dim,
            dropout=dropout,
            use_film=use_film,
            max_depth=max_depth,
            device=str(device),
        )

        # Stored during forward for get_slice_attention()
        self._last_D: Optional[int] = None
        self._last_K: int = num_latents

        self.to(torch.device(device))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_tokens(self) -> int:
        """Number of output tokens M delivered to the LLM."""
        return self._num_tokens

    @property
    def num_latents(self) -> int:
        """Number of per-slice latents K from Stage 1."""
        return self._num_latents

    @property
    def embed_dim(self) -> int:
        """Hidden dimension C."""
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
        """Run the full two-stage A-ADP pipeline.

        Args:
            patch_tokens:     ``(B, D, N, C)`` ViT patch tokens for all slices.
            etext:            ``(B, cond_dim)`` instruction embedding.
            H_patches:        Patch grid height, passed to Stage 1's pos enc.
            W_patches:        Patch grid width, passed to Stage 1's pos enc.
            depth_spacing_mm: Physical slice spacing in mm; passed to Stage 2.

        Returns:
            ``(B, M, C)`` tokens ready for the LLM.
        """
        B, D, N, C = patch_tokens.shape

        # Cache D for get_slice_attention()
        self._last_D = D

        # Step 1 — Stage 1: compress each slice independently
        x = patch_tokens.reshape(B * D, N, C)
        x = self.stage1(x, H_patches, W_patches)   # (B*D, K, C)

        # Step 2 — Stage 2: aggregate across slices with instruction conditioning
        K = x.shape[1]
        x = x.reshape(B, D, K, C)
        out = self.stage2(x, etext, depth_spacing_mm)  # (B, M, C)

        return out

    # ── Additional methods ────────────────────────────────────────────────────

    def get_slice_attention(self) -> torch.Tensor:
        """Per-slice attention mass ``(B, D)`` from the last forward pass.

        Delegates to ``self.stage2.get_slice_attention(D, K)``.

        Raises:
            RuntimeError: If called before any forward pass.
        """
        if self._last_D is None:
            raise RuntimeError(
                "get_slice_attention() called before any forward pass. "
                "Run forward() first."
            )
        return self.stage2.get_slice_attention(self._last_D, self._num_latents)

    def rebuild_at_budget(self, M: int) -> None:
        """Rebuild Stage 2 with a new token budget M; Stage 1 weights are preserved."""
        self.stage2 = InterSliceAggregator(num_tokens=M, **self._stage2_kwargs)
        self._num_tokens = M

    def num_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per stage and in total.

        Returns:
            Dict with keys ``"stage1"``, ``"stage2"``, and ``"total"``.
        """

        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        s1 = _count(self.stage1)
        s2 = _count(self.stage2)
        return {"stage1": s1, "stage2": s2, "total": s1 + s2}
