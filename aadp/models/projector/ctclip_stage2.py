"""CTCLIPStage2Projector — Stage 2 projector for pre-extracted CT-CLIP features.

Wraps InterSliceAggregator directly (no Stage 1) since CT-CLIP features are
already spatially compressed to (D=24, K=576, C=512) per volume.

Input tensor convention (from CTCLIPFeatureDataset):
    (B, 24, 576, 512)   i.e. (B, D, K, ctclip_dim)

After optional input_proj:
    (B, 24, 576, embed_dim)  → InterSliceAggregator → (B, M, embed_dim)
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from aadp.ablations.attention_conditioned_ctclip_stage2 import (
    AttentionConditionedInterSliceAggregator,
)
from aadp.models.projector.stage2 import InterSliceAggregator

# Fixed CT-CLIP spatial constants
_D = 24    # depth slices
_K = 576   # 24 × 24 spatial tokens per slice


class CTCLIPStage2Projector(nn.Module):
    """Projector that feeds CT-CLIP pre-extracted features into Stage 2.

    Skips Stage 1 entirely — CT-CLIP features are treated as K=576 latents
    per slice, feeding directly into InterSliceAggregator.

    Args:
        ctclip_dim:    Channel dim of CT-CLIP features (default 512).
        embed_dim:     Working dimension inside Stage 2. If equal to
                       ctclip_dim, input_proj is nn.Identity. Default 512.
        num_tokens:    M — number of output tokens for the LLM. Default 64.
        num_heads:     Attention heads in Stage 2. Default 8.
        cond_dim:      Instruction encoder output dimension. Default 2048.
        dropout:       Attention dropout. Default 0.0.
        use_film:      FiLM conditioning in Stage 2, only used when
                       ``conditioning="film"``. Default True.
        max_depth:     Max depth passed to LearnableDepthEnc1D. Default 24.
        conditioning:  ``"film"`` (default) uses InterSliceAggregator's FiLM
                       conditioning; ``"attention"`` uses
                       AttentionConditionedInterSliceAggregator, which
                       replaces FiLM entirely with a second cross-attention
                       operation over the instruction embedding (see
                       aadp/ablations/attention_conditioned_ctclip_stage2.py).
                       Both share the identical top-K sparse + cosine visual
                       cross-attention mechanism.
        device:        Target device. Default "cuda".
    """

    def __init__(
        self,
        ctclip_dim: int = 512,
        embed_dim: int = 512,
        num_tokens: int = 64,
        num_heads: int = 8,
        cond_dim: int = 2048,
        dropout: float = 0.0,
        use_film: bool = True,
        max_depth: int = 24,
        top_k: int = 128,
        conditioning: str = "film",
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        if conditioning not in ("film", "attention"):
            raise ValueError(
                f"conditioning must be 'film' or 'attention', got {conditioning!r}"
            )

        self._ctclip_dim = ctclip_dim
        self._embed_dim = embed_dim
        self._conditioning = conditioning

        if ctclip_dim != embed_dim:
            self.input_proj: nn.Module = nn.Linear(ctclip_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()

        if conditioning == "attention":
            self.stage2: nn.Module = AttentionConditionedInterSliceAggregator(
                embed_dim=embed_dim,
                num_tokens=num_tokens,
                num_heads=num_heads,
                cond_dim=cond_dim,
                dropout=dropout,
                max_depth=max_depth,
                top_k=top_k,
                device=device,
            )
            self._stage2_kwargs: Dict = dict(
                embed_dim=embed_dim,
                num_heads=num_heads,
                cond_dim=cond_dim,
                dropout=dropout,
                max_depth=max_depth,
                top_k=top_k,
                device=str(device),
            )
        else:
            self.stage2 = InterSliceAggregator(
                embed_dim=embed_dim,
                num_tokens=num_tokens,
                num_heads=num_heads,
                cond_dim=cond_dim,
                dropout=dropout,
                use_film=use_film,
                max_depth=max_depth,
                top_k=top_k,
                device=device,
            )
            self._stage2_kwargs = dict(
                embed_dim=embed_dim,
                num_heads=num_heads,
                cond_dim=cond_dim,
                dropout=dropout,
                use_film=use_film,
                max_depth=max_depth,
                top_k=top_k,
                device=str(device),
            )

        self.to(torch.device(device))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_tokens(self) -> int:
        return self.stage2.num_tokens

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, features: torch.Tensor, etext: torch.Tensor) -> torch.Tensor:
        """Project CT-CLIP features to M output tokens.

        Args:
            features: ``(B, 24, 576, ctclip_dim)`` pre-extracted CT-CLIP features.
            etext:    ``(B, cond_dim)`` instruction embedding.

        Returns:
            ``(B, M, embed_dim)`` tokens ready for the LLM bridge.
        """
        # features: (B, D, K, ctclip_dim)
        x = self.input_proj(features)    # (B, D, K, embed_dim)
        out = self.stage2(x, etext)      # (B, M, embed_dim)
        return out

    # ── Budget sweep ──────────────────────────────────────────────────────────

    def rebuild_at_budget(self, M: int) -> None:
        """Replace Stage 2 with a fresh aggregator at token budget M.

        No weights are copied — this is for VTCB budget sweeps where the
        Stage 2 head is re-initialised at each M. Uses whichever aggregator
        class this projector was built with (``conditioning``).

        Args:
            M: New number of LLM output tokens.
        """
        device = next(self.parameters()).device
        kwargs = {**self._stage2_kwargs, "num_tokens": M, "device": str(device)}
        stage2_cls = (
            AttentionConditionedInterSliceAggregator
            if self._conditioning == "attention"
            else InterSliceAggregator
        )
        self.stage2 = stage2_cls(**kwargs)
        self._stage2_kwargs["num_tokens"] = M

    def get_slice_attention(self) -> Optional[torch.Tensor]:
        """Per-slice attention mass ``(B, D)`` from the last forward pass."""
        return self.stage2.get_slice_attention(_D, _K)

    def num_parameters(self) -> Dict[str, int]:
        """Count trainable parameters in input_proj and stage2."""
        ip = sum(p.numel() for p in self.input_proj.parameters() if p.requires_grad)
        s2 = sum(p.numel() for p in self.stage2.parameters() if p.requires_grad)
        return {"input_proj": ip, "stage2": s2, "total": ip + s2}
