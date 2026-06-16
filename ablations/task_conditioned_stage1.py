"""A1 Ablation: Task-Conditioned Stage 1 (intra-slice FiLM modulation).

Standard A-ADP conditions on the clinical instruction only in Stage 2
(InterSliceAggregator), leaving Stage 1 task-agnostic.  This ablation
injects FiLM conditioning into Stage 1 as well, modulating the K query
latents before cross-attention so each slice compresses tokens in a
task-aware manner from the very first stage.

``TaskConditionedIntraSliceDistiller`` is a drop-in replacement for
``IntraSliceDistiller`` with one additional ``etext`` input that shapes
the queries via a FiLMLayer initialised to identity (γ=1, β=0), so that
at training start the behaviour is identical to the task-agnostic baseline.

``TaskConditionedAADP`` is a drop-in replacement for ``AADPProjector``
that uses ``TaskConditionedIntraSliceDistiller`` for Stage 1.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from aadp.models.film import FiLMLayer
from aadp.models.projector.pos_encoding import SinusoidalPosEnc2D
from aadp.models.projector.stage2 import InterSliceAggregator


class TaskConditionedIntraSliceDistiller(nn.Module):
    """FiLM-modulated Stage 1: task-conditioned intra-slice Perceiver.

    Identical to ``IntraSliceDistiller`` except that the query latents are
    FiLM-modulated with the instruction embedding ``etext`` before the
    cross-attention step.  At initialisation the FiLMLayer produces identity
    transforms (γ=1, β=0), so the behaviour matches the task-agnostic
    baseline at the start of training.

    Args:
        embed_dim:   Token embedding dimension C.
        num_latents: K — per-slice query latents. Default 32.
        num_heads:   Attention heads. Default 8.
        dropout:     Dropout probability. Default 0.0.
        cond_dim:    Instruction embedding dimension. Default 768.
        device:      Target device. Default ``"cuda"``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_latents: int = 32,
        num_heads: int = 8,
        dropout: float = 0.0,
        cond_dim: int = 768,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self._embed_dim = embed_dim
        self._num_latents = num_latents

        self.latents = nn.Parameter(torch.empty(num_latents, embed_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.pos_enc = SinusoidalPosEnc2D(embed_dim, device=device)

        # FiLM layer: modulates query latents with instruction embedding.
        # Initialised to identity (γ=1, β=0) so Stage 1 starts task-agnostic.
        self.film = FiLMLayer(cond_dim=cond_dim, target_dim=embed_dim, device=device)

        self.to(torch.device(device))

    @property
    def num_latents(self) -> int:
        return self._num_latents

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(
        self,
        x: torch.Tensor,
        H_patches: int,
        W_patches: int,
        etext: torch.Tensor,
        B: int,
        D: int,
    ) -> torch.Tensor:
        """Compress slice patch tokens with task-conditioned query latents.

        Args:
            x:         ``(B*D, N, C)`` patch tokens from SliceEncoder.
            H_patches: Patch grid height.
            W_patches: Patch grid width.
            etext:     ``(B, cond_dim)`` instruction embedding.
            B:         Batch size (before D was flattened into batch dim).
            D:         Number of depth slices.

        Returns:
            ``(B*D, K, C)`` — K task-conditioned latents per slice.
        """
        B_D = x.shape[0]

        pos = self.pos_enc(H_patches, W_patches)   # (N, C)
        x = x + pos.unsqueeze(0)                   # (B_D, N, C)

        kv = self.norm_kv(x)

        q = self.latents.unsqueeze(0).expand(B_D, -1, -1)  # (B_D, K, C)
        q = self.norm_q(q)

        # Broadcast etext across slices: (B, cond_dim) → (B*D, cond_dim)
        etext_exp = etext.unsqueeze(1).expand(B, D, -1).reshape(B_D, -1)
        q = self.film(q, etext_exp)                         # (B_D, K, C)

        out, _ = self.cross_attn(q, kv, kv, need_weights=False)  # (B_D, K, C)
        return out


class TaskConditionedAADP(nn.Module):
    """A-ADP variant with task conditioning applied to both Stage 1 and Stage 2.

    Replaces ``IntraSliceDistiller`` with
    ``TaskConditionedIntraSliceDistiller`` so the clinical instruction
    guides per-slice compression as well as slice selection in Stage 2.

    Interface is identical to ``AADPProjector``.

    Args:
        embed_dim:         Hidden dimension C.
        num_latents:       K — per-slice latents. Default 32.
        num_tokens:        M — LLM tokens. Default 64.
        num_heads_stage1:  Stage 1 attention heads. Default 8.
        num_heads_stage2:  Stage 2 attention heads. Default 8.
        cond_dim:          Instruction embedding dimension. Default 768.
        dropout:           Dropout. Default 0.0.
        use_film:          FiLM in Stage 2. Default True.
        max_depth:         Max depth for Stage 2. Default 512.
        device:            Target device. Default ``"cuda"``.
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

        self.stage1 = TaskConditionedIntraSliceDistiller(
            embed_dim=embed_dim,
            num_latents=num_latents,
            num_heads=num_heads_stage1,
            dropout=dropout,
            cond_dim=cond_dim,
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

        self._last_D: Optional[int] = None
        self._last_K: int = num_latents

        self.to(torch.device(device))

    @property
    def num_tokens(self) -> int:
        """Number of output tokens M delivered to the LLM."""
        return self._num_tokens

    @property
    def num_latents(self) -> int:
        return self._num_latents

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(
        self,
        patch_tokens: torch.Tensor,
        etext: torch.Tensor,
        H_patches: int,
        W_patches: int,
        depth_spacing_mm: Optional[float] = None,
    ) -> torch.Tensor:
        """Run Task-Conditioned A-ADP pipeline.

        Args:
            patch_tokens:     ``(B, D, N, C)`` ViT patch tokens.
            etext:            ``(B, cond_dim)`` instruction embedding.
            H_patches:        Patch grid height.
            W_patches:        Patch grid width.
            depth_spacing_mm: Physical slice spacing in mm.

        Returns:
            ``(B, M, C)`` tokens for the LLM.
        """
        B, D, N, C = patch_tokens.shape
        self._last_D = D

        x = patch_tokens.reshape(B * D, N, C)
        x = self.stage1(x, H_patches, W_patches, etext, B, D)  # (B*D, K, C)

        K = x.shape[1]
        x = x.reshape(B, D, K, C)
        return self.stage2(x, etext, depth_spacing_mm)          # (B, M, C)

    def get_slice_attention(self) -> torch.Tensor:
        """Per-slice attention mass ``(B, D)`` from the last forward pass."""
        if self._last_D is None:
            raise RuntimeError(
                "get_slice_attention() called before any forward pass. "
                "Run forward() first."
            )
        return self.stage2.get_slice_attention(self._last_D, self._num_latents)

    def rebuild_at_budget(self, M: int) -> None:
        """Rebuild Stage 2 with token budget M; Stage 1 weights are preserved."""
        self.stage2 = InterSliceAggregator(num_tokens=M, **self._stage2_kwargs)
        self._num_tokens = M

    def num_parameters(self) -> Dict[str, int]:
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        s1 = _count(self.stage1)
        s2 = _count(self.stage2)
        return {"stage1": s1, "stage2": s2, "total": s1 + s2}
