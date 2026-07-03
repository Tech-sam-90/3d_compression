"""PerceiverProjector — single-stage isotropic Perceiver resampler baseline.

Replicates the RadFM / M3D projector design used in prior medical VLMs.

**Known limitations vs. A-ADP (documented by design):**

1. Geometry-blind: the D (depth) and N (patch) axes are flattened together
   before attention.  The model has no way to reason about which tokens come
   from which slice or where those slices sit in anatomical space.

2. Task-blind: the instruction embedding ``etext`` is accepted for interface
   compatibility but is completely ignored.  Every instruction receives the
   same compressed representation — there is no mechanism to concentrate
   tokens on clinically relevant regions.

These are the two failure modes A-ADP is designed to address.
"""

from typing import Optional, Union

import torch
import torch.nn as nn


class PerceiverProjector(nn.Module):
    """Single-stage Perceiver resampler — geometry-blind, task-blind.

    Accepts the same ``(B, D, N, C)`` input and ``(B, M, C)`` output
    contract as ``AADPProjector`` so it can be swapped in as a drop-in
    baseline without touching the training loop or evaluation harness.

    Args:
        embed_dim:  Hidden dimension C.
        num_tokens: M — number of output tokens. Default 64.
        num_heads:  Attention heads. Default 8.
        dropout:    Attention dropout. Default 0.0.
        device:     Target device. Default ``"cuda"``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
        device: Union[torch.device, str] = "cuda",
        **kwargs,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self._embed_dim = embed_dim
        self._num_tokens = num_tokens

        self.latents = nn.Parameter(torch.empty(num_tokens, embed_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

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
        patch_tokens: torch.Tensor,
        etext: torch.Tensor,
        H_patches: int = 0,
        W_patches: int = 0,
    ) -> torch.Tensor:
        """Compress ``(B, D, N, C)`` patch tokens to ``(B, M, C)``.

        ``etext``, ``H_patches``, and ``W_patches`` are accepted for interface
        compatibility but are **ignored**.
        """
        _ = etext, H_patches, W_patches  # task-blind, geometry-blind

        B, D, N, C = patch_tokens.shape

        # Flatten D×N into a single sequence — no axis distinction
        kv = patch_tokens.reshape(B, D * N, C)
        kv = self.norm_kv(kv)

        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        q = self.norm_q(q)

        out, _ = self.cross_attn(q, kv, kv, need_weights=False)  # (B, M, C)
        return out

    # ── Interface compatibility ────────────────────────────────────────────────

    def get_slice_attention(self) -> None:
        """Always returns ``None`` — the Perceiver has no per-slice attention.

        Callers that use slice attention for recall@k evaluation must guard
        against ``None`` when swapping in this baseline.
        """
        return None
