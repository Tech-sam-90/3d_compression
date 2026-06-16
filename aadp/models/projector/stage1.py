"""Stage 1 of the A-ADP projector: IntraSliceDistiller.

Compresses each CT slice's patch tokens independently using Perceiver-style
cross-attention. Stage 1 is task-agnostic — it runs before instruction
conditioning and reduces N patch tokens per slice down to K latents.
"""

from typing import Union

import torch
import torch.nn as nn

from aadp.models.projector.pos_encoding import SinusoidalPosEnc2D


class IntraSliceDistiller(nn.Module):
    """Perceiver-style cross-attention compressor applied independently per slice.

    Each slice's N patch tokens are used as keys/values. K learnable query
    latents (shared across all slices and volumes) attend over them to produce
    K compressed output tokens.  The output size K is fixed regardless of the
    input resolution N.

    Args:
        embed_dim:   Token embedding dimension C. Must match the ViT encoder's
                     ``output_dim``. ``num_heads`` must divide ``embed_dim``.
        num_latents: K — number of output latents per slice. Default ``32``.
        num_heads:   Number of attention heads. Default ``8``.
        dropout:     Attention dropout probability. Default ``0.0``.
        device:      Device to place the module on. Default ``"cuda"``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_latents: int = 32,
        num_heads: int = 8,
        dropout: float = 0.0,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self._embed_dim = embed_dim
        self._num_latents = num_latents

        # Learnable query latents: (K, C), shared across all slices and volumes
        self.latents = nn.Parameter(torch.empty(num_latents, embed_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.pos_enc = SinusoidalPosEnc2D(embed_dim, device=device)

        self.to(torch.device(device))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_latents(self) -> int:
        """Number of output latent tokens per slice (K)."""
        return self._num_latents

    @property
    def embed_dim(self) -> int:
        """Token embedding dimension (C)."""
        return self._embed_dim

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        H_patches: int,
        W_patches: int,
    ) -> torch.Tensor:
        """Compress slice patch tokens to K latents via cross-attention.

        Args:
            x:         ``(B_D, N, C)`` patch tokens from SliceEncoder.
                       B_D = batch_size × num_slices flattened into batch dim.
                       N = H_patches × W_patches.
            H_patches: Patch grid height (e.g. 32 for 512×512 with patch16).
            W_patches: Patch grid width.

        Returns:
            ``(B_D, K, C)`` — K compressed latents per slice.
        """
        B_D = x.shape[0]

        # Step 1-2: add 2D sinusoidal positional encoding to patch tokens
        pos = self.pos_enc(H_patches, W_patches)   # (N, C)
        x = x + pos.unsqueeze(0)                   # (B_D, N, C)

        # Step 3: normalise keys/values
        kv = self.norm_kv(x)

        # Step 4-5: expand and normalise query latents
        q = self.latents.unsqueeze(0).expand(B_D, -1, -1)  # (B_D, K, C)
        q = self.norm_q(q)

        # Step 6: cross-attention — queries attend over patch-token keys/values
        out, _ = self.cross_attn(q, kv, kv, need_weights=False)  # (B_D, K, C)

        return out
