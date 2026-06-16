"""Positional encodings for the A-ADP projector.

Two encodings are provided:
  - SinusoidalPosEnc2D  — fixed 2D sin/cos encoding for patch grids (variable H×W)
  - LearnableDepthEnc1D — learnable 1D embedding for CT depth (slice index)
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEnc2D(nn.Module):
    """Non-learnable 2D sinusoidal positional encoding for patch grids.

    The ``embed_dim``-dimensional encoding is split equally across four
    components: row-sin, row-cos, col-sin, col-cos (each ``embed_dim // 4``
    dimensions).  The encoding is computed dynamically on every call so any
    ``(H_patches, W_patches)`` resolution is supported without retraining.

    Args:
        embed_dim: Token embedding dimension. **Must be divisible by 4.**
        device:    Device to create output tensors on (default ``"cuda"``).
    """

    def __init__(
        self,
        embed_dim: int,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError(
                f"embed_dim must be divisible by 4, got {embed_dim}"
            )
        self.embed_dim = embed_dim
        self._device = torch.device(device)

    def forward(self, H_patches: int, W_patches: int) -> torch.Tensor:
        """Compute 2D sinusoidal encoding for a patch grid.

        Args:
            H_patches: Number of patch rows.
            W_patches: Number of patch columns.

        Returns:
            ``(H_patches * W_patches, embed_dim)`` float32 tensor on ``self._device``.
        """
        d_quarter = self.embed_dim // 4
        device = self._device

        # Frequency basis: shape (d_quarter,)
        # freq[i] = 1 / 10000^(i / d_quarter)
        i = torch.arange(d_quarter, dtype=torch.float32, device=device)
        freq = 1.0 / (10000.0 ** (i / d_quarter))

        # Row and column position vectors
        rows = torch.arange(H_patches, dtype=torch.float32, device=device)  # (H,)
        cols = torch.arange(W_patches, dtype=torch.float32, device=device)  # (W,)

        # Outer product → sin/cos tables:  (H, d_quarter) and (W, d_quarter)
        row_angles = rows.unsqueeze(1) * freq.unsqueeze(0)
        col_angles = cols.unsqueeze(1) * freq.unsqueeze(0)

        row_sin = torch.sin(row_angles)  # (H, d_quarter)
        row_cos = torch.cos(row_angles)
        col_sin = torch.sin(col_angles)  # (W, d_quarter)
        col_cos = torch.cos(col_angles)

        # Broadcast to a (H, W, d_quarter) grid for each component
        row_sin_g = row_sin.unsqueeze(1).expand(-1, W_patches, -1)  # (H, W, d_quarter)
        row_cos_g = row_cos.unsqueeze(1).expand(-1, W_patches, -1)
        col_sin_g = col_sin.unsqueeze(0).expand(H_patches, -1, -1)
        col_cos_g = col_cos.unsqueeze(0).expand(H_patches, -1, -1)

        # Concatenate along the channel axis → (H, W, embed_dim)
        enc = torch.cat([row_sin_g, row_cos_g, col_sin_g, col_cos_g], dim=-1)

        return enc.reshape(H_patches * W_patches, self.embed_dim)


class LearnableDepthEnc1D(nn.Module):
    """Learnable 1D depth positional encoding for CT volumes.

    An ``nn.Embedding`` table of size ``(max_depth, embed_dim)`` is trained
    end-to-end.  When ``forward(D)`` is called with ``D > max_depth`` the
    weight matrix is linearly interpolated to produce ``D`` embeddings
    (forward-compat for unusually deep volumes).

    Args:
        max_depth:  Maximum number of depth positions in the embedding table.
        embed_dim:  Token embedding dimension.
        device:     Device to create the module on (default ``"cuda"``).
    """

    def __init__(
        self,
        max_depth: int,
        embed_dim: int,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(max_depth, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.to(torch.device(device))

    def forward(self, D: int) -> torch.Tensor:
        """Return depth positional embeddings for ``D`` slices.

        Args:
            D: Number of depth slices.

        Returns:
            ``(D, embed_dim)`` float32 tensor on the module's device.
        """
        if D <= self.max_depth:
            idx = torch.arange(D, device=self.embedding.weight.device)
            return self.embedding(idx)

        # Linear interpolation when D exceeds the table size
        weight = self.embedding.weight           # (max_depth, embed_dim)
        weight_t = weight.T.unsqueeze(0)         # (1, embed_dim, max_depth)
        interpolated = F.interpolate(
            weight_t, size=D, mode="linear", align_corners=False
        )                                        # (1, embed_dim, D)
        return interpolated.squeeze(0).T         # (D, embed_dim)

    def with_spacing(self, D: int, spacing_mm: float) -> torch.Tensor:
        """Depth embeddings with optional spacing-based modulation (forward-compat).

        Args:
            D:          Number of depth slices.
            spacing_mm: Slice thickness in mm.

        Returns:
            ``(D, embed_dim)`` — currently identical to ``forward(D)``.
        """
        _ = spacing_mm  # TODO: scale or modulate embeddings by spacing_mm
        return self.forward(D)
