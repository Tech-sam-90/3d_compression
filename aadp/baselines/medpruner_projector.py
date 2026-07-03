"""MedPrunerProjector — similarity-based hard slice deletion baseline.

Replicates the MedPruner approach: consecutive slices whose mean-token
representations are highly similar are considered redundant and hard-deleted
before pooling to the target token budget.

**Known limitation vs. A-ADP (documented by design):**

Hard deletion is irreversible.  Once a slice is removed at the pruning step
its tokens cannot be recovered downstream.  If the similarity threshold is
too aggressive — or if the instruction asks about a finding that happens to
occupy a region of slices that are similar to their neighbours — those slices
are silently discarded.  This is the recall failure mode that A-ADP avoids
by using soft, attention-weighted aggregation instead of hard pruning.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedPrunerProjector(nn.Module):
    """Similarity-based hard slice pruner followed by adaptive mean pooling.

    Accepts the same ``(B, D, N, C)`` → ``(B, M, C)`` contract as
    ``AADPProjector`` for drop-in comparison.

    Args:
        embed_dim:            Hidden dimension C.
        num_tokens:           M — target output tokens after pooling. Default 64.
        similarity_threshold: Cosine similarity above which a slice is deemed
                              redundant and deleted. Default 0.95.
        device:               Target device. Default ``"cuda"``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 64,
        similarity_threshold: float = 0.95,
        device: Union[torch.device, str] = "cuda",
        **kwargs,
    ) -> None:
        super().__init__()

        self._embed_dim = embed_dim
        self._num_tokens = num_tokens
        self.similarity_threshold = similarity_threshold

        self._last_keep_mask: Optional[torch.Tensor] = None

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
        """Prune redundant slices, then pool to ``(B, M, C)``.

        ``etext``, ``H_patches``, and ``W_patches`` are accepted for interface
        compatibility but are **ignored**.
        """
        _ = etext, H_patches, W_patches

        B, D, N, C = patch_tokens.shape

        # ── Step 1: per-slice mean token representation ───────────────────────
        slice_means = patch_tokens.mean(dim=2)  # (B, D, C)

        # ── Step 2: cosine similarity between consecutive slices ─────────────
        # Normalise along the channel axis for cosine computation
        normed = F.normalize(slice_means, dim=-1)  # (B, D, C)
        # sim[:, d] = cosine similarity between slice d and slice d+1
        sim = (normed[:, :-1, :] * normed[:, 1:, :]).sum(dim=-1)  # (B, D-1)

        # ── Step 3-4: build per-batch keep mask ──────────────────────────────
        # Slice 0 is always kept; slice d+1 is dropped if sim[:, d] > threshold
        keep = torch.ones(B, D, dtype=torch.bool, device=patch_tokens.device)
        keep[:, 1:] = sim <= self.similarity_threshold  # (B, D)

        self._last_keep_mask = keep.clone()

        # ── Step 5-6: pad kept slices and pool to M tokens ───────────────────
        # Compute per-batch mean over kept slice tokens, then adaptive pool to M
        # We process each batch item independently to handle variable keep counts,
        # then stack the results.
        outputs = []
        max_kept = keep.sum(dim=1).max().item()  # max kept slices in batch

        for b in range(B):
            mask_b = keep[b]               # (D,)
            kept = patch_tokens[b][mask_b]  # (K_b, N, C) where K_b = kept slices

            # Flatten K_b×N → single sequence, then pool to M tokens
            # shape: (K_b*N, C) → treat as 1D sequence of length K_b*N
            seq = kept.reshape(1, kept.shape[0] * N, C)  # (1, K_b*N, C)
            # adaptive_avg_pool1d expects (B, C, L) → transpose, pool, transpose
            pooled = F.adaptive_avg_pool1d(
                seq.transpose(1, 2), self._num_tokens
            ).transpose(1, 2)              # (1, M, C)
            outputs.append(pooled.squeeze(0))  # (M, C)

        return torch.stack(outputs, dim=0)  # (B, M, C)

    # ── Interface compatibility ────────────────────────────────────────────────

    def get_slice_attention(self) -> None:
        """Always returns ``None`` — MedPruner uses hard deletion, not attention.

        Callers that use slice attention for recall@k evaluation must guard
        against ``None`` when swapping in this baseline.
        """
        return None

    def get_keep_mask(self) -> Optional[torch.Tensor]:
        """Return the boolean keep mask from the last forward pass.

        Returns:
            ``(B, D)`` bool tensor — ``True`` for kept slices, ``False`` for
            deleted.  Returns ``None`` if called before any forward pass.
        """
        return self._last_keep_mask
