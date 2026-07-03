"""Stage 2 of the A-ADP projector: InterSliceAggregator.

Aggregates D×K slice latents (from Stage 1) into a fixed M-token budget using
FiLM-conditioned depth queries and cross-attention. Depth positional encodings
make the model aware of slice position; FiLM modulation steers which slices
the M queries attend to based on the clinical instruction.
"""

from typing import Optional, Union

import torch
import torch.nn as nn

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
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self._embed_dim = embed_dim
        self._num_tokens = num_tokens

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
        self.norm_q = nn.LayerNorm(embed_dim)
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

        # Step 7: pre-norm
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # Step 8: cross-attention with weight capture for metrics
        out, attn_weights = self.cross_attn(q, kv, kv, need_weights=True)
        # out: (B, M, C),  attn_weights: (B, M, D*K)

        # Store weights detached (no grad accumulation in buffer)
        self._last_attn_weights = attn_weights.detach()

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
