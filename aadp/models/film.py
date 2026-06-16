"""FiLM (Feature-wise Linear Modulation) conditioning layers for A-ADP Stage 2.

FiLM applies a learned affine transformation (scale γ, shift β) derived from
a conditioning signal to modulate intermediate representations.  In A-ADP the
conditioning signal is the instruction embedding ``etext`` and the target is
the depth query tensor ``Qd`` in Stage 2.

References:
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
"""

from typing import Union

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Affine modulation conditioned on an instruction embedding.

    Projects the conditioning vector ``cond`` into per-channel scale (γ) and
    shift (β) vectors, then applies ``x * γ + β`` element-wise across the
    sequence dimension.

    The projections are initialised so that at step 0 the layer is an identity
    (γ = 1, β = 0), meaning the modulated queries start identical to the
    unmodulated ones and the network learns to deviate from this as training
    progresses.

    Args:
        cond_dim:   Dimensionality of the conditioning input ``etext``
                    (e.g. 768 for OPT-350M, 2048 for LLaMA-3.2-1B).
        target_dim: Dimensionality of the tensor being modulated ``x``
                    (Stage 2's ``embed_dim`` C).
        device:     Device to place the module on. Default ``"cuda"``.
    """

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        self.gamma_proj = nn.Linear(cond_dim, target_dim, bias=True)
        self.beta_proj = nn.Linear(cond_dim, target_dim, bias=True)

        # Identity initialisation: γ=1, β=0 at training start
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

        self.to(torch.device(device))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x:    ``(B, M, target_dim)`` — depth queries Qd to be modulated.
            cond: ``(B, cond_dim)`` — instruction embedding ``etext``.

        Returns:
            ``(B, M, target_dim)`` — modulated queries.
        """
        gamma = self.gamma_proj(cond).unsqueeze(1)  # (B, 1, target_dim)
        beta = self.beta_proj(cond).unsqueeze(1)    # (B, 1, target_dim)
        return x * gamma + beta


class NullFiLMLayer(nn.Module):
    """Drop-in FiLM replacement that performs no modulation.

    Accepts the same constructor and forward signature as ``FiLMLayer`` but
    returns ``x`` unchanged, ignoring ``cond`` entirely.  Used in ablations
    to test the task-agnostic baseline (Stage 2 without instruction
    conditioning).

    Args:
        cond_dim:   Accepted for API compatibility; ignored.
        target_dim: Accepted for API compatibility; ignored.
        device:     Accepted for API compatibility; ignored.
    """

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()
        _ = cond_dim, target_dim, device  # unused; kept for signature parity

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged.

        Args:
            x:    ``(B, M, target_dim)`` — queries (returned as-is).
            cond: Ignored.

        Returns:
            ``x`` unmodified.
        """
        _ = cond
        return x
