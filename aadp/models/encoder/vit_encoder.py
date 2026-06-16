from typing import Literal, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceEncoder(nn.Module):
    """Wraps a pretrained 2D ViT to encode individual CT slices into patch tokens.

    **Resolution policy:**
    By default (``resize_to=None``) slices are passed at their native CT
    resolution — 512×512 or 1024×1024 — without any resizing. This preserves
    small findings such as pulmonary nodules that may span only 2-4 pixels.
    Use ``timm.create_model(..., dynamic_img_size=True)`` so the ViT
    interpolates its positional embeddings at runtime to match whatever
    resolution is fed in.

    Set ``resize_to`` only when GPU memory is the hard constraint. Resizing
    to 224 reduces N from 1024 to 196 patches per slice, which can cause
    small lesions to fall below the patch-resolution floor.

    Args:
        model_name:   timm ViT model name. The "224" in the default name
                      refers to the pretrained-weight initialisation size,
                      not the inference size.
        pretrained:   Load pretrained ImageNet weights.
        frozen:       Freeze all ViT parameters. Only the A-ADP projector
                      trains; the slice encoder is kept fixed.
        output_layer: ``"patch_tokens"`` (default) — strips the CLS token
                      and returns only spatial patch tokens ``(B_D, N, C)``.
                      ``"all_tokens"`` keeps CLS at index 0.
        resize_to:    If not None, resize each slice to this square size
                      before encoding. ``None`` preserves native resolution.
        patch_size:   ViT patch size in pixels; must match the model name.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        frozen: bool = True,
        output_layer: Literal["patch_tokens", "all_tokens"] = "patch_tokens",
        resize_to: Optional[int] = None,
        patch_size: int = 16,
    ) -> None:
        super().__init__()

        if output_layer not in ("patch_tokens", "all_tokens"):
            raise ValueError(
                f"output_layer must be 'patch_tokens' or 'all_tokens', got '{output_layer}'"
            )

        self.output_layer = output_layer
        self.frozen = frozen
        self.resize_to = resize_to
        self._patch_size = patch_size

        # dynamic_img_size=True makes timm interpolate positional embeddings
        # at runtime so the model accepts any (H, W) without retraining.
        self._model = timm.create_model(
            model_name, pretrained=pretrained, dynamic_img_size=True
        )
        self._model.reset_classifier(0)

        if frozen:
            self._model.requires_grad_(False)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        """Hidden dimension C of the ViT (e.g. 768 for ViT-Base, 192 for ViT-Tiny)."""
        return self._model.embed_dim

    @property
    def patch_size(self) -> int:
        """Patch size in pixels (e.g. 16)."""
        return self._patch_size

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def patch_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        """Number of patch rows and columns for an input of size (H, W).

        Stage 1 uses this to build the 2D sinusoidal positional encoding
        that matches the current slice resolution.
        """
        return H // self._patch_size, W // self._patch_size

    def num_patches(self, H: int, W: int) -> int:
        """Total number of patch tokens for an input of size (H, W)."""
        gh, gw = self.patch_grid_size(H, W)
        return gh * gw

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of grayscale CT slices into patch token embeddings.

        Args:
            x: ``(B_D, 1, H, W)`` — a batch of B×D slices (depth flattened
               into the batch dimension), single grayscale channel.

        Returns:
            ``(B_D, N, C)`` patch token embeddings.
            N = ``num_patches(H, W)`` when ``output_layer="patch_tokens"``.
            N = ``num_patches(H, W) + 1`` when ``output_layer="all_tokens"``.
        """
        # Step 1: CT slices are single-channel; ViT expects 3 channels.
        x = x.repeat(1, 3, 1, 1)  # (B_D, 3, H, W)

        # Step 2: optional resize (only when GPU memory is the constraint).
        if self.resize_to is not None:
            x = F.interpolate(
                x,
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
            )

        # Steps 3-4: feature extraction, optionally without gradients.
        if self.frozen:
            with torch.no_grad():
                return self._extract(x)
        return self._extract(x)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        # Step 3: forward_features returns (B_D, N+1, C) — CLS token at index 0.
        tokens = self._model.forward_features(x)

        # Step 4: strip CLS token unless the caller wants all tokens.
        if self.output_layer == "patch_tokens":
            tokens = tokens[:, 1:, :]  # (B_D, N, C)

        return tokens
