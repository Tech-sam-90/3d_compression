"""Tests for SliceEncoder using vit_tiny_patch16_224 (no auth required).

pretrained=False is used throughout so no weights are downloaded during CI.
All forward passes run on GPU (DEVICE).
"""

import pytest
import torch

from aadp.models.encoder.vit_encoder import SliceEncoder

_MODEL = "vit_tiny_patch16_224"
_C = 192   # ViT-Tiny embed_dim
_P = 16    # patch_size

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Module-scoped fixtures — model created once per file ──────────────────────


@pytest.fixture(scope="module")
def frozen_encoder() -> SliceEncoder:
    return SliceEncoder(model_name=_MODEL, pretrained=False, frozen=True).to(DEVICE)


@pytest.fixture(scope="module")
def unfrozen_encoder() -> SliceEncoder:
    return SliceEncoder(model_name=_MODEL, pretrained=False, frozen=False).to(DEVICE)


# ── Native-resolution (resize_to=None) tests ──────────────────────────────────


def test_native_512_shape(frozen_encoder: SliceEncoder) -> None:
    """512×512 with patch16 → 32×32 = 1024 patches per slice."""
    x = torch.randn(4, 1, 512, 512, device=DEVICE)
    out = frozen_encoder(x)
    assert out.shape == (4, 1024, _C), f"got {out.shape}"


def test_native_512_values_finite(frozen_encoder: SliceEncoder) -> None:
    x = torch.randn(4, 1, 512, 512, device=DEVICE)
    out = frozen_encoder(x)
    assert torch.isfinite(out).all(), "output contains NaN or Inf"


def test_native_1024_shape(frozen_encoder: SliceEncoder) -> None:
    """1024×1024 with patch16 → 64×64 = 4096 patches per slice."""
    x = torch.randn(2, 1, 1024, 1024, device=DEVICE)
    out = frozen_encoder(x)
    assert out.shape == (2, 4096, _C), f"got {out.shape}"


# ── resize_to mode ────────────────────────────────────────────────────────────


def test_resize_224_shape() -> None:
    """resize_to=224 with patch16 → 14×14 = 196 patches per slice."""
    enc = SliceEncoder(model_name=_MODEL, pretrained=False, resize_to=224).to(DEVICE)
    x = torch.randn(4, 1, 512, 512, device=DEVICE)
    out = enc(x)
    assert out.shape == (4, 196, _C), f"got {out.shape}"


# ── all_tokens mode ───────────────────────────────────────────────────────────


def test_all_tokens_includes_cls() -> None:
    """all_tokens at 224×224 → 196 patches + 1 CLS = 197 tokens."""
    enc = SliceEncoder(
        model_name=_MODEL, pretrained=False, output_layer="all_tokens"
    ).to(DEVICE)
    x = torch.randn(2, 1, 224, 224, device=DEVICE)
    out = enc(x)
    assert out.shape == (2, 197, _C), f"got {out.shape}"


def test_patch_tokens_has_no_cls(frozen_encoder: SliceEncoder) -> None:
    """patch_tokens at 224×224 → exactly 196 tokens (CLS stripped)."""
    x = torch.randn(2, 1, 224, 224, device=DEVICE)
    out = frozen_encoder(x)
    assert out.shape == (2, 196, _C)


# ── output_dim property ───────────────────────────────────────────────────────


def test_output_dim_matches_tensor_last_dim(frozen_encoder: SliceEncoder) -> None:
    x = torch.randn(2, 1, 224, 224, device=DEVICE)
    out = frozen_encoder(x)
    assert frozen_encoder.output_dim == out.shape[-1]


def test_output_dim_is_correct_for_tiny(frozen_encoder: SliceEncoder) -> None:
    assert frozen_encoder.output_dim == _C


# ── patch_grid_size ───────────────────────────────────────────────────────────


def test_patch_grid_512(frozen_encoder: SliceEncoder) -> None:
    assert frozen_encoder.patch_grid_size(512, 512) == (32, 32)


def test_patch_grid_1024(frozen_encoder: SliceEncoder) -> None:
    assert frozen_encoder.patch_grid_size(1024, 1024) == (64, 64)


def test_patch_grid_224(frozen_encoder: SliceEncoder) -> None:
    assert frozen_encoder.patch_grid_size(224, 224) == (14, 14)


# ── num_patches ───────────────────────────────────────────────────────────────


def test_num_patches_512(frozen_encoder: SliceEncoder) -> None:
    assert frozen_encoder.num_patches(512, 512) == 1024


def test_num_patches_1024(frozen_encoder: SliceEncoder) -> None:
    assert frozen_encoder.num_patches(1024, 1024) == 4096


def test_num_patches_224(frozen_encoder: SliceEncoder) -> None:
    assert frozen_encoder.num_patches(224, 224) == 196


# ── Frozen / unfrozen gradient behaviour ─────────────────────────────────────


def test_frozen_encoder_no_requires_grad(frozen_encoder: SliceEncoder) -> None:
    assert not any(p.requires_grad for p in frozen_encoder.parameters())


def test_unfrozen_encoder_has_requires_grad(unfrozen_encoder: SliceEncoder) -> None:
    assert any(p.requires_grad for p in unfrozen_encoder.parameters())


# ── Single-channel input ──────────────────────────────────────────────────────


def test_single_channel_input_does_not_raise(frozen_encoder: SliceEncoder) -> None:
    """(B_D, 1, H, W) — channel repeat is handled internally."""
    x = torch.randn(2, 1, 224, 224, device=DEVICE)
    out = frozen_encoder(x)
    assert out.shape[0] == 2


# ── Device ────────────────────────────────────────────────────────────────────


def test_output_device_matches_input(frozen_encoder: SliceEncoder) -> None:
    x = torch.randn(2, 1, 224, 224, device=DEVICE)
    out = frozen_encoder(x)
    assert out.device.type == DEVICE.type


# ── Dynamic resolution — different sizes in sequence ─────────────────────────


def test_different_resolutions_sequentially_do_not_raise(
    frozen_encoder: SliceEncoder,
) -> None:
    """dynamic_img_size=True interpolates pos embeds at runtime per batch."""
    out1 = frozen_encoder(torch.randn(2, 1, 224, 224, device=DEVICE))
    out2 = frozen_encoder(torch.randn(2, 1, 512, 512, device=DEVICE))
    assert out1.shape == (2, 196, _C)
    assert out2.shape == (2, 1024, _C)


# ── Invalid output_layer ──────────────────────────────────────────────────────


def test_invalid_output_layer_raises() -> None:
    with pytest.raises(ValueError, match="output_layer"):
        SliceEncoder(model_name=_MODEL, pretrained=False, output_layer="mean")  # type: ignore[arg-type]
