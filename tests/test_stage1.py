"""Tests for IntraSliceDistiller (Stage 1 of A-ADP). All on CUDA."""

import pytest
import torch

from aadp.models.projector.stage1 import IntraSliceDistiller

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


# ── Module-scoped fixture (ViT-Base config, K=32) ────────────────────────────


@pytest.fixture(scope="module")
def distiller_base() -> IntraSliceDistiller:
    _skip_if_no_cuda()
    return IntraSliceDistiller(embed_dim=768, num_latents=32, num_heads=8, device=DEVICE)


# ── Shape tests ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "B, D, N, C, K, H, W",
    [
        (2, 128, 1024, 768, 32, 32, 32),   # typical: batch=2, 128 slices, 512x512, ViT-Base
        (1, 303, 1024, 768, 32, 32, 32),   # single volume, full CT-RATE depth
        (2, 128, 1024, 768, 16, 32, 32),   # K=16 ablation
        (2, 128, 1024, 768, 64, 32, 32),   # K=64 ablation
        (1,  50,  196, 192, 32, 14, 14),   # ViT-Tiny at 224x224
    ],
)
def test_output_shape(B: int, D: int, N: int, C: int, K: int, H: int, W: int) -> None:
    _skip_if_no_cuda()
    model = IntraSliceDistiller(embed_dim=C, num_latents=K, num_heads=8, device=DEVICE)
    x = torch.randn(B * D, N, C, device=DEVICE)
    out = model(x, H, W)
    assert out.shape == (B * D, K, C), f"expected ({B*D}, {K}, {C}), got {out.shape}"


# ── Device & dtype ────────────────────────────────────────────────────────────


def test_output_on_cuda(distiller_base: IntraSliceDistiller) -> None:
    _skip_if_no_cuda()
    x = torch.randn(4, 1024, 768, device=DEVICE)
    out = distiller_base(x, 32, 32)
    assert out.device.type == "cuda"


def test_output_dtype_float32(distiller_base: IntraSliceDistiller) -> None:
    _skip_if_no_cuda()
    x = torch.randn(4, 1024, 768, device=DEVICE)
    out = distiller_base(x, 32, 32)
    assert out.dtype == torch.float32


# ── Finite values ─────────────────────────────────────────────────────────────


def test_output_finite(distiller_base: IntraSliceDistiller) -> None:
    _skip_if_no_cuda()
    x = torch.randn(4, 1024, 768, device=DEVICE)
    out = distiller_base(x, 32, 32)
    assert torch.isfinite(out).all()


# ── Latent sharing ────────────────────────────────────────────────────────────


def test_latents_shape(distiller_base: IntraSliceDistiller) -> None:
    """self.latents is (K, C) shared across all slices."""
    _skip_if_no_cuda()
    assert distiller_base.latents.shape == (32, 768)


# ── Properties ────────────────────────────────────────────────────────────────


def test_num_latents_property(distiller_base: IntraSliceDistiller) -> None:
    _skip_if_no_cuda()
    assert distiller_base.num_latents == 32


def test_embed_dim_property(distiller_base: IntraSliceDistiller) -> None:
    _skip_if_no_cuda()
    assert distiller_base.embed_dim == 768


# ── Gradients ─────────────────────────────────────────────────────────────────


def test_gradients_flow_through_latents() -> None:
    _skip_if_no_cuda()
    model = IntraSliceDistiller(embed_dim=192, num_latents=16, num_heads=8, device=DEVICE)
    x = torch.randn(4, 196, 192, device=DEVICE)
    out = model(x, 14, 14)
    out.sum().backward()
    assert model.latents.grad is not None, "latents.grad should not be None after backward"


# ── Positional encoding integration ──────────────────────────────────────────


def test_pos_enc_changes_output() -> None:
    """Adding non-zero pos enc changes the output vs. all-zero pos enc."""
    _skip_if_no_cuda()
    model = IntraSliceDistiller(embed_dim=192, num_latents=16, num_heads=8, device=DEVICE)
    x = torch.randn(4, 196, 192, device=DEVICE)

    out_with_pos = model(x, 14, 14)

    # Temporarily replace pos_enc to return zeros
    def zero_pos_enc(H: int, W: int) -> torch.Tensor:
        return torch.zeros(H * W, 192, device=DEVICE)

    original = model.pos_enc.forward
    model.pos_enc.forward = zero_pos_enc  # type: ignore[method-assign]
    out_without_pos = model(x, 14, 14)
    model.pos_enc.forward = original  # type: ignore[method-assign]

    assert not torch.allclose(out_with_pos, out_without_pos), (
        "pos enc should change the output"
    )


# ── Variable N (cross-attention handles variable sequence length) ─────────────


def test_variable_n_sequentially(distiller_base: IntraSliceDistiller) -> None:
    """Same distiller handles N=196 and N=1024 in sequence without error."""
    _skip_if_no_cuda()
    out1 = distiller_base(torch.randn(2, 196, 768, device=DEVICE), 14, 14)
    out2 = distiller_base(torch.randn(2, 1024, 768, device=DEVICE), 32, 32)
    assert out1.shape == (2, 32, 768)
    assert out2.shape == (2, 32, 768)


# ── num_heads mismatch ────────────────────────────────────────────────────────


def test_num_heads_not_divisor_raises() -> None:
    _skip_if_no_cuda()
    with pytest.raises((ValueError, AssertionError)):
        IntraSliceDistiller(embed_dim=768, num_heads=7, device=DEVICE)
