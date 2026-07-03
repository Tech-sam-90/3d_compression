"""Tests for SinusoidalPosEnc2D and LearnableDepthEnc1D. All on CUDA."""

import pytest
import torch

from aadp.models.projector.pos_encoding import LearnableDepthEnc1D, SinusoidalPosEnc2D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_D_EMB = 64


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


# ── SinusoidalPosEnc2D fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def sin2d() -> SinusoidalPosEnc2D:
    _skip_if_no_cuda()
    return SinusoidalPosEnc2D(embed_dim=_D_EMB, device=DEVICE)


# ── SinusoidalPosEnc2D — shape ────────────────────────────────────────────────


def test_sin2d_shape_14x14(sin2d: SinusoidalPosEnc2D) -> None:
    _skip_if_no_cuda()
    out = sin2d(14, 14)
    assert out.shape == (196, _D_EMB), f"got {out.shape}"


def test_sin2d_shape_32x32(sin2d: SinusoidalPosEnc2D) -> None:
    _skip_if_no_cuda()
    out = sin2d(32, 32)
    assert out.shape == (1024, _D_EMB), f"got {out.shape}"


def test_sin2d_shape_64x64(sin2d: SinusoidalPosEnc2D) -> None:
    _skip_if_no_cuda()
    out = sin2d(64, 64)
    assert out.shape == (4096, _D_EMB), f"got {out.shape}"


def test_sin2d_shape_asymmetric(sin2d: SinusoidalPosEnc2D) -> None:
    _skip_if_no_cuda()
    out = sin2d(16, 32)
    assert out.shape == (512, _D_EMB), f"got {out.shape}"


# ── SinusoidalPosEnc2D — device & dtype ──────────────────────────────────────


def test_sin2d_output_on_cuda(sin2d: SinusoidalPosEnc2D) -> None:
    _skip_if_no_cuda()
    out = sin2d(14, 14)
    assert out.device.type == "cuda"


def test_sin2d_output_dtype_float32(sin2d: SinusoidalPosEnc2D) -> None:
    _skip_if_no_cuda()
    out = sin2d(14, 14)
    assert out.dtype == torch.float32


# ── SinusoidalPosEnc2D — no learnable parameters ────────────────────────────


def test_sin2d_no_parameters(sin2d: SinusoidalPosEnc2D) -> None:
    _skip_if_no_cuda()
    assert list(sin2d.parameters()) == []


# ── SinusoidalPosEnc2D — correctness ─────────────────────────────────────────


def test_sin2d_values_finite(sin2d: SinusoidalPosEnc2D) -> None:
    _skip_if_no_cuda()
    out = sin2d(14, 14)
    assert torch.isfinite(out).all()


def test_sin2d_different_patches_differ(sin2d: SinusoidalPosEnc2D) -> None:
    """Patch (0,0) and patch (0,1) must have different encodings."""
    _skip_if_no_cuda()
    out = sin2d(4, 4)  # 16 patches, row-major
    # patch (0,0) = index 0;  patch (0,1) = index 1
    assert not torch.allclose(out[0], out[1])


def test_sin2d_same_call_reproducible(sin2d: SinusoidalPosEnc2D) -> None:
    _skip_if_no_cuda()
    a = sin2d(14, 14)
    b = sin2d(14, 14)
    assert torch.equal(a, b)


# ── SinusoidalPosEnc2D — validation ──────────────────────────────────────────


def test_sin2d_embed_dim_not_divisible_by_4_raises() -> None:
    _skip_if_no_cuda()
    with pytest.raises(ValueError, match="divisible by 4"):
        SinusoidalPosEnc2D(embed_dim=30, device=DEVICE)


# ── LearnableDepthEnc1D fixtures ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def depth_enc() -> LearnableDepthEnc1D:
    _skip_if_no_cuda()
    return LearnableDepthEnc1D(max_depth=256, embed_dim=_D_EMB, device=DEVICE)


# ── LearnableDepthEnc1D — shape (D ≤ max_depth) ──────────────────────────────


def test_depth_enc_shape_128(depth_enc: LearnableDepthEnc1D) -> None:
    _skip_if_no_cuda()
    out = depth_enc(128)
    assert out.shape == (128, _D_EMB), f"got {out.shape}"


def test_depth_enc_shape_256(depth_enc: LearnableDepthEnc1D) -> None:
    _skip_if_no_cuda()
    out = depth_enc(256)
    assert out.shape == (256, _D_EMB), f"got {out.shape}"


def test_depth_enc_shape_303(depth_enc: LearnableDepthEnc1D) -> None:
    _skip_if_no_cuda()
    out = depth_enc(303)
    assert out.shape == (303, _D_EMB), f"got {out.shape}"


# ── LearnableDepthEnc1D — interpolation (D > max_depth) ──────────────────────


def test_depth_enc_interpolates_beyond_max_depth() -> None:
    _skip_if_no_cuda()
    enc = LearnableDepthEnc1D(max_depth=256, embed_dim=_D_EMB, device=DEVICE)
    out = enc(350)
    assert out.shape == (350, _D_EMB), f"got {out.shape}"


def test_depth_enc_interpolated_output_finite() -> None:
    _skip_if_no_cuda()
    enc = LearnableDepthEnc1D(max_depth=256, embed_dim=_D_EMB, device=DEVICE)
    out = enc(400)
    assert torch.isfinite(out).all()


# ── LearnableDepthEnc1D — device & dtype ─────────────────────────────────────


def test_depth_enc_output_on_cuda(depth_enc: LearnableDepthEnc1D) -> None:
    _skip_if_no_cuda()
    out = depth_enc(128)
    assert out.device.type == "cuda"


def test_depth_enc_output_dtype_float32(depth_enc: LearnableDepthEnc1D) -> None:
    _skip_if_no_cuda()
    out = depth_enc(128)
    assert out.dtype == torch.float32


# ── LearnableDepthEnc1D — learnable parameters ───────────────────────────────


def test_depth_enc_has_parameters(depth_enc: LearnableDepthEnc1D) -> None:
    _skip_if_no_cuda()
    params = list(depth_enc.parameters())
    assert len(params) > 0


def test_depth_enc_parameters_require_grad(depth_enc: LearnableDepthEnc1D) -> None:
    _skip_if_no_cuda()
    assert all(p.requires_grad for p in depth_enc.parameters())


def test_depth_enc_gradients_flow() -> None:
    _skip_if_no_cuda()
    enc = LearnableDepthEnc1D(max_depth=256, embed_dim=_D_EMB, device=DEVICE)
    out = enc(128)
    out.sum().backward()
    assert enc.embedding.weight.grad is not None
