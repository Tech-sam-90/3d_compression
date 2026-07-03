"""Tests for PerceiverProjector and MedPrunerProjector baselines. All on CUDA."""

import pytest
import torch

from baselines.perceiver_projector import PerceiverProjector
from baselines.medpruner_projector import MedPrunerProjector
from aadp.models.projector.aadp import AADPProjector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


def _patch(B: int, D: int, N: int, C: int) -> torch.Tensor:
    return torch.randn(B, D, N, C, device=DEVICE)


def _etext(B: int, cond_dim: int = 768) -> torch.Tensor:
    return torch.randn(B, cond_dim, device=DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# PerceiverProjector
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def perceiver() -> PerceiverProjector:
    _skip_if_no_cuda()
    return PerceiverProjector(embed_dim=768, num_tokens=64, device=DEVICE)


# ── Shape ─────────────────────────────────────────────────────────────────────


def test_perceiver_shape_standard(perceiver: PerceiverProjector) -> None:
    _skip_if_no_cuda()
    out = perceiver(_patch(2, 128, 1024, 768), _etext(2), 32, 32)
    assert out.shape == (2, 64, 768), f"got {out.shape}"


def test_perceiver_shape_full_ct_rate(perceiver: PerceiverProjector) -> None:
    _skip_if_no_cuda()
    out = perceiver(_patch(1, 303, 1024, 768), _etext(1), 32, 32)
    assert out.shape == (1, 64, 768), f"got {out.shape}"


# ── Task-blind: etext is ignored ──────────────────────────────────────────────


def test_perceiver_task_blind(perceiver: PerceiverProjector) -> None:
    """Different etext vectors must produce identical outputs."""
    _skip_if_no_cuda()
    tokens = _patch(2, 32, 196, 768)
    etext_a = _etext(2)
    etext_b = _etext(2)
    with torch.no_grad():
        out_a = perceiver(tokens, etext_a, 14, 14)
        out_b = perceiver(tokens, etext_b, 14, 14)
    assert torch.equal(out_a, out_b), "Perceiver must ignore etext"


# ── Geometry-blind: patch grid ignored ───────────────────────────────────────


def test_perceiver_geometry_blind(perceiver: PerceiverProjector) -> None:
    """Different H_patches/W_patches produce identical outputs for the same tokens."""
    _skip_if_no_cuda()
    tokens = _patch(2, 32, 196, 768)
    etext = _etext(2)
    with torch.no_grad():
        out_14 = perceiver(tokens, etext, H_patches=14, W_patches=14)
        out_32 = perceiver(tokens, etext, H_patches=32, W_patches=32)
    assert torch.equal(out_14, out_32), "Perceiver must ignore patch grid"


# ── get_slice_attention returns None ──────────────────────────────────────────


def test_perceiver_slice_attention_is_none(perceiver: PerceiverProjector) -> None:
    _skip_if_no_cuda()
    assert perceiver.get_slice_attention() is None


# ── Device, dtype, finite ─────────────────────────────────────────────────────


def test_perceiver_output_on_cuda(perceiver: PerceiverProjector) -> None:
    _skip_if_no_cuda()
    out = perceiver(_patch(2, 32, 196, 768), _etext(2), 14, 14)
    assert out.device.type == "cuda"


def test_perceiver_output_dtype_float32(perceiver: PerceiverProjector) -> None:
    _skip_if_no_cuda()
    out = perceiver(_patch(2, 32, 196, 768), _etext(2), 14, 14)
    assert out.dtype == torch.float32


def test_perceiver_output_finite(perceiver: PerceiverProjector) -> None:
    _skip_if_no_cuda()
    out = perceiver(_patch(2, 32, 196, 768), _etext(2), 14, 14)
    assert torch.isfinite(out).all()


# ── Gradients flow through latents ───────────────────────────────────────────


def test_perceiver_gradients_flow() -> None:
    _skip_if_no_cuda()
    model = PerceiverProjector(embed_dim=192, num_tokens=16, device=DEVICE)
    out = model(_patch(2, 16, 196, 192), _etext(2, 192), 14, 14)
    out.sum().backward()
    assert model.latents.grad is not None


# ── Properties ────────────────────────────────────────────────────────────────


def test_perceiver_num_tokens_property(perceiver: PerceiverProjector) -> None:
    _skip_if_no_cuda()
    assert perceiver.num_tokens == 64


def test_perceiver_embed_dim_property(perceiver: PerceiverProjector) -> None:
    _skip_if_no_cuda()
    assert perceiver.embed_dim == 768


# ═══════════════════════════════════════════════════════════════════════════════
# MedPrunerProjector
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def medpruner() -> MedPrunerProjector:
    _skip_if_no_cuda()
    return MedPrunerProjector(embed_dim=768, num_tokens=64, device=DEVICE)


# ── Shape ─────────────────────────────────────────────────────────────────────


def test_medpruner_shape_standard(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    out = medpruner(_patch(2, 128, 1024, 768), _etext(2), 32, 32)
    assert out.shape == (2, 64, 768), f"got {out.shape}"


def test_medpruner_shape_full_ct_rate(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    out = medpruner(_patch(1, 303, 1024, 768), _etext(1), 32, 32)
    assert out.shape == (1, 64, 768), f"got {out.shape}"


# ── Threshold extremes still produce (B, M, C) ───────────────────────────────


def test_medpruner_threshold_one_keeps_all() -> None:
    """threshold=1.0 keeps all slices (nothing exactly equals 1 on random data)."""
    _skip_if_no_cuda()
    model = MedPrunerProjector(embed_dim=192, num_tokens=32,
                               similarity_threshold=1.0, device=DEVICE)
    tokens = _patch(2, 64, 196, 192)
    with torch.no_grad():
        out = model(tokens, _etext(2, 192), 14, 14)
    assert out.shape == (2, 32, 192)
    mask = model.get_keep_mask()
    assert mask is not None
    # With random data, cosine sim between consecutive slices is essentially
    # never exactly 1.0, so all slices should be kept
    assert mask.all(), "threshold=1.0 should keep all slices on random data"


def test_medpruner_threshold_zero_prunes_aggressively() -> None:
    """threshold=0.0 marks almost every slice as redundant; output still (B,M,C)."""
    _skip_if_no_cuda()
    model = MedPrunerProjector(embed_dim=192, num_tokens=32,
                               similarity_threshold=0.0, device=DEVICE)
    with torch.no_grad():
        out = model(_patch(2, 64, 196, 192), _etext(2, 192), 14, 14)
    assert out.shape == (2, 32, 192)


# ── keep_mask ────────────────────────────────────────────────────────────────


def test_medpruner_keep_mask_shape(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    B, D = 2, 64
    medpruner(_patch(B, D, 196, 768), _etext(B), 14, 14)
    mask = medpruner.get_keep_mask()
    assert mask is not None
    assert mask.shape == (B, D), f"expected ({B}, {D}), got {mask.shape}"


def test_medpruner_keep_mask_dtype_bool(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    medpruner(_patch(2, 32, 196, 768), _etext(2), 14, 14)
    mask = medpruner.get_keep_mask()
    assert mask is not None
    assert mask.dtype == torch.bool


def test_medpruner_keep_mask_none_before_forward() -> None:
    _skip_if_no_cuda()
    model = MedPrunerProjector(embed_dim=192, num_tokens=16, device=DEVICE)
    assert model.get_keep_mask() is None


def test_medpruner_first_slice_always_kept() -> None:
    """Slice 0 is never pruned (no predecessor to compare against)."""
    _skip_if_no_cuda()
    model = MedPrunerProjector(embed_dim=192, num_tokens=16,
                               similarity_threshold=0.0, device=DEVICE)
    model(_patch(3, 32, 196, 192), _etext(3, 192), 14, 14)
    mask = model.get_keep_mask()
    assert mask is not None
    assert mask[:, 0].all(), "slice 0 must always be kept"


# ── Task-blind ────────────────────────────────────────────────────────────────


def test_medpruner_task_blind(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    tokens = _patch(2, 32, 196, 768)
    with torch.no_grad():
        out_a = medpruner(tokens, _etext(2), 14, 14)
        out_b = medpruner(tokens, _etext(2), 14, 14)
    assert torch.equal(out_a, out_b), "MedPruner must ignore etext"


# ── get_slice_attention returns None ──────────────────────────────────────────


def test_medpruner_slice_attention_is_none(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    assert medpruner.get_slice_attention() is None


# ── Device, dtype, finite ─────────────────────────────────────────────────────


def test_medpruner_output_on_cuda(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    out = medpruner(_patch(2, 32, 196, 768), _etext(2), 14, 14)
    assert out.device.type == "cuda"


def test_medpruner_output_dtype_float32(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    out = medpruner(_patch(2, 32, 196, 768), _etext(2), 14, 14)
    assert out.dtype == torch.float32


def test_medpruner_output_finite(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    out = medpruner(_patch(2, 32, 196, 768), _etext(2), 14, 14)
    assert torch.isfinite(out).all()


# ── Properties ────────────────────────────────────────────────────────────────


def test_medpruner_num_tokens_property(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    assert medpruner.num_tokens == 64


def test_medpruner_embed_dim_property(medpruner: MedPrunerProjector) -> None:
    _skip_if_no_cuda()
    assert medpruner.embed_dim == 768


# ═══════════════════════════════════════════════════════════════════════════════
# Drop-in interface contract
# ═══════════════════════════════════════════════════════════════════════════════


def test_drop_in_interface_contract() -> None:
    """All three projectors accept the same call signature and return (B, M, C)."""
    _skip_if_no_cuda()
    B, D, N, C, M = 2, 32, 1024, 768, 64
    patch_tokens = _patch(B, D, N, C)
    etext = _etext(B)

    results = {}
    for name, cls, extra in [
        ("Perceiver", PerceiverProjector, {"num_tokens": M}),
        ("MedPruner", MedPrunerProjector, {"num_tokens": M}),
        ("AADP",      AADPProjector,      {"num_tokens": M, "cond_dim": 768}),
    ]:
        proj = cls(embed_dim=C, **extra, device=DEVICE)
        with torch.no_grad():
            out = proj(patch_tokens, etext, H_patches=32, W_patches=32)
        assert out.shape == (B, M, C), (
            f"{name}: expected ({B}, {M}, {C}), got {out.shape}"
        )
        results[name] = out

    # All three return the right shape — no crashes, no signature mismatches
    assert len(results) == 3
