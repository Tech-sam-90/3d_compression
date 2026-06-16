"""Tests for FiLMLayer and NullFiLMLayer. All on CUDA."""

import pytest
import torch
import torch.nn as nn

from aadp.models.film import FiLMLayer, NullFiLMLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


# ── FiLMLayer fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def film_square() -> FiLMLayer:
    """cond_dim == target_dim == 768."""
    _skip_if_no_cuda()
    return FiLMLayer(cond_dim=768, target_dim=768, device=DEVICE)


# ── FiLMLayer — shape ─────────────────────────────────────────────────────────


def test_film_output_shape_square(film_square: FiLMLayer) -> None:
    _skip_if_no_cuda()
    x = torch.randn(2, 64, 768, device=DEVICE)
    cond = torch.randn(2, 768, device=DEVICE)
    out = film_square(x, cond)
    assert out.shape == (2, 64, 768), f"got {out.shape}"


def test_film_output_shape_mismatched_dims() -> None:
    """cond_dim=512, target_dim=768 — projection handles the mismatch."""
    _skip_if_no_cuda()
    model = FiLMLayer(cond_dim=512, target_dim=768, device=DEVICE)
    x = torch.randn(2, 64, 768, device=DEVICE)
    cond = torch.randn(2, 512, device=DEVICE)
    out = model(x, cond)
    assert out.shape == (2, 64, 768), f"got {out.shape}"


# ── FiLMLayer — device & dtype ────────────────────────────────────────────────


def test_film_output_on_cuda(film_square: FiLMLayer) -> None:
    _skip_if_no_cuda()
    x = torch.randn(2, 64, 768, device=DEVICE)
    cond = torch.randn(2, 768, device=DEVICE)
    out = film_square(x, cond)
    assert out.device.type == "cuda"


def test_film_output_dtype_float32(film_square: FiLMLayer) -> None:
    _skip_if_no_cuda()
    x = torch.randn(2, 64, 768, device=DEVICE)
    cond = torch.randn(2, 768, device=DEVICE)
    out = film_square(x, cond)
    assert out.dtype == torch.float32


# ── FiLMLayer — finite values ─────────────────────────────────────────────────


def test_film_output_finite(film_square: FiLMLayer) -> None:
    _skip_if_no_cuda()
    x = torch.randn(2, 64, 768, device=DEVICE)
    cond = torch.randn(2, 768, device=DEVICE)
    out = film_square(x, cond)
    assert torch.isfinite(out).all()


# ── FiLMLayer — identity initialisation ──────────────────────────────────────


def test_film_identity_at_init() -> None:
    """Immediately after construction gamma=1, beta=0, so out == x."""
    _skip_if_no_cuda()
    model = FiLMLayer(cond_dim=768, target_dim=768, device=DEVICE)
    x = torch.randn(2, 64, 768, device=DEVICE)
    cond = torch.randn(2, 768, device=DEVICE)
    with torch.no_grad():
        out = model(x, cond)
    assert torch.allclose(out, x, atol=1e-6), (
        f"max deviation at init: {(out - x).abs().max().item()}"
    )


# ── FiLMLayer — modulation after a gradient step ──────────────────────────────


def test_film_modulates_after_update() -> None:
    """After one random gradient step FiLM deviates from identity."""
    _skip_if_no_cuda()
    model = FiLMLayer(cond_dim=768, target_dim=768, device=DEVICE)
    x = torch.randn(2, 64, 768, device=DEVICE)
    cond = torch.randn(2, 768, device=DEVICE)

    # One random weight update
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    loss = model(x, cond).sum()
    loss.backward()
    opt.step()

    with torch.no_grad():
        out = model(x, cond)
    assert not torch.allclose(out, x, atol=1e-6)


# ── FiLMLayer — gradients ─────────────────────────────────────────────────────


def test_film_gradients_flow_through_cond() -> None:
    """Instruction embedding receives gradients through FiLM."""
    _skip_if_no_cuda()
    model = FiLMLayer(cond_dim=768, target_dim=768, device=DEVICE)
    x = torch.randn(2, 64, 768, device=DEVICE)
    cond = torch.randn(2, 768, device=DEVICE, requires_grad=True)
    out = model(x, cond)
    out.sum().backward()
    assert cond.grad is not None


def test_film_gradients_flow_through_x() -> None:
    """Depth queries receive gradients through FiLM."""
    _skip_if_no_cuda()
    model = FiLMLayer(cond_dim=768, target_dim=768, device=DEVICE)
    x = torch.randn(2, 64, 768, device=DEVICE, requires_grad=True)
    cond = torch.randn(2, 768, device=DEVICE)
    out = model(x, cond)
    out.sum().backward()
    assert x.grad is not None


# ── FiLMLayer — learnable parameters ─────────────────────────────────────────


def test_film_gamma_proj_requires_grad(film_square: FiLMLayer) -> None:
    _skip_if_no_cuda()
    assert film_square.gamma_proj.weight.requires_grad
    assert film_square.gamma_proj.bias.requires_grad


def test_film_beta_proj_requires_grad(film_square: FiLMLayer) -> None:
    _skip_if_no_cuda()
    assert film_square.beta_proj.weight.requires_grad
    assert film_square.beta_proj.bias.requires_grad


# ── FiLMLayer — conditioning effectiveness ───────────────────────────────────


def test_film_different_cond_different_output() -> None:
    """Different cond vectors produce different outputs for the same x."""
    _skip_if_no_cuda()
    model = FiLMLayer(cond_dim=768, target_dim=768, device=DEVICE)
    # Perturb weights so the layer is not identity
    with torch.no_grad():
        model.gamma_proj.weight.normal_(std=0.1)
        model.beta_proj.weight.normal_(std=0.1)

    x = torch.randn(2, 64, 768, device=DEVICE)
    cond_a = torch.randn(2, 768, device=DEVICE)
    cond_b = torch.randn(2, 768, device=DEVICE)

    with torch.no_grad():
        out_a = model(x, cond_a)
        out_b = model(x, cond_b)

    assert not torch.allclose(out_a, out_b)


# ── FiLMLayer — batch independence ───────────────────────────────────────────


def test_film_batch_independence() -> None:
    """Sample 0 gives the same result when processed alone vs. in a batch."""
    _skip_if_no_cuda()
    model = FiLMLayer(cond_dim=768, target_dim=768, device=DEVICE)
    with torch.no_grad():
        model.gamma_proj.weight.normal_(std=0.1)
        model.beta_proj.weight.normal_(std=0.1)

    x = torch.randn(3, 32, 768, device=DEVICE)
    cond = torch.randn(3, 768, device=DEVICE)

    with torch.no_grad():
        out_batch = model(x, cond)
        out_single = model(x[:1], cond[:1])

    assert torch.allclose(out_batch[:1], out_single, atol=1e-5)


# ── NullFiLMLayer ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def null_film() -> NullFiLMLayer:
    _skip_if_no_cuda()
    return NullFiLMLayer(cond_dim=768, target_dim=768, device=DEVICE)


def test_null_film_output_equals_input(null_film: NullFiLMLayer) -> None:
    _skip_if_no_cuda()
    x = torch.randn(2, 64, 768, device=DEVICE)
    cond = torch.randn(2, 768, device=DEVICE)
    out = null_film(x, cond)
    assert torch.equal(out, x)


def test_null_film_output_shape(null_film: NullFiLMLayer) -> None:
    _skip_if_no_cuda()
    x = torch.randn(2, 64, 768, device=DEVICE)
    cond = torch.randn(2, 768, device=DEVICE)
    out = null_film(x, cond)
    assert out.shape == x.shape


def test_null_film_no_parameters(null_film: NullFiLMLayer) -> None:
    _skip_if_no_cuda()
    assert list(null_film.parameters()) == []


def test_null_film_ignores_cond_shape() -> None:
    """NullFiLMLayer should accept any cond tensor without error."""
    _skip_if_no_cuda()
    model = NullFiLMLayer(cond_dim=768, target_dim=768, device=DEVICE)
    x = torch.randn(2, 64, 768, device=DEVICE)
    # Pass a deliberately wrong-shape cond — should be ignored silently
    cond_wrong = torch.randn(2, 1234, device=DEVICE)
    out = model(x, cond_wrong)
    assert torch.equal(out, x)


def test_null_film_drop_in_for_film() -> None:
    """Same call signature as FiLMLayer works unchanged."""
    _skip_if_no_cuda()
    for cls in (FiLMLayer, NullFiLMLayer):
        model = cls(cond_dim=512, target_dim=256, device=DEVICE)
        x = torch.randn(4, 16, 256, device=DEVICE)
        cond = torch.randn(4, 512, device=DEVICE)
        out = model(x, cond)
        assert out.shape == (4, 16, 256)
