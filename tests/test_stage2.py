"""Tests for InterSliceAggregator (Stage 2 of A-ADP). All on CUDA."""

import pytest
import torch

from aadp.models.projector.stage2 import InterSliceAggregator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


def _make_agg(C: int = 768, M: int = 64, cond_dim: int = 768, use_film: bool = True) -> InterSliceAggregator:
    return InterSliceAggregator(
        embed_dim=C, num_tokens=M, num_heads=8, cond_dim=cond_dim,
        use_film=use_film, device=DEVICE,
    )


# ── Module-scoped fixture (main training config) ──────────────────────────────


@pytest.fixture(scope="module")
def agg_base() -> InterSliceAggregator:
    _skip_if_no_cuda()
    return _make_agg(C=768, M=64, cond_dim=768)


# ── Shape tests ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "B, D, K, C, M, cond_dim",
    [
        (2, 128, 32, 768, 64, 768),    # main training case
        (1, 303, 32, 768, 64, 768),    # full CT-RATE depth
        (2, 128, 32, 768, 16, 768),    # M=16 ablation
        (2, 128, 32, 768, 128, 768),   # M=128 ablation
        (2, 128, 16, 768, 64, 768),    # K=16 ablation
    ],
)
def test_output_shape(B: int, D: int, K: int, C: int, M: int, cond_dim: int) -> None:
    _skip_if_no_cuda()
    model = InterSliceAggregator(
        embed_dim=C, num_tokens=M, num_heads=8, cond_dim=cond_dim, device=DEVICE
    )
    latents = torch.randn(B, D, K, C, device=DEVICE)
    etext = torch.randn(B, cond_dim, device=DEVICE)
    out = model(latents, etext)
    assert out.shape == (B, M, C), f"expected ({B}, {M}, {C}), got {out.shape}"


# ── Device & dtype ────────────────────────────────────────────────────────────


def test_output_on_cuda(agg_base: InterSliceAggregator) -> None:
    _skip_if_no_cuda()
    latents = torch.randn(2, 128, 32, 768, device=DEVICE)
    etext = torch.randn(2, 768, device=DEVICE)
    out = agg_base(latents, etext)
    assert out.device.type == "cuda"


def test_output_dtype_float32(agg_base: InterSliceAggregator) -> None:
    _skip_if_no_cuda()
    latents = torch.randn(2, 128, 32, 768, device=DEVICE)
    etext = torch.randn(2, 768, device=DEVICE)
    out = agg_base(latents, etext)
    assert out.dtype == torch.float32


def test_output_finite(agg_base: InterSliceAggregator) -> None:
    _skip_if_no_cuda()
    latents = torch.randn(2, 128, 32, 768, device=DEVICE)
    etext = torch.randn(2, 768, device=DEVICE)
    out = agg_base(latents, etext)
    assert torch.isfinite(out).all()


# ── Properties ────────────────────────────────────────────────────────────────


def test_num_tokens_property(agg_base: InterSliceAggregator) -> None:
    _skip_if_no_cuda()
    assert agg_base.num_tokens == 64


def test_embed_dim_property(agg_base: InterSliceAggregator) -> None:
    _skip_if_no_cuda()
    assert agg_base.embed_dim == 768


# ── Attention weights ─────────────────────────────────────────────────────────


def test_attn_weights_shape_after_forward(agg_base: InterSliceAggregator) -> None:
    _skip_if_no_cuda()
    B, D, K = 2, 128, 32
    latents = torch.randn(B, D, K, 768, device=DEVICE)
    etext = torch.randn(B, 768, device=DEVICE)
    agg_base(latents, etext)
    assert agg_base._last_attn_weights is not None
    assert agg_base._last_attn_weights.shape == (B, 64, D * K), (
        f"got {agg_base._last_attn_weights.shape}"
    )


def test_get_slice_attention_shape() -> None:
    _skip_if_no_cuda()
    B, D, K = 2, 64, 32
    model = _make_agg()
    latents = torch.randn(B, D, K, 768, device=DEVICE)
    etext = torch.randn(B, 768, device=DEVICE)
    model(latents, etext)
    sa = model.get_slice_attention(D, K)
    assert sa.shape == (B, D), f"expected ({B}, {D}), got {sa.shape}"


def test_get_slice_attention_raises_before_forward() -> None:
    _skip_if_no_cuda()
    model = _make_agg()
    with pytest.raises(RuntimeError, match="forward"):
        model.get_slice_attention(128, 32)


def test_attn_weights_sum_to_one() -> None:
    """Softmax property: attention weights over D*K sum to ~1 per query."""
    _skip_if_no_cuda()
    B, D, K = 2, 32, 16
    model = _make_agg()
    latents = torch.randn(B, D, K, 768, device=DEVICE)
    etext = torch.randn(B, 768, device=DEVICE)
    model(latents, etext)
    w = model._last_attn_weights  # (B, M, D*K)
    assert w is not None
    row_sums = w.sum(dim=-1)      # (B, M) — should be all ~1.0
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        f"max deviation from 1.0: {(row_sums - 1).abs().max().item()}"
    )


# ── FiLM conditioning ─────────────────────────────────────────────────────────


def test_film_on_different_etext_different_output() -> None:
    """use_film=True: different instruction embeddings produce different outputs."""
    _skip_if_no_cuda()
    model = _make_agg(use_film=True)
    # Perturb FiLM weights away from identity init so conditioning is visible
    with torch.no_grad():
        model.film.gamma_proj.weight.normal_(std=0.1)   # type: ignore[union-attr]
        model.film.beta_proj.weight.normal_(std=0.1)    # type: ignore[union-attr]

    latents = torch.randn(2, 64, 32, 768, device=DEVICE)
    etext_a = torch.randn(2, 768, device=DEVICE)
    etext_b = torch.randn(2, 768, device=DEVICE)

    with torch.no_grad():
        out_a = model(latents, etext_a)
        out_b = model(latents, etext_b)

    assert not torch.allclose(out_a, out_b), "FiLM should produce different outputs for different etext"


def test_null_film_same_output_for_different_etext() -> None:
    """use_film=False: etext is ignored, outputs are identical."""
    _skip_if_no_cuda()
    model = _make_agg(use_film=False)

    latents = torch.randn(2, 64, 32, 768, device=DEVICE)
    etext_a = torch.randn(2, 768, device=DEVICE)
    etext_b = torch.randn(2, 768, device=DEVICE)

    with torch.no_grad():
        out_a = model(latents, etext_a)
        out_b = model(latents, etext_b)

    assert torch.equal(out_a, out_b), "NullFiLM should produce identical outputs regardless of etext"


# ── Depth spacing ─────────────────────────────────────────────────────────────


def test_depth_spacing_none_runs() -> None:
    _skip_if_no_cuda()
    model = _make_agg()
    latents = torch.randn(2, 64, 32, 768, device=DEVICE)
    etext = torch.randn(2, 768, device=DEVICE)
    out = model(latents, etext, depth_spacing_mm=None)
    assert out.shape == (2, 64, 768)


def test_depth_spacing_float_runs() -> None:
    _skip_if_no_cuda()
    model = _make_agg()
    latents = torch.randn(2, 64, 32, 768, device=DEVICE)
    etext = torch.randn(2, 768, device=DEVICE)
    out = model(latents, etext, depth_spacing_mm=3.0)
    assert out.shape == (2, 64, 768)


# ── Gradients ─────────────────────────────────────────────────────────────────


def test_gradients_flow_through_depth_queries() -> None:
    _skip_if_no_cuda()
    model = _make_agg()
    latents = torch.randn(2, 64, 32, 768, device=DEVICE)
    etext = torch.randn(2, 768, device=DEVICE)
    out = model(latents, etext)
    out.sum().backward()
    assert model.depth_queries.grad is not None


def test_gradients_flow_through_etext_with_film() -> None:
    _skip_if_no_cuda()
    model = _make_agg(use_film=True)
    latents = torch.randn(2, 64, 32, 768, device=DEVICE)
    etext = torch.randn(2, 768, device=DEVICE, requires_grad=True)
    out = model(latents, etext)
    out.sum().backward()
    assert etext.grad is not None
