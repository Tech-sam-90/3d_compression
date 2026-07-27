"""Tests for AttentionConditionedInterSliceAggregator (attention-conditioned
ablation of the CT-CLIP Stage 2 aggregator — replaces FiLM entirely).
"""

import pytest
import torch

from aadp.ablations.attention_conditioned_ctclip_stage2 import (
    AttentionConditionedInterSliceAggregator,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


def _make_agg(C: int = 512, M: int = 64, cond_dim: int = 2560, top_k: int = 128) -> AttentionConditionedInterSliceAggregator:
    return AttentionConditionedInterSliceAggregator(
        embed_dim=C, num_tokens=M, num_heads=8, cond_dim=cond_dim,
        top_k=top_k, device=DEVICE,
    )


@pytest.fixture(scope="module")
def agg_base() -> AttentionConditionedInterSliceAggregator:
    _skip_if_no_cuda()
    return _make_agg(C=512, M=64, cond_dim=2560)


# ── Shape tests ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "B, D, K, C, M, cond_dim",
    [
        (2, 24, 576, 512, 64, 2560),   # main CT-CLIP training case
        (1, 24, 576, 512, 64, 2560),
        (2, 24, 576, 512, 16, 2560),   # M=16 ablation
        (2, 24, 576, 512, 128, 2560),  # M=128 ablation
    ],
)
def test_output_shape(B: int, D: int, K: int, C: int, M: int, cond_dim: int) -> None:
    _skip_if_no_cuda()
    model = AttentionConditionedInterSliceAggregator(
        embed_dim=C, num_tokens=M, num_heads=8, cond_dim=cond_dim, device=DEVICE
    )
    latents = torch.randn(B, D, K, C, device=DEVICE)
    etext = torch.randn(B, cond_dim, device=DEVICE)
    out = model(latents, etext)
    assert out.shape == (B, M, C), f"expected ({B}, {M}, {C}), got {out.shape}"


def test_output_dtype_float32(agg_base: AttentionConditionedInterSliceAggregator) -> None:
    _skip_if_no_cuda()
    latents = torch.randn(2, 24, 576, 512, device=DEVICE)
    etext = torch.randn(2, 2560, device=DEVICE)
    out = agg_base(latents, etext)
    assert out.dtype == torch.float32


def test_output_finite(agg_base: AttentionConditionedInterSliceAggregator) -> None:
    _skip_if_no_cuda()
    latents = torch.randn(2, 24, 576, 512, device=DEVICE)
    etext = torch.randn(2, 2560, device=DEVICE)
    out = agg_base(latents, etext)
    assert torch.isfinite(out).all()


def test_num_tokens_property(agg_base: AttentionConditionedInterSliceAggregator) -> None:
    assert agg_base.num_tokens == 64


def test_embed_dim_property(agg_base: AttentionConditionedInterSliceAggregator) -> None:
    assert agg_base.embed_dim == 512


# ── Attention weights / slice attention ────────────────────────────────────────


def test_attn_weights_shape_after_forward(agg_base: AttentionConditionedInterSliceAggregator) -> None:
    _skip_if_no_cuda()
    B, D, K = 2, 24, 576
    latents = torch.randn(B, D, K, 512, device=DEVICE)
    etext = torch.randn(B, 2560, device=DEVICE)
    agg_base(latents, etext)
    assert agg_base._last_attn_weights is not None
    assert agg_base._last_attn_weights.shape == (B, 64, D * K)


def test_get_slice_attention_shape() -> None:
    _skip_if_no_cuda()
    B, D, K = 2, 24, 576
    model = _make_agg()
    latents = torch.randn(B, D, K, 512, device=DEVICE)
    etext = torch.randn(B, 2560, device=DEVICE)
    model(latents, etext)
    sa = model.get_slice_attention(D, K)
    assert sa.shape == (B, D)
    assert (sa >= 0).all()


def test_get_slice_attention_raises_before_forward() -> None:
    _skip_if_no_cuda()
    model = _make_agg()
    with pytest.raises(RuntimeError, match="forward"):
        model.get_slice_attention(24, 576)


def test_attn_weights_sum_to_one() -> None:
    """Softmax property: attention weights over D*K sum to ~1 per query,
    even with top-K masking applied (masked positions get exactly 0)."""
    _skip_if_no_cuda()
    B, D, K = 2, 24, 576
    model = _make_agg(top_k=64)
    latents = torch.randn(B, D, K, 512, device=DEVICE)
    etext = torch.randn(B, 2560, device=DEVICE)
    model(latents, etext)
    w = model._last_attn_weights
    assert w is not None
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        f"max deviation from 1.0: {(row_sums - 1).abs().max().item()}"
    )


# ── Attention-based conditioning (replaces FiLM) ────────────────────────────────


def test_different_etext_different_output() -> None:
    """Different instruction embeddings must produce different outputs —
    the whole point of replacing FiLM with attention conditioning."""
    _skip_if_no_cuda()
    model = _make_agg()

    latents = torch.randn(2, 24, 576, 512, device=DEVICE)
    etext_a = torch.randn(2, 2560, device=DEVICE)
    etext_b = torch.randn(2, 2560, device=DEVICE)

    with torch.no_grad():
        out_a = model(latents, etext_a)
        out_b = model(latents, etext_b)

    assert not torch.allclose(out_a, out_b), (
        "attention conditioning should produce different outputs for different etext"
    )


def test_no_film_module_present() -> None:
    """Sanity check that this ablation genuinely has no FiLM submodule."""
    _skip_if_no_cuda()
    model = _make_agg()
    assert not hasattr(model, "film")
    assert not hasattr(model, "gamma_v_proj")
    assert not hasattr(model, "beta_v_proj")
    assert hasattr(model, "cond_cross_attn")
    assert hasattr(model, "text_proj")


# ── Forward ───────────────────────────────────────────────────────────────────


def test_forward_runs() -> None:
    _skip_if_no_cuda()
    model = _make_agg()
    latents = torch.randn(2, 24, 576, 512, device=DEVICE)
    etext = torch.randn(2, 2560, device=DEVICE)
    out = model(latents, etext)
    assert out.shape == (2, 64, 512)


def test_top_k_fallback_to_full_softmax_when_top_k_ge_n() -> None:
    """top_k >= N should not mask anything (mathematically full softmax)."""
    _skip_if_no_cuda()
    B, D, K = 1, 2, 4   # N = 8
    model = AttentionConditionedInterSliceAggregator(
        embed_dim=512, num_tokens=4, num_heads=8, cond_dim=2560, top_k=999, device=DEVICE,
    )
    latents = torch.randn(B, D, K, 512, device=DEVICE)
    etext = torch.randn(B, 2560, device=DEVICE)
    out = model(latents, etext)
    assert torch.isfinite(out).all()


# ── Gradients ─────────────────────────────────────────────────────────────────


def test_gradients_flow_through_depth_queries() -> None:
    _skip_if_no_cuda()
    model = _make_agg()
    latents = torch.randn(2, 24, 576, 512, device=DEVICE)
    etext = torch.randn(2, 2560, device=DEVICE)
    out = model(latents, etext)
    out.sum().backward()
    assert model.depth_queries.grad is not None


def test_gradients_flow_through_etext() -> None:
    _skip_if_no_cuda()
    model = _make_agg()
    latents = torch.randn(2, 24, 576, 512, device=DEVICE)
    etext = torch.randn(2, 2560, device=DEVICE, requires_grad=True)
    out = model(latents, etext)
    out.sum().backward()
    assert etext.grad is not None


def test_gradients_flow_through_cond_cross_attn() -> None:
    _skip_if_no_cuda()
    model = _make_agg()
    latents = torch.randn(2, 24, 576, 512, device=DEVICE)
    etext = torch.randn(2, 2560, device=DEVICE)
    out = model(latents, etext)
    out.sum().backward()
    assert model.text_proj.weight.grad is not None
    assert model.cond_cross_attn.in_proj_weight.grad is not None


def test_invalid_num_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        AttentionConditionedInterSliceAggregator(
            embed_dim=512, num_heads=7, device=DEVICE,
        )
