"""Tests for Phase 4 ablations. All on CUDA.

Covers:
    - AttentionConditionedStage2  (cross-attention conditioning)
    - AttentionConditionedAADP    (drop-in for AADPProjector)
    - AttentionAlignmentLoss      (KL divergence auxiliary loss)
    - CombinedLoss                (LM loss + optional alignment loss)
"""

import pytest
import torch
import torch.nn as nn

from aadp.ablations.attention_conditioned_stage2 import (
    AttentionConditionedStage2,
    AttentionConditionedAADP,
)
from aadp.ablations.auxiliary_attention_loss import (
    AttentionAlignmentLoss,
    CombinedLoss,
)
from aadp.models.projector.aadp import AADPProjector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


def _slice_latents(B: int, D: int, K: int, C: int) -> torch.Tensor:
    return torch.randn(B, D, K, C, device=DEVICE)


def _etext(B: int, cond_dim: int = 768) -> torch.Tensor:
    return torch.randn(B, cond_dim, device=DEVICE)


def _patch(B: int, D: int, N: int, C: int) -> torch.Tensor:
    return torch.randn(B, D, N, C, device=DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# AttentionConditionedStage2
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def stage2() -> AttentionConditionedStage2:
    _skip_if_no_cuda()
    return AttentionConditionedStage2(
        embed_dim=768, num_tokens=64, num_heads=8, cond_dim=768, device=DEVICE
    )


# ── Output shape ──────────────────────────────────────────────────────────────


def test_stage2_shape_standard(stage2: AttentionConditionedStage2) -> None:
    _skip_if_no_cuda()
    out = stage2(_slice_latents(2, 32, 16, 768), _etext(2))
    assert out.shape == (2, 64, 768), f"got {out.shape}"


def test_stage2_shape_deep_volume(stage2: AttentionConditionedStage2) -> None:
    _skip_if_no_cuda()
    out = stage2(_slice_latents(1, 303, 16, 768), _etext(1))
    assert out.shape == (1, 64, 768), f"got {out.shape}"


def test_stage2_shape_batch3(stage2: AttentionConditionedStage2) -> None:
    _skip_if_no_cuda()
    out = stage2(_slice_latents(3, 64, 8, 768), _etext(3))
    assert out.shape == (3, 64, 768), f"got {out.shape}"


def test_stage2_shape_different_num_tokens() -> None:
    _skip_if_no_cuda()
    m = AttentionConditionedStage2(
        embed_dim=192, num_tokens=32, num_heads=8, cond_dim=256, device=DEVICE
    )
    out = m(_slice_latents(2, 16, 8, 192), _etext(2, 256))
    assert out.shape == (2, 32, 192), f"got {out.shape}"


# ── Device and dtype ──────────────────────────────────────────────────────────


def test_stage2_output_on_cuda(stage2: AttentionConditionedStage2) -> None:
    _skip_if_no_cuda()
    out = stage2(_slice_latents(2, 16, 8, 768), _etext(2))
    assert out.device.type == "cuda"


def test_stage2_output_dtype_float32(stage2: AttentionConditionedStage2) -> None:
    _skip_if_no_cuda()
    out = stage2(_slice_latents(2, 16, 8, 768), _etext(2))
    assert out.dtype == torch.float32


def test_stage2_output_finite(stage2: AttentionConditionedStage2) -> None:
    _skip_if_no_cuda()
    out = stage2(_slice_latents(2, 16, 8, 768), _etext(2))
    assert torch.isfinite(out).all()


# ── get_slice_attention ───────────────────────────────────────────────────────


def test_stage2_slice_attention_shape(stage2: AttentionConditionedStage2) -> None:
    _skip_if_no_cuda()
    B, D, K = 2, 32, 16
    stage2(_slice_latents(B, D, K, 768), _etext(B))
    attn = stage2.get_slice_attention(D, K)
    assert attn.shape == (B, D), f"expected ({B}, {D}), got {attn.shape}"


def test_stage2_slice_attention_non_negative(stage2: AttentionConditionedStage2) -> None:
    _skip_if_no_cuda()
    stage2(_slice_latents(2, 16, 8, 768), _etext(2))
    attn = stage2.get_slice_attention(16, 8)
    assert (attn >= 0).all(), "attention weights must be non-negative"


def test_stage2_slice_attention_error_before_forward() -> None:
    _skip_if_no_cuda()
    m = AttentionConditionedStage2(
        embed_dim=192, num_tokens=16, num_heads=8, cond_dim=192, device=DEVICE
    )
    with pytest.raises(RuntimeError):
        m.get_slice_attention(8, 4)


# ── etext actually changes the output (not ignored like baselines) ────────────


def test_stage2_etext_matters() -> None:
    _skip_if_no_cuda()
    m = AttentionConditionedStage2(
        embed_dim=192, num_tokens=16, num_heads=8, cond_dim=192, device=DEVICE
    )
    tokens = _slice_latents(2, 16, 8, 192)
    etext_a = _etext(2, 192)
    etext_b = _etext(2, 192)
    with torch.no_grad():
        out_a = m(tokens, etext_a)
        out_b = m(tokens, etext_b)
    assert not torch.equal(out_a, out_b), \
        "Different etext must produce different outputs"


# ── Gradients ────────────────────────────────────────────────────────────────


def test_stage2_gradients_flow() -> None:
    _skip_if_no_cuda()
    m = AttentionConditionedStage2(
        embed_dim=192, num_tokens=16, num_heads=8, cond_dim=192, device=DEVICE
    )
    out = m(_slice_latents(2, 8, 4, 192), _etext(2, 192))
    out.sum().backward()
    assert m.depth_queries.grad is not None
    assert m.text_proj.weight.grad is not None


# ── Properties ────────────────────────────────────────────────────────────────


def test_stage2_num_tokens_property(stage2: AttentionConditionedStage2) -> None:
    _skip_if_no_cuda()
    assert stage2.num_tokens == 64


def test_stage2_embed_dim_property(stage2: AttentionConditionedStage2) -> None:
    _skip_if_no_cuda()
    assert stage2.embed_dim == 768


# ── Invalid embed_dim ─────────────────────────────────────────────────────────


def test_stage2_invalid_embed_raises() -> None:
    _skip_if_no_cuda()
    with pytest.raises(ValueError):
        AttentionConditionedStage2(
            embed_dim=100, num_tokens=8, num_heads=8, cond_dim=100, device=DEVICE
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AttentionConditionedAADP
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def ac_aadp() -> AttentionConditionedAADP:
    _skip_if_no_cuda()
    return AttentionConditionedAADP(
        embed_dim=768, num_latents=32, num_tokens=64,
        num_heads_stage1=8, num_heads_stage2=8, cond_dim=768, device=DEVICE,
    )


# ── Output shape ──────────────────────────────────────────────────────────────


def test_ac_aadp_shape_standard(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    out = ac_aadp(_patch(2, 32, 196, 768), _etext(2), 14, 14)
    assert out.shape == (2, 64, 768), f"got {out.shape}"


def test_ac_aadp_shape_deep_volume(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    out = ac_aadp(_patch(1, 303, 196, 768), _etext(1), 14, 14)
    assert out.shape == (1, 64, 768), f"got {out.shape}"


def test_ac_aadp_shape_batch1(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    out = ac_aadp(_patch(1, 16, 196, 768), _etext(1), 14, 14)
    assert out.shape == (1, 64, 768), f"got {out.shape}"


def test_ac_aadp_shape_custom_tokens() -> None:
    _skip_if_no_cuda()
    m = AttentionConditionedAADP(
        embed_dim=192, num_latents=8, num_tokens=32, cond_dim=256, device=DEVICE
    )
    out = m(_patch(2, 16, 49, 192), _etext(2, 256), 7, 7)
    assert out.shape == (2, 32, 192), f"got {out.shape}"


# ── Device and dtype ──────────────────────────────────────────────────────────


def test_ac_aadp_output_on_cuda(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    out = ac_aadp(_patch(2, 16, 196, 768), _etext(2), 14, 14)
    assert out.device.type == "cuda"


def test_ac_aadp_output_dtype_float32(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    out = ac_aadp(_patch(2, 16, 196, 768), _etext(2), 14, 14)
    assert out.dtype == torch.float32


def test_ac_aadp_output_finite(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    out = ac_aadp(_patch(2, 16, 196, 768), _etext(2), 14, 14)
    assert torch.isfinite(out).all()


# ── get_slice_attention ───────────────────────────────────────────────────────


def test_ac_aadp_slice_attention_shape(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    B, D = 2, 32
    ac_aadp(_patch(B, D, 196, 768), _etext(B), 14, 14)
    attn = ac_aadp.get_slice_attention()
    assert attn.shape == (B, D), f"expected ({B}, {D}), got {attn.shape}"


def test_ac_aadp_slice_attention_non_negative(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    ac_aadp(_patch(2, 16, 196, 768), _etext(2), 14, 14)
    attn = ac_aadp.get_slice_attention()
    assert (attn >= 0).all()


def test_ac_aadp_slice_attention_error_before_forward() -> None:
    _skip_if_no_cuda()
    m = AttentionConditionedAADP(
        embed_dim=192, num_latents=8, num_tokens=16, cond_dim=192, device=DEVICE
    )
    with pytest.raises(RuntimeError):
        m.get_slice_attention()



# ── use_film kwarg silently ignored ──────────────────────────────────────────


def test_ac_aadp_use_film_ignored() -> None:
    """use_film=False must be accepted without error (signature parity)."""
    _skip_if_no_cuda()
    m = AttentionConditionedAADP(
        embed_dim=192, num_latents=8, num_tokens=16, cond_dim=192,
        use_film=False, device=DEVICE,
    )
    out = m(_patch(2, 8, 49, 192), _etext(2, 192), 7, 7)
    assert out.shape == (2, 16, 192)


# ── num_parameters ────────────────────────────────────────────────────────────


def test_ac_aadp_num_parameters_positive(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    params = ac_aadp.num_parameters()
    assert params["stage1"] > 0
    assert params["stage2"] > 0
    assert params["total"] == params["stage1"] + params["stage2"]


# ── Properties ────────────────────────────────────────────────────────────────


def test_ac_aadp_num_tokens_property(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    assert ac_aadp.num_tokens == 64


def test_ac_aadp_num_latents_property(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    assert ac_aadp.num_latents == 32


def test_ac_aadp_embed_dim_property(ac_aadp: AttentionConditionedAADP) -> None:
    _skip_if_no_cuda()
    assert ac_aadp.embed_dim == 768


# ── Gradients ────────────────────────────────────────────────────────────────


def test_ac_aadp_gradients_flow() -> None:
    _skip_if_no_cuda()
    m = AttentionConditionedAADP(
        embed_dim=192, num_latents=8, num_tokens=16, cond_dim=192, device=DEVICE
    )
    out = m(_patch(2, 8, 49, 192), _etext(2, 192), 7, 7)
    out.sum().backward()
    assert m.stage1.latents.grad is not None
    assert m.stage2.depth_queries.grad is not None
    assert m.stage2.text_proj.weight.grad is not None


# ── Drop-in contract with AADPProjector ───────────────────────────────────────


def test_ac_aadp_drop_in_contract() -> None:
    """AttentionConditionedAADP and AADPProjector accept identical call signature."""
    _skip_if_no_cuda()
    B, D, N, C, M = 2, 16, 196, 768, 32
    patch_tokens = _patch(B, D, N, C)
    etext = _etext(B, 768)

    for name, cls in [
        ("AADPProjector", AADPProjector),
        ("AttentionConditionedAADP", AttentionConditionedAADP),
    ]:
        m = cls(
            embed_dim=C, num_latents=16, num_tokens=M, cond_dim=768, device=DEVICE
        )
        with torch.no_grad():
            out = m(patch_tokens, etext, 14, 14)
        assert out.shape == (B, M, C), f"{name}: expected ({B}, {M}, {C}), got {out.shape}"


# ═══════════════════════════════════════════════════════════════════════════════
# AttentionAlignmentLoss
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def align_loss() -> AttentionAlignmentLoss:
    _skip_if_no_cuda()
    return AttentionAlignmentLoss(lambda_attn=0.1)


def _attn(B: int, D: int, uniform: bool = False) -> torch.Tensor:
    if uniform:
        return torch.ones(B, D, device=DEVICE) / D
    return torch.randn(B, D, device=DEVICE)


# ── Scalar output ─────────────────────────────────────────────────────────────


def test_align_loss_is_scalar(align_loss: AttentionAlignmentLoss) -> None:
    _skip_if_no_cuda()
    loss = align_loss(_attn(2, 32), [[0, 1, 2], [5, 6]])
    assert loss.shape == torch.Size([])


def test_align_loss_on_cuda(align_loss: AttentionAlignmentLoss) -> None:
    _skip_if_no_cuda()
    loss = align_loss(_attn(2, 32), [[0, 1], [10, 11]])
    assert loss.device.type == "cuda"


def test_align_loss_is_non_negative(align_loss: AttentionAlignmentLoss) -> None:
    _skip_if_no_cuda()
    loss = align_loss(_attn(2, 32), [[0, 1, 2], [5]])
    assert loss.item() >= 0.0


def test_align_loss_finite(align_loss: AttentionAlignmentLoss) -> None:
    _skip_if_no_cuda()
    loss = align_loss(_attn(2, 32), [[0, 1], [10]])
    assert torch.isfinite(loss)


# ── Lambda scaling ────────────────────────────────────────────────────────────


def test_align_loss_lambda_scales_output() -> None:
    _skip_if_no_cuda()
    weights = _attn(2, 32)
    gt = [[0, 1, 2], [5, 6]]
    loss_01 = AttentionAlignmentLoss(lambda_attn=0.1)(weights, gt).item()
    loss_10 = AttentionAlignmentLoss(lambda_attn=1.0)(weights, gt).item()
    assert abs(loss_10 / loss_01 - 10.0) < 1e-4, \
        f"Expected 10x scaling, got {loss_10 / loss_01}"


# ── Empty GT indices → zero loss ──────────────────────────────────────────────


def test_align_loss_empty_gt_zero(align_loss: AttentionAlignmentLoss) -> None:
    _skip_if_no_cuda()
    loss = align_loss(_attn(2, 32), [[], []])
    assert loss.item() == pytest.approx(0.0)


# ── Mixed empty and non-empty ─────────────────────────────────────────────────


def test_align_loss_mixed_batch(align_loss: AttentionAlignmentLoss) -> None:
    _skip_if_no_cuda()
    loss = align_loss(_attn(3, 32), [[], [5, 10], []])
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


# ── Out-of-range indices are silently ignored ─────────────────────────────────


def test_align_loss_out_of_range_indices(align_loss: AttentionAlignmentLoss) -> None:
    _skip_if_no_cuda()
    # index 999 is out of range for D=32 — should not crash
    loss = align_loss(_attn(1, 32), [[0, 999]])
    assert torch.isfinite(loss)


def test_align_loss_all_out_of_range_zero(align_loss: AttentionAlignmentLoss) -> None:
    _skip_if_no_cuda()
    # All indices out of range → treated as empty → zero loss
    loss = align_loss(_attn(1, 32), [[999, 1000]])
    assert loss.item() == pytest.approx(0.0)


# ── Gradient flows through attn_weights ───────────────────────────────────────


def test_align_loss_gradient_flows() -> None:
    _skip_if_no_cuda()
    weights = torch.randn(2, 32, device=DEVICE, requires_grad=True)
    loss = AttentionAlignmentLoss()(weights, [[0, 1], [10]])
    loss.backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


# ── Perfect prediction → lower loss than random ───────────────────────────────


def test_align_loss_peaked_lower_than_uniform() -> None:
    _skip_if_no_cuda()
    D = 32
    gt = [[5]]
    fn = AttentionAlignmentLoss()

    # Very peaked at the correct slice
    peaked = torch.full((1, D), -100.0, device=DEVICE)
    peaked[0, 5] = 100.0

    # Uniform distribution
    uniform = torch.zeros(1, D, device=DEVICE)

    loss_peaked = fn(peaked, gt).item()
    loss_uniform = fn(uniform, gt).item()
    assert loss_peaked < loss_uniform, \
        "Peaked prediction at GT should have lower KL than uniform"


# ═══════════════════════════════════════════════════════════════════════════════
# CombinedLoss
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def combined() -> CombinedLoss:
    _skip_if_no_cuda()
    return CombinedLoss(lambda_attn=0.1)


def _lm_loss() -> torch.Tensor:
    return torch.tensor(2.5, device=DEVICE)


# ── Scalar output ─────────────────────────────────────────────────────────────


def test_combined_is_scalar(combined: CombinedLoss) -> None:
    _skip_if_no_cuda()
    loss = combined(_lm_loss(), _attn(2, 32), [[0, 1], [5]], use_attn_loss=True)
    assert loss.shape == torch.Size([])


def test_combined_on_cuda(combined: CombinedLoss) -> None:
    _skip_if_no_cuda()
    loss = combined(_lm_loss(), _attn(2, 32), [[0, 1], [5]], use_attn_loss=True)
    assert loss.device.type == "cuda"


# ── use_attn_loss=False → identical to lm_loss ────────────────────────────────


def test_combined_no_attn_loss_equals_lm_loss(combined: CombinedLoss) -> None:
    _skip_if_no_cuda()
    lm = _lm_loss()
    loss = combined(lm, _attn(2, 32), [[0, 1], [5]], use_attn_loss=False)
    assert loss.item() == pytest.approx(lm.item())


# ── attn_weights=None → identical to lm_loss ──────────────────────────────────


def test_combined_attn_weights_none_equals_lm_loss(combined: CombinedLoss) -> None:
    _skip_if_no_cuda()
    lm = _lm_loss()
    loss = combined(lm, attn_weights=None, gt_slice_indices=[[0]], use_attn_loss=True)
    assert loss.item() == pytest.approx(lm.item())


# ── gt_slice_indices=None → identical to lm_loss ─────────────────────────────


def test_combined_gt_none_equals_lm_loss(combined: CombinedLoss) -> None:
    _skip_if_no_cuda()
    lm = _lm_loss()
    loss = combined(lm, _attn(2, 32), gt_slice_indices=None, use_attn_loss=True)
    assert loss.item() == pytest.approx(lm.item())


# ── With attn loss, total > lm_loss (positive KL) ────────────────────────────


def test_combined_with_attn_loss_greater_than_lm(combined: CombinedLoss) -> None:
    _skip_if_no_cuda()
    lm = _lm_loss()
    attn = _attn(2, 32)
    gt = [[0, 1, 2], [10, 11]]
    with_attn = combined(lm, attn, gt, use_attn_loss=True)
    without_attn = combined(lm, attn, gt, use_attn_loss=False)
    # KL is non-negative, so combined >= lm_loss
    assert with_attn.item() >= without_attn.item() - 1e-6


# ── Finite output ─────────────────────────────────────────────────────────────


def test_combined_finite(combined: CombinedLoss) -> None:
    _skip_if_no_cuda()
    loss = combined(_lm_loss(), _attn(2, 32), [[0, 1], [5]], use_attn_loss=True)
    assert torch.isfinite(loss)


# ── Gradient flows from both terms ───────────────────────────────────────────


def test_combined_gradient_flows() -> None:
    _skip_if_no_cuda()
    attn = torch.randn(2, 32, device=DEVICE, requires_grad=True)
    lm_base = torch.tensor(2.5, device=DEVICE, requires_grad=True)
    loss = CombinedLoss(lambda_attn=0.1)(
        lm_base, attn, [[0, 1], [10]], use_attn_loss=True
    )
    loss.backward()
    assert lm_base.grad is not None
    assert attn.grad is not None


# ── Empty GT does not add to loss ────────────────────────────────────────────


def test_combined_empty_gt_equals_lm_loss(combined: CombinedLoss) -> None:
    _skip_if_no_cuda()
    lm = _lm_loss()
    loss = combined(lm, _attn(2, 32), [[], []], use_attn_loss=True)
    assert loss.item() == pytest.approx(lm.item(), abs=1e-5)
