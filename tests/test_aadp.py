"""Tests for AADPProjector (full two-stage pipeline). All on CUDA."""

import pytest
import torch

from aadp.models.projector.aadp import AADPProjector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


def _make_proj(
    C: int = 768,
    K: int = 32,
    M: int = 64,
    cond_dim: int = 768,
    use_film: bool = True,
) -> AADPProjector:
    return AADPProjector(
        embed_dim=C,
        num_latents=K,
        num_tokens=M,
        cond_dim=cond_dim,
        use_film=use_film,
        device=DEVICE,
    )


# ── Module-scoped fixture (main training config) ──────────────────────────────


@pytest.fixture(scope="module")
def proj_base() -> AADPProjector:
    _skip_if_no_cuda()
    return _make_proj()


# ── End-to-end shape tests ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "B, D, N, C, K, M, H, W, cond_dim",
    [
        (2, 128, 1024, 768, 32,  64, 32, 32, 768),  # main training case
        (1, 303, 1024, 768, 32,  64, 32, 32, 768),  # full CT-RATE depth
        (2, 128,  196, 192, 32,  64, 14, 14, 192),  # ViT-Tiny at 224x224
        (2, 128, 1024, 768, 16,  64, 32, 32, 768),  # K=16 ablation
        (2, 128, 1024, 768, 64,  64, 32, 32, 768),  # K=64 ablation
        (2, 128, 1024, 768, 32,  16, 32, 32, 768),  # M=16 ablation
        (2, 128, 1024, 768, 32, 128, 32, 32, 768),  # M=128 ablation
    ],
)
def test_output_shape(
    B: int, D: int, N: int, C: int, K: int, M: int, H: int, W: int, cond_dim: int
) -> None:
    _skip_if_no_cuda()
    proj = AADPProjector(
        embed_dim=C, num_latents=K, num_tokens=M, cond_dim=cond_dim, device=DEVICE
    )
    patch_tokens = torch.randn(B, D, N, C, device=DEVICE)
    etext = torch.randn(B, cond_dim, device=DEVICE)
    out = proj(patch_tokens, etext, H, W)
    assert out.shape == (B, M, C), f"expected ({B}, {M}, {C}), got {out.shape}"


# ── Reshaping correctness: Stage 1 receives (B*D, N, C) ──────────────────────


def test_stage1_receives_correct_shape() -> None:
    """Stage 1 should see (B*D, N, C), not (B, D, N, C)."""
    _skip_if_no_cuda()
    B, D, N, C = 2, 32, 196, 192
    proj = _make_proj(C=C, cond_dim=C)

    recorded: list = []
    original_forward = proj.stage1.forward

    def patched_forward(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        recorded.append(x.shape)
        return original_forward(x, H, W)

    proj.stage1.forward = patched_forward  # type: ignore[method-assign]

    patch_tokens = torch.randn(B, D, N, C, device=DEVICE)
    etext = torch.randn(B, C, device=DEVICE)
    proj(patch_tokens, etext, 14, 14)

    proj.stage1.forward = original_forward  # type: ignore[method-assign]

    assert len(recorded) == 1
    assert recorded[0] == torch.Size([B * D, N, C]), (
        f"Stage 1 received {recorded[0]}, expected ({B*D}, {N}, {C})"
    )


# ── Attention access ──────────────────────────────────────────────────────────


def test_get_slice_attention_shape(proj_base: AADPProjector) -> None:
    _skip_if_no_cuda()
    B, D = 2, 64
    patch_tokens = torch.randn(B, D, 196, 768, device=DEVICE)
    etext = torch.randn(B, 768, device=DEVICE)
    proj_base(patch_tokens, etext, 14, 14)
    sa = proj_base.get_slice_attention()
    assert sa.shape == (B, D), f"expected ({B}, {D}), got {sa.shape}"


def test_get_slice_attention_raises_before_forward() -> None:
    _skip_if_no_cuda()
    proj = _make_proj()
    with pytest.raises(RuntimeError, match="forward"):
        proj.get_slice_attention()


# ── FiLM toggle ───────────────────────────────────────────────────────────────


def test_film_on_different_etext_different_output() -> None:
    """use_film=True: different instructions → different LLM tokens."""
    _skip_if_no_cuda()
    proj = _make_proj(use_film=True)
    # Perturb FiLM weights away from identity so conditioning is visible
    with torch.no_grad():
        proj.stage2.film.gamma_proj.weight.normal_(std=0.1)   # type: ignore[union-attr]
        proj.stage2.film.beta_proj.weight.normal_(std=0.1)    # type: ignore[union-attr]

    patch_tokens = torch.randn(2, 32, 196, 768, device=DEVICE)
    etext_a = torch.randn(2, 768, device=DEVICE)
    etext_b = torch.randn(2, 768, device=DEVICE)

    with torch.no_grad():
        out_a = proj(patch_tokens, etext_a, 14, 14)
        out_b = proj(patch_tokens, etext_b, 14, 14)

    assert not torch.allclose(out_a, out_b)


def test_null_film_same_output_for_different_etext() -> None:
    """use_film=False: etext is ignored → identical outputs."""
    _skip_if_no_cuda()
    proj = _make_proj(use_film=False)

    patch_tokens = torch.randn(2, 32, 196, 768, device=DEVICE)
    etext_a = torch.randn(2, 768, device=DEVICE)
    etext_b = torch.randn(2, 768, device=DEVICE)

    with torch.no_grad():
        out_a = proj(patch_tokens, etext_a, 14, 14)
        out_b = proj(patch_tokens, etext_b, 14, 14)

    assert torch.equal(out_a, out_b)


# ── num_parameters() ─────────────────────────────────────────────────────────


def test_num_parameters_keys(proj_base: AADPProjector) -> None:
    _skip_if_no_cuda()
    params = proj_base.num_parameters()
    assert set(params.keys()) == {"stage1", "stage2", "total"}


def test_num_parameters_total_equals_sum(proj_base: AADPProjector) -> None:
    _skip_if_no_cuda()
    params = proj_base.num_parameters()
    assert params["total"] == params["stage1"] + params["stage2"]


def test_num_parameters_all_positive(proj_base: AADPProjector) -> None:
    _skip_if_no_cuda()
    params = proj_base.num_parameters()
    assert all(v > 0 for v in params.values())


# ── Gradients ─────────────────────────────────────────────────────────────────


def test_gradients_flow_to_stage1_latents() -> None:
    _skip_if_no_cuda()
    proj = _make_proj(C=192, cond_dim=192)
    patch_tokens = torch.randn(2, 16, 196, 192, device=DEVICE)
    etext = torch.randn(2, 192, device=DEVICE)
    out = proj(patch_tokens, etext, 14, 14)
    out.sum().backward()
    assert proj.stage1.latents.grad is not None


def test_gradients_flow_to_stage2_depth_queries() -> None:
    _skip_if_no_cuda()
    proj = _make_proj(C=192, cond_dim=192)
    patch_tokens = torch.randn(2, 16, 196, 192, device=DEVICE)
    etext = torch.randn(2, 192, device=DEVICE)
    out = proj(patch_tokens, etext, 14, 14)
    out.sum().backward()
    assert proj.stage2.depth_queries.grad is not None


def test_gradients_flow_through_etext_with_film() -> None:
    _skip_if_no_cuda()
    proj = _make_proj(C=192, cond_dim=192, use_film=True)
    patch_tokens = torch.randn(2, 16, 196, 192, device=DEVICE)
    etext = torch.randn(2, 192, device=DEVICE, requires_grad=True)
    out = proj(patch_tokens, etext, 14, 14)
    out.sum().backward()
    assert etext.grad is not None


# ── Device, dtype, finite ─────────────────────────────────────────────────────


def test_output_on_cuda(proj_base: AADPProjector) -> None:
    _skip_if_no_cuda()
    patch_tokens = torch.randn(2, 32, 196, 768, device=DEVICE)
    etext = torch.randn(2, 768, device=DEVICE)
    out = proj_base(patch_tokens, etext, 14, 14)
    assert out.device.type == "cuda"


def test_output_dtype_float32(proj_base: AADPProjector) -> None:
    _skip_if_no_cuda()
    patch_tokens = torch.randn(2, 32, 196, 768, device=DEVICE)
    etext = torch.randn(2, 768, device=DEVICE)
    out = proj_base(patch_tokens, etext, 14, 14)
    assert out.dtype == torch.float32


def test_output_finite(proj_base: AADPProjector) -> None:
    _skip_if_no_cuda()
    patch_tokens = torch.randn(2, 32, 196, 768, device=DEVICE)
    etext = torch.randn(2, 768, device=DEVICE)
    out = proj_base(patch_tokens, etext, 14, 14)
    assert torch.isfinite(out).all()


# ── Properties ────────────────────────────────────────────────────────────────


def test_num_tokens_property(proj_base: AADPProjector) -> None:
    _skip_if_no_cuda()
    assert proj_base.num_tokens == 64


def test_num_latents_property(proj_base: AADPProjector) -> None:
    _skip_if_no_cuda()
    assert proj_base.num_latents == 32


def test_embed_dim_property(proj_base: AADPProjector) -> None:
    _skip_if_no_cuda()
    assert proj_base.embed_dim == 768
