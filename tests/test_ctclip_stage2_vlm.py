"""Tests for CTCLIPStage2Projector and CTCLIPStage2VLM."""

import math

import pytest
import torch

# ── CTCLIPStage2Projector ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def projector():
    from aadp.models.projector.ctclip_stage2 import CTCLIPStage2Projector

    return CTCLIPStage2Projector(
        ctclip_dim=512,
        embed_dim=512,
        num_tokens=64,
        num_heads=8,
        cond_dim=64,    # small cond_dim to avoid needing a real LM
        dropout=0.0,
        use_film=True,
        max_depth=24,
        device="cpu",
    )


def test_projector_forward_shape(projector):
    B = 2
    features = torch.randn(B, 24, 576, 512)
    etext = torch.randn(B, 64)
    out = projector(features, etext)
    assert out.shape == (B, 64, 512), f"Expected (2, 64, 512), got {out.shape}"


def test_projector_no_nan(projector):
    features = torch.randn(2, 24, 576, 512)
    etext = torch.randn(2, 64)
    out = projector(features, etext)
    assert not torch.isnan(out).any()


def test_projector_rebuild_at_budget(projector):
    projector.rebuild_at_budget(32)
    assert projector.stage2.num_tokens == 32
    assert projector.num_tokens == 32
    features = torch.randn(1, 24, 576, 512)
    etext = torch.randn(1, 64)
    out = projector(features, etext)
    assert out.shape == (1, 32, 512)
    # Restore for other tests
    projector.rebuild_at_budget(64)


def test_projector_num_parameters(projector):
    counts = projector.num_parameters()
    assert "input_proj" in counts
    assert "stage2" in counts
    assert "total" in counts
    assert counts["total"] == counts["input_proj"] + counts["stage2"]
    assert counts["total"] > 0


def test_projector_input_proj_identity():
    """When ctclip_dim == embed_dim, input_proj should be Identity."""
    from aadp.models.projector.ctclip_stage2 import CTCLIPStage2Projector
    import torch.nn as nn

    proj = CTCLIPStage2Projector(ctclip_dim=256, embed_dim=256, cond_dim=32, device="cpu")
    assert isinstance(proj.input_proj, nn.Identity)


def test_projector_input_proj_linear():
    """When ctclip_dim != embed_dim, input_proj should be Linear."""
    from aadp.models.projector.ctclip_stage2 import CTCLIPStage2Projector
    import torch.nn as nn

    proj = CTCLIPStage2Projector(ctclip_dim=512, embed_dim=256, cond_dim=32, device="cpu")
    assert isinstance(proj.input_proj, nn.Linear)
    features = torch.randn(1, 24, 576, 512)
    etext = torch.randn(1, 32)
    out = proj(features, etext)
    assert out.shape == (1, 64, 256)


def test_projector_get_slice_attention(projector):
    features = torch.randn(2, 24, 576, 512)
    etext = torch.randn(2, 64)
    projector(features, etext)
    attn = projector.get_slice_attention()
    assert attn is not None
    assert attn.shape == (2, 24)
    # Attention masses should be non-negative
    assert (attn >= 0).all()


# ── CTCLIPStage2VLM ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vlm():
    from aadp.models.ctclip_vlm import CTCLIPStage2VLM

    return CTCLIPStage2VLM(
        ctclip_dim=512,
        embed_dim=512,
        num_tokens=16,        # small M for speed
        num_heads=8,
        cond_dim=768,         # matches opt-125m hidden_size
        use_film=True,
        max_depth=24,
        dropout=0.0,
        llm_model_name="facebook/opt-125m",
        llm_frozen=True,
        llm_lora=None,
        instruction_encoder_model="facebook/opt-125m",
        device="cpu",
    )


def test_vlm_training_loss_scalar(vlm):
    """Training forward should return a finite scalar loss."""
    B = 2
    features = torch.randn(B, 24, 576, 512)
    instructions = ["Generate a radiology report for this CT scan."] * B
    # Tiny fake report token ids (use EOS=2 for OPT)
    report_tokens = torch.randint(4, 50, (B, 8))

    out = vlm(features, instructions, report_tokens=report_tokens, training=True)
    loss = out["loss"]
    assert loss.shape == torch.Size([]), f"Loss should be scalar, got {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"


def test_vlm_training_logits_shape(vlm):
    B = 2
    features = torch.randn(B, 24, 576, 512)
    instructions = ["Describe findings related to lung in this CT scan."] * B
    report_tokens = torch.randint(4, 50, (B, 6))
    out = vlm(features, instructions, report_tokens=report_tokens, training=True)
    assert "logits" in out
    # logits: (B, M + L_inst + L_rep, vocab_size)
    assert out["logits"].ndim == 3
    assert out["logits"].shape[0] == B


def test_vlm_inference_returns_ids(vlm):
    B = 1
    features = torch.randn(B, 24, 576, 512)
    instructions = ["Is there evidence of cardiomegaly in this scan? Answer yes or no."]
    out = vlm(features, instructions, training=False)
    assert "generated_ids" in out
    gen = out["generated_ids"]
    assert gen.ndim == 2
    assert gen.shape[0] == B


def test_vlm_rebuild_at_budget(vlm):
    vlm.rebuild_at_budget(32)
    assert vlm.projector.stage2.num_tokens == 32
    assert vlm.num_tokens == 32

    features = torch.randn(1, 24, 576, 512)
    instructions = ["Generate a radiology report for this CT scan."]
    report_tokens = torch.randint(4, 50, (1, 5))
    out = vlm(features, instructions, report_tokens=report_tokens, training=True)
    loss = out["loss"]
    assert not torch.isnan(loss)

    # Restore
    vlm.rebuild_at_budget(16)


def test_vlm_cond_dim_mismatch_raises():
    """Wrong cond_dim should raise AssertionError at init."""
    from aadp.models.ctclip_vlm import CTCLIPStage2VLM

    with pytest.raises(AssertionError, match="cond_dim mismatch"):
        CTCLIPStage2VLM(
            cond_dim=9999,   # wrong — opt-125m hidden_size=768
            llm_model_name="facebook/opt-125m",
            instruction_encoder_model="facebook/opt-125m",
            device="cpu",
        )
