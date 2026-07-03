"""Tests for MedVLM and variable_depth_collate_fn.

Uses vit_tiny_patch16_224 (pretrained=False) + facebook/opt-125m for speed.
ViT-Tiny embed_dim=192, OPT-125m hidden_size=768 → visual_proj is nn.Linear.
All forward passes on CUDA.
"""

import pytest
import torch
import torch.nn as nn

from aadp.models.vlm import MedVLM, variable_depth_collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_VIT = "vit_tiny_patch16_224"   # 192-dim, no auth needed
_LLM = "facebook/opt-125m"      # 768-dim, no auth needed


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


# ── Module-scoped fixture ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model() -> MedVLM:
    _skip_if_no_cuda()
    return MedVLM(
        vit_model_name=_VIT,
        vit_pretrained=False,           # no download in tests
        vit_frozen=True,
        llm_model_name=_LLM,
        llm_frozen=True,
        num_latents=16,                 # small K for VRAM
        num_tokens=32,                  # small M for speed
        instruction_encoder_model=_LLM,
        device=DEVICE,
    )


def _volumes(B: int = 2, D: int = 32) -> torch.Tensor:
    return torch.rand(B, D, 224, 224, device=DEVICE)


def _instructions(B: int = 2) -> list:
    return ["Assess for lung nodules", "Evaluate liver lesion"][:B]


# ── Training forward pass ─────────────────────────────────────────────────────


def test_training_forward_returns_loss(model: MedVLM) -> None:
    _skip_if_no_cuda()
    report_tokens = torch.randint(0, 100, (2, 20), device=DEVICE)
    out = model(_volumes(), _instructions(), report_tokens=report_tokens)
    assert hasattr(out, "loss"), "training output must have a .loss attribute"


def test_training_loss_is_finite(model: MedVLM) -> None:
    _skip_if_no_cuda()
    report_tokens = torch.randint(0, 100, (2, 20), device=DEVICE)
    out = model(_volumes(), _instructions(), report_tokens=report_tokens)
    assert torch.isfinite(out.loss), f"loss is not finite: {out.loss}"


# ── Inference forward pass ────────────────────────────────────────────────────


def test_inference_returns_tensor(model: MedVLM) -> None:
    _skip_if_no_cuda()
    with torch.no_grad():
        out = model(_volumes(), _instructions(), report_tokens=None, max_new_tokens=8)
    assert isinstance(out, torch.Tensor)


def test_inference_output_shape(model: MedVLM) -> None:
    _skip_if_no_cuda()
    with torch.no_grad():
        out = model(_volumes(), _instructions(), report_tokens=None, max_new_tokens=8)
    assert out.ndim == 2, f"expected 2-D token ids, got shape {out.shape}"
    assert out.shape[0] == 2, f"expected batch dim 2, got {out.shape[0]}"


# ── Visual projection ─────────────────────────────────────────────────────────


def test_visual_proj_is_linear_when_dims_differ(model: MedVLM) -> None:
    """ViT-Tiny (192) + OPT-125m (768) → must be nn.Linear."""
    _skip_if_no_cuda()
    assert isinstance(model.visual_proj, nn.Linear), (
        f"expected nn.Linear, got {type(model.visual_proj)}"
    )


def test_visual_proj_is_identity_when_dims_match() -> None:
    """ViT-Base (768) + OPT-125m (768) → must be nn.Identity."""
    _skip_if_no_cuda()
    m = MedVLM(
        vit_model_name="vit_base_patch16_224",
        vit_pretrained=False,
        llm_model_name=_LLM,
        instruction_encoder_model=_LLM,
        num_latents=8,
        num_tokens=16,
        device=DEVICE,
    )
    assert isinstance(m.visual_proj, nn.Identity), (
        f"expected nn.Identity, got {type(m.visual_proj)}"
    )


# ── Variable-depth collate ────────────────────────────────────────────────────


def test_variable_depth_collate_pads_to_max_d() -> None:
    _skip_if_no_cuda()
    items = [
        {
            "volumes": torch.zeros(d, 224, 224),
            "instructions": "inst",
            "report_tokens": None,
            "depth_spacing_mm": None,
            "label_dict": None,
            "patient_id": None,
        }
        for d in (100, 128, 90)
    ]
    batch = variable_depth_collate_fn(items)
    assert batch["volumes"].shape == (3, 128, 224, 224), (
        f"expected (3, 128, 224, 224), got {batch['volumes'].shape}"
    )


def test_variable_depth_collate_keys() -> None:
    items = [
        {
            "volumes": torch.zeros(50, 224, 224),
            "instructions": "inst",
            "report_tokens": None,
            "depth_spacing_mm": None,
            "label_dict": None,
            "patient_id": "p001",
        }
    ]
    batch = variable_depth_collate_fn(items)
    assert set(batch.keys()) == {
        "volumes", "instructions", "report_tokens",
        "label_dicts", "patient_ids",
    }


def test_variable_depth_collate_report_tokens_padded() -> None:
    """Report tokens of different lengths are padded to max length."""
    items = [
        {
            "volumes": torch.zeros(64, 224, 224),
            "instructions": "inst",
            "report_tokens": torch.randint(0, 100, (30,)),
            "depth_spacing_mm": None,
            "label_dict": None,
            "patient_id": None,
        },
        {
            "volumes": torch.zeros(64, 224, 224),
            "instructions": "inst",
            "report_tokens": torch.randint(0, 100, (50,)),
            "depth_spacing_mm": None,
            "label_dict": None,
            "patient_id": None,
        },
    ]
    batch = variable_depth_collate_fn(items)
    assert batch["report_tokens"] is not None
    assert batch["report_tokens"].shape == (2, 50)


# ── Frozen components ─────────────────────────────────────────────────────────


def test_vit_is_frozen(model: MedVLM) -> None:
    _skip_if_no_cuda()
    assert not any(p.requires_grad for p in model.vit.parameters()), (
        "ViT should be frozen"
    )


def test_llm_is_frozen(model: MedVLM) -> None:
    _skip_if_no_cuda()
    assert not any(p.requires_grad for p in model.llm.parameters()), (
        "LLM should be frozen"
    )


def test_instruction_encoder_is_frozen(model: MedVLM) -> None:
    _skip_if_no_cuda()
    assert not any(p.requires_grad for p in model.instruction_encoder.parameters()), (
        "instruction encoder should always be frozen"
    )


# ── Trainable projector ───────────────────────────────────────────────────────


def test_projector_has_trainable_params(model: MedVLM) -> None:
    _skip_if_no_cuda()
    assert any(p.requires_grad for p in model.projector.parameters()), (
        "projector must have trainable parameters"
    )


# ── Slice attention access ────────────────────────────────────────────────────


def test_get_slice_attention_after_forward(model: MedVLM) -> None:
    _skip_if_no_cuda()
    B, D = 2, 32
    with torch.no_grad():
        model(_volumes(B, D), _instructions(B), report_tokens=None, max_new_tokens=4)
    sa = model.projector.get_slice_attention()
    assert sa.shape == (B, D), f"expected ({B}, {D}), got {sa.shape}"


# ── Output device ─────────────────────────────────────────────────────────────


def test_training_loss_on_correct_device(model: MedVLM) -> None:
    _skip_if_no_cuda()
    report_tokens = torch.randint(0, 100, (2, 10), device=DEVICE)
    out = model(_volumes(), _instructions(), report_tokens=report_tokens)
    assert out.loss.device.type == "cuda"
