"""Tests for Phase 5 training pipeline. All on CUDA.

Models used:
    ViT  : vit_tiny_patch16_224  (C_vit=192, fast to load)
    LLM  : facebook/opt-125m     (C_llm=768, no auth required)
    Instr: facebook/opt-125m     (C_cond=768)

Volumes: (D=16, H=224, W=224) — small enough to fit on RTX 2050 4GB.
"""

import copy
import math
import tempfile
from pathlib import Path
from typing import Iterator, Tuple

import pytest
import torch
import torch.nn as nn
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_VIT = "vit_tiny_patch16_224"
_LLM = "facebook/opt-125m"
_D, _H, _W = 16, 224, 224
_REPORT = "No acute cardiopulmonary findings."


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


# ── Minimal mock dataset (no HF network calls) ────────────────────────────────


class _MockCTRATEDataset:
    """Yields (vol, report_text, label_dict, patient_id) tuples like CTRATEDataset."""

    def __init__(self, n: int = 4, D: int = _D, H: int = _H, W: int = _W) -> None:
        self.n = n
        self.D, self.H, self.W = D, H, W

    def __iter__(self) -> Iterator[Tuple]:
        for i in range(self.n):
            vol = torch.zeros(self.D, self.H, self.W)  # CPU, like real dataset
            yield vol, _REPORT, {}, f"patient_{i}"


# ── Shared module-scoped model ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vlm():
    _skip_if_no_cuda()
    from aadp.models.vlm import MedVLM

    model = MedVLM(
        vit_model_name=_VIT,
        llm_model_name=_LLM,
        instruction_encoder_model=_LLM,
        num_latents=8,
        num_tokens=16,
        device=DEVICE,
    )
    return model


def _make_batch(B: int = 1, L: int = 16) -> Tuple:
    """Return (volumes, instructions, report_tokens) on DEVICE."""
    volumes = torch.zeros(B, _D, _H, _W, device=DEVICE)
    instructions = ["Describe CT findings."] * B
    report_tokens = torch.randint(2, 1000, (B, L), dtype=torch.long, device=DEVICE)
    return volumes, instructions, report_tokens


def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _proj_optimizer(model) -> torch.optim.AdamW:
    params = list(model.projector.parameters())
    if not isinstance(model.visual_proj, nn.Identity):
        params += list(model.visual_proj.parameters())
    return torch.optim.AdamW(params, lr=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# NextTokenLoss
# ═══════════════════════════════════════════════════════════════════════════════


def test_next_token_loss_scalar() -> None:
    _skip_if_no_cuda()
    from aadp.training.losses import NextTokenLoss

    loss_fn = NextTokenLoss()
    B, L, V = 2, 32, 1000
    logits = torch.randn(B, L, V, device=DEVICE)
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=DEVICE)
    loss = loss_fn(logits, labels)
    assert loss.shape == torch.Size([])
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_next_token_loss_ignore_index() -> None:
    """All-ignored labels must raise ValueError (NaN guard catches silent corruption)."""
    _skip_if_no_cuda()
    from aadp.training.losses import NextTokenLoss

    loss_fn = NextTokenLoss(ignore_index=-100)
    B, L, V = 2, 32, 1000
    logits = torch.randn(B, L, V, device=DEVICE)
    labels = torch.full((B, L), -100, dtype=torch.long, device=DEVICE)
    with pytest.raises(ValueError, match="NaN"):
        loss_fn(logits, labels)


def test_next_token_loss_partial_ignore() -> None:
    """Mixed real and -100 labels should give a positive finite loss."""
    _skip_if_no_cuda()
    from aadp.training.losses import NextTokenLoss

    loss_fn = NextTokenLoss()
    B, L, V = 2, 32, 1000
    logits = torch.randn(B, L, V, device=DEVICE)
    labels = torch.full((B, L), -100, dtype=torch.long, device=DEVICE)
    labels[:, 16:] = torch.randint(0, V, (B, 16), dtype=torch.long, device=DEVICE)
    loss = loss_fn(logits, labels)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_next_token_loss_custom_ignore_index() -> None:
    _skip_if_no_cuda()
    from aadp.training.losses import NextTokenLoss

    loss_fn = NextTokenLoss(ignore_index=0)
    B, L, V = 2, 8, 100
    logits = torch.randn(B, L, V, device=DEVICE)
    labels = torch.zeros(B, L, dtype=torch.long, device=DEVICE)  # all ignored
    with pytest.raises(ValueError, match="NaN"):
        loss_fn(logits, labels)


def test_combined_loss_re_exported() -> None:
    _skip_if_no_cuda()
    from aadp.training.losses import CombinedLoss

    c = CombinedLoss(lambda_attn=0.1)
    lm = torch.tensor(2.5, device=DEVICE)
    out = c(lm, use_attn_loss=False)
    assert out.item() == pytest.approx(2.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ═══════════════════════════════════════════════════════════════════════════════


def _dummy_optimizer(lr: float = 1e-3) -> torch.optim.AdamW:
    param = nn.Linear(4, 4)
    return torch.optim.AdamW(param.parameters(), lr=lr)


def test_scheduler_warmup_start_near_zero() -> None:
    _skip_if_no_cuda()
    from aadp.training.scheduler import get_cosine_schedule_with_warmup

    opt = _dummy_optimizer(lr=1e-3)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=100, num_training_steps=1000)
    # At step 0 the multiplier is 0/100 = 0.0
    assert sched.get_last_lr()[0] == pytest.approx(0.0, abs=1e-9)


def test_scheduler_peaks_at_warmup_end() -> None:
    _skip_if_no_cuda()
    from aadp.training.scheduler import get_cosine_schedule_with_warmup

    opt = _dummy_optimizer(lr=1e-3)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=100)
    # Step through the warmup
    for _ in range(10):
        opt.step()
        sched.step()
    # At step 10 the multiplier should be 1.0
    assert sched.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-4)


def test_scheduler_decays_after_warmup() -> None:
    _skip_if_no_cuda()
    from aadp.training.scheduler import get_cosine_schedule_with_warmup

    warmup = 10
    total = 100
    opt = _dummy_optimizer(lr=1e-3)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total)

    # Step to end of warmup
    for _ in range(warmup):
        opt.step(); sched.step()
    lr_at_warmup = sched.get_last_lr()[0]

    # Step to halfway through cosine phase
    for _ in range((total - warmup) // 2):
        opt.step(); sched.step()
    lr_mid = sched.get_last_lr()[0]

    assert lr_mid < lr_at_warmup


def test_scheduler_floor_at_min_lr_ratio() -> None:
    _skip_if_no_cuda()
    from aadp.training.scheduler import get_cosine_schedule_with_warmup

    min_ratio = 0.1
    base_lr = 1e-3
    opt = _dummy_optimizer(lr=base_lr)
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=1, num_training_steps=100, min_lr_ratio=min_ratio
    )
    # Fast-forward past the end of cosine
    for _ in range(100):
        opt.step(); sched.step()
    # Should be at or near the floor (cosine converges to min_lr_ratio * base_lr)
    assert sched.get_last_lr()[0] >= min_ratio * base_lr - 1e-7


def test_scheduler_works_with_multiple_param_groups() -> None:
    _skip_if_no_cuda()
    from aadp.training.scheduler import get_cosine_schedule_with_warmup

    m1, m2 = nn.Linear(4, 4), nn.Linear(4, 4)
    opt = torch.optim.AdamW(
        [{"params": m1.parameters(), "lr": 1e-3}, {"params": m2.parameters(), "lr": 1e-4}]
    )
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=50)
    opt.step(); sched.step()
    # Both groups should have their respective LRs scaled
    assert len(sched.get_last_lr()) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Single training step (5 steps, loss should decrease on fixed batch)
# ═══════════════════════════════════════════════════════════════════════════════


def test_single_training_step_loss_decreases(vlm) -> None:
    _skip_if_no_cuda()
    from aadp.training.scheduler import get_cosine_schedule_with_warmup

    optimizer = _proj_optimizer(vlm)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 5)

    # Save and restore projector state so we don't corrupt the shared fixture
    proj_state = copy.deepcopy(vlm.projector.state_dict())
    vis_state = copy.deepcopy(vlm.visual_proj.state_dict())

    volumes, instructions, report_tokens = _make_batch()
    losses = []

    vlm.train()
    optimizer.zero_grad()
    try:
        for _ in range(5):
            output = vlm(volumes, instructions, report_tokens)
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
    finally:
        vlm.projector.load_state_dict(proj_state)
        vlm.visual_proj.load_state_dict(vis_state)

    assert losses[-1] < losses[0], (
        f"Loss did not decrease after 5 steps: {losses}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Gradient accumulation
# ═══════════════════════════════════════════════════════════════════════════════


def test_gradient_accumulation_params_unchanged_before_step(vlm) -> None:
    """Projector params must stay frozen for the first 3 of 4 micro-batches."""
    _skip_if_no_cuda()

    proj_state = copy.deepcopy(vlm.projector.state_dict())
    vis_state = copy.deepcopy(vlm.visual_proj.state_dict())

    optimizer = _proj_optimizer(vlm)
    grad_accum = 4
    volumes, instructions, report_tokens = _make_batch()

    vlm.train()
    optimizer.zero_grad()

    try:
        # 3 micro-batches — no optimizer.step() yet
        for i in range(3):
            output = vlm(volumes, instructions, report_tokens)
            (output.loss / grad_accum).backward()
            # Params must still match original after each backward
            for k, v in vlm.projector.named_parameters():
                assert torch.equal(v, proj_state[k]), (
                    f"Projector param '{k}' changed before optimizer.step() at micro-batch {i+1}"
                )

        # 4th micro-batch + step
        output = vlm(volumes, instructions, report_tokens)
        (output.loss / grad_accum).backward()
        optimizer.step()
        optimizer.zero_grad()

        # Now params must have changed
        any_changed = any(
            not torch.equal(v, proj_state[k])
            for k, v in vlm.projector.named_parameters()
        )
        assert any_changed, "Projector params did not change after optimizer.step()"
    finally:
        vlm.projector.load_state_dict(proj_state)
        vlm.visual_proj.load_state_dict(vis_state)


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed precision
# ═══════════════════════════════════════════════════════════════════════════════


def test_mixed_precision_forward_finite(vlm) -> None:
    _skip_if_no_cuda()
    volumes, instructions, report_tokens = _make_batch()
    vlm.eval()
    with torch.amp.autocast("cuda", enabled=True):
        with torch.no_grad():
            output = vlm(volumes, instructions, report_tokens)
    assert torch.isfinite(output.loss)


def test_mixed_precision_backward_runs(vlm) -> None:
    _skip_if_no_cuda()
    proj_state = copy.deepcopy(vlm.projector.state_dict())
    vis_state = copy.deepcopy(vlm.visual_proj.state_dict())

    optimizer = _proj_optimizer(vlm)
    scaler = torch.amp.GradScaler("cuda")
    volumes, instructions, report_tokens = _make_batch()

    vlm.train()
    optimizer.zero_grad()
    try:
        with torch.amp.autocast("cuda", enabled=True):
            output = vlm(volumes, instructions, report_tokens)
            loss = output.loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vlm.projector.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        assert torch.isfinite(loss)
    finally:
        vlm.projector.load_state_dict(proj_state)
        vlm.visual_proj.load_state_dict(vis_state)


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint save and load
# ═══════════════════════════════════════════════════════════════════════════════


def test_checkpoint_save_and_load(vlm) -> None:
    _skip_if_no_cuda()
    from aadp.training.losses import NextTokenLoss
    from aadp.training.scheduler import get_cosine_schedule_with_warmup
    from aadp.training.trainer import Trainer

    optimizer = _proj_optimizer(vlm)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)
    loss_fn = NextTokenLoss()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "checkpoint_dir": tmpdir,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "num_epochs": 1,
            "val_every_n_steps": 9999,
            "save_every_n_steps": 9999,
            "mixed_precision": False,
            "use_attn_loss": False,
        }
        trainer = Trainer(
            model=vlm,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            train_dataset=_MockCTRATEDataset(n=0),
            val_dataset=_MockCTRATEDataset(n=0),
            config=config,
            device=DEVICE,
            use_wandb=False,
        )

        # Snapshot projector weights before save
        proj_before = {k: v.clone().detach() for k, v in vlm.projector.named_parameters()}

        # Save
        ckpt_path = trainer.save_checkpoint()
        assert Path(ckpt_path).exists()

        # Perturb projector weights
        with torch.no_grad():
            for p in vlm.projector.parameters():
                p.add_(torch.ones_like(p) * 999.0)

        # Load checkpoint back into the same model
        step = trainer.load_checkpoint(ckpt_path)
        assert step == 0  # saved at step 0

        # Verify weights were restored exactly
        for k, v in vlm.projector.named_parameters():
            assert torch.allclose(v, proj_before[k]), (
                f"Projector param '{k}' not restored after load_checkpoint()"
            )


def test_checkpoint_contains_required_keys(vlm) -> None:
    _skip_if_no_cuda()
    from aadp.training.losses import NextTokenLoss
    from aadp.training.scheduler import get_cosine_schedule_with_warmup
    from aadp.training.trainer import Trainer

    optimizer = _proj_optimizer(vlm)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=vlm,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=NextTokenLoss(),
            train_dataset=_MockCTRATEDataset(n=0),
            val_dataset=_MockCTRATEDataset(n=0),
            config={"checkpoint_dir": tmpdir},
            device=DEVICE,
            use_wandb=False,
        )
        path = trainer.save_checkpoint()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("projector", "optimizer", "scheduler", "step", "epoch"):
            assert key in ckpt, f"Missing key '{key}' in checkpoint"


# ═══════════════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════════════

_REQUIRED_KEYS = [
    "experiment_name", "projector", "device", "mixed_precision",
    "num_epochs", "batch_size", "gradient_accumulation_steps",
    "max_grad_norm", "learning_rate", "warmup_steps",
    "val_every_n_steps", "save_every_n_steps", "checkpoint_dir",
    "use_wandb", "use_attn_loss",
    "vit_model_name", "vit_frozen", "llm_model_name", "llm_frozen",
    "embed_dim", "cond_dim", "max_depth",
    "hf_token", "shuffle", "shuffle_buffer_size", "max_samples",
]


def test_config_aadp_base_required_keys() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "aadp_base.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    missing = [k for k in _REQUIRED_KEYS if k not in config]
    assert not missing, f"Missing keys in aadp_base.yaml: {missing}"


def test_config_all_six_files_exist() -> None:
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    expected = [
        "aadp_base.yaml",
        "aadp_ablation_k_m_grid.yaml",
        "aadp_ablation_attention_cond.yaml",
        "aadp_ablation_aux_loss.yaml",
        "baseline_perceiver.yaml",
        "baseline_medpruner.yaml",
    ]
    for name in expected:
        assert (configs_dir / name).exists(), f"Config file missing: {name}"


def test_config_aadp_base_projector_is_aadp() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "aadp_base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert config["projector"] == "aadp"
    assert config["use_film"] is True
    assert config["num_latents"] == 32
    assert config["num_tokens"] == 64


# ═══════════════════════════════════════════════════════════════════════════════
# --set override
# ═══════════════════════════════════════════════════════════════════════════════


def test_set_override_num_latents() -> None:
    """--set num_latents=16 should produce a projector with K=16."""
    _skip_if_no_cuda()
    from aadp.training.factory import build_projector

    config_path = Path(__file__).resolve().parents[1] / "configs" / "aadp_base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Simulate --set num_latents=16
    config["num_latents"] = 16
    # Use small dims for speed
    proj = build_projector(config, embed_dim=192, cond_dim=768, device=DEVICE)
    assert proj.num_latents == 16


def test_set_override_num_tokens() -> None:
    """--set num_tokens=32 should produce a projector with M=32."""
    _skip_if_no_cuda()
    from aadp.training.factory import build_projector

    config_path = Path(__file__).resolve().parents[1] / "configs" / "aadp_base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["num_tokens"] = 32
    proj = build_projector(config, embed_dim=192, cond_dim=768, device=DEVICE)
    assert proj.num_tokens == 32


def test_set_override_type_coercion() -> None:
    """Values in --set should be coerced: int, float, bool, null."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.train import _parse_value

    assert _parse_value("16") == 16
    assert isinstance(_parse_value("16"), int)
    assert _parse_value("1e-4") == pytest.approx(1e-4)
    assert _parse_value("true") is True
    assert _parse_value("false") is False
    assert _parse_value("null") is None
    assert _parse_value("hello") == "hello"


# ═══════════════════════════════════════════════════════════════════════════════
# Projector switching
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("projector_type", [
    "aadp",
    "attention_conditioned_aadp",
    "perceiver",
    "medpruner",
])
def test_projector_switching(projector_type: str) -> None:
    """build_projector factory must work for all four projector strings."""
    _skip_if_no_cuda()
    from aadp.training.factory import build_projector

    config = {
        "projector": projector_type,
        "num_latents": 8,
        "num_tokens": 16,
        "use_film": True,
        "max_depth": 128,
        "similarity_threshold": 0.95,
    }
    proj = build_projector(config, embed_dim=192, cond_dim=768, device=DEVICE)
    assert proj is not None
    assert proj.num_tokens == 16

    # Verify forward pass runs
    patch_tokens = torch.randn(1, 8, 49, 192, device=DEVICE)
    etext = torch.randn(1, 768, device=DEVICE)
    with torch.no_grad():
        out = proj(patch_tokens, etext, H_patches=7, W_patches=7)
    assert out.shape == (1, 16, 192)


def test_projector_switching_invalid_raises() -> None:
    """Unknown projector type must raise ValueError."""
    _skip_if_no_cuda()
    from aadp.training.factory import build_projector

    with pytest.raises(ValueError, match="Unknown projector type"):
        build_projector({"projector": "nonexistent"}, 192, 768, DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# MedVLM projector injection
# ═══════════════════════════════════════════════════════════════════════════════


def test_medvlm_accepts_external_projector() -> None:
    """MedVLM should use an externally supplied projector."""
    _skip_if_no_cuda()
    from aadp.models.projector.aadp import AADPProjector
    from aadp.models.vlm import MedVLM

    # C_vit for vit_tiny_patch16_224 is 192, C_cond for opt-125m is 768
    proj = AADPProjector(
        embed_dim=192, num_latents=4, num_tokens=8, cond_dim=768, device=DEVICE
    )
    model = MedVLM(
        vit_model_name=_VIT,
        llm_model_name=_LLM,
        instruction_encoder_model=_LLM,
        projector=proj,
        device=DEVICE,
    )
    # The injected projector should be used
    assert model.projector is proj
    assert model._num_tokens == 8

    # Forward should run with the injected projector
    vols = torch.zeros(1, _D, _H, _W, device=DEVICE)
    report = torch.randint(2, 100, (1, 8), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model(vols, ["Describe."], report)
    assert torch.isfinite(out.loss)
