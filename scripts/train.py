"""CLI entry point for training MedVLM with any projector variant.

Usage examples::

    # Main A-ADP run
    python scripts/train.py --config configs/aadp_base.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/aadp_base.yaml \\
        --resume checkpoints/checkpoint_step_1000.pt

    # K×M grid sweep (override any config key)
    python scripts/train.py --config configs/aadp_ablation_k_m_grid.yaml \\
        --set num_latents=16 num_tokens=32

    # Quick debug run
    python scripts/train.py --config configs/aadp_base.yaml --max_samples 100
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# ── Ensure project root is on PYTHONPATH when run as a script ────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aadp.data.ctrate_dataset import CTRATEDataset
from aadp.data.multitask_sampler import MultiTaskCTRATEDataset
from aadp.data.radgenome_dataset import RadGenomeDataset
from aadp.models.vlm import MedVLM
from aadp.training.factory import build_projector
from aadp.training.losses import NextTokenLoss
from aadp.training.scheduler import get_cosine_schedule_with_warmup
from aadp.training.trainer import Trainer

log = logging.getLogger(__name__)


# ── Config helpers ────────────────────────────────────────────────────────────


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _parse_value(s: str) -> Any:
    """Try int → float → bool → string."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "null" or s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _apply_overrides(config: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    """Apply ``key=value`` CLI overrides to the config dict."""
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"--set argument must be 'key=value', got: {kv!r}")
        k, v = kv.split("=", 1)
        config[k] = _parse_value(v)
    return config


# ── Model factory ─────────────────────────────────────────────────────────────


def build_model(config: Dict[str, Any], device: torch.device) -> MedVLM:
    """Build MedVLM with the projector specified by ``config["projector"]``.

    Dimensions are derived cheaply from the ViT architecture (pretrained=False)
    and from the LLM config (no weight loading needed).
    """
    from transformers import AutoConfig

    from aadp.models.encoder.vit_encoder import SliceEncoder

    # Get C_vit without downloading pretrained weights
    _vit_stub = SliceEncoder(
        model_name=config["vit_model_name"],
        pretrained=False,
        frozen=True,
        resize_to=config.get("vit_resize_to"),
    )
    C_vit = _vit_stub.output_dim
    del _vit_stub

    # Get C_cond from the LLM config (no weight loading)
    C_cond = AutoConfig.from_pretrained(
        config.get("instruction_encoder_model", config["llm_model_name"]),
        token=config.get("hf_token") or None,
    ).hidden_size

    # Build the projector
    proj = build_projector(config, embed_dim=C_vit, cond_dim=C_cond, device=device)

    # Build MedVLM with the injected projector
    model = MedVLM(
        vit_model_name=config["vit_model_name"],
        vit_frozen=config.get("vit_frozen", True),
        vit_pretrained=True,
        vit_resize_to=config.get("vit_resize_to"),
        llm_model_name=config["llm_model_name"],
        llm_frozen=config.get("llm_frozen", True),
        llm_lora=config.get("llm_lora"),
        instruction_encoder_model=config.get(
            "instruction_encoder_model", config["llm_model_name"]
        ),
        projector=proj,
        device=device,
    )
    return model


# ── Optimizer ─────────────────────────────────────────────────────────────────


def build_optimizer(
    model: MedVLM, config: Dict[str, Any]
) -> torch.optim.AdamW:
    """AdamW over every trainable parameter.

    Collecting ``requires_grad`` params is correct because the model has already
    frozen everything that must stay fixed: the ViT and instruction encoder are
    frozen at construction, and LoRA wrapping (when enabled) freezes the base LLM
    weights while marking only the adapter weights trainable.  This therefore
    covers the projector, ``visual_proj``, and any LoRA adapters in one pass.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.AdamW(
        trainable_params,
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.0),
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train MedVLM.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config keys, e.g. --set num_latents=16 num_tokens=32",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit dataset size (debug runs).",
    )
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    config = _load_config(args.config)
    config = _apply_overrides(config, args.set or [])
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    # Read HF token from env if not set in config
    if not config.get("hf_token"):
        config["hf_token"] = os.environ.get("HF_TOKEN")

    device = torch.device(config.get("device", "cuda"))

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = config.get("use_wandb", True)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="aadp",
                name=config.get("experiment_name", "run"),
                config=config,
            )
        except ImportError:
            log.warning("wandb not installed — disabling WandB logging.")
            use_wandb = False

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset_kwargs = dict(
        shuffle=config.get("shuffle", True),
        shuffle_buffer_size=config.get("shuffle_buffer_size", 1000),
        max_samples=config.get("max_samples"),
        token=config.get("hf_token"),
        local_data_dir=config.get("local_data_dir"),
    )
    # Optional RadGenome grounding: enables T4 localisation instructions and the
    # A4 attention-alignment loss.  Joined to CT-RATE volumes by the volume id
    # (filename stem, e.g. "train_1_a_1"), which both datasets share.
    radgenome = None
    radgenome_root = config.get("radgenome_root")
    if radgenome_root:
        if Path(radgenome_root).exists():
            radgenome = RadGenomeDataset(radgenome_root)
            log.info("Loaded RadGenome grounding from %s", radgenome_root)
        else:
            log.warning(
                "radgenome_root '%s' does not exist — T4 localisation and the "
                "A4 attention loss will be inactive.", radgenome_root,
            )

    train_ds = MultiTaskCTRATEDataset(
        CTRATEDataset(split="train", **dataset_kwargs), radgenome_dataset=radgenome
    )
    val_ds = MultiTaskCTRATEDataset(
        CTRATEDataset(split="valid", **dataset_kwargs), radgenome_dataset=radgenome
    )

    # ── Model, optimizer, scheduler, loss ────────────────────────────────────
    model = build_model(config, device)
    optimizer = build_optimizer(model, config)

    num_train_steps = config.get("num_epochs", 10) * (
        config.get("max_samples") or 1000
    ) // max(config.get("batch_size", 2), 1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get("warmup_steps", 500),
        num_training_steps=num_train_steps,
    )
    loss_fn = NextTokenLoss()

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        device=device,
        use_wandb=use_wandb,
    )

    if args.resume:
        step = trainer.load_checkpoint(args.resume)
        log.info("Resumed from step %d", step)

    trainer.train()

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
