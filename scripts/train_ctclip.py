#!/usr/bin/env python3
"""Training entry point for the CT-CLIP Stage 2 VLM.

Reads pre-extracted CT-CLIP features (.pt files) and trains the
CTCLIPStage2VLM (Stage 2 + LLM) end-to-end with multi-task instruction tuning.

Usage
-----
Full training run::

    python scripts/train_ctclip.py --config configs/ctclip_stage2.yaml

Override individual config keys::

    python scripts/train_ctclip.py \\
        --config configs/ctclip_stage2.yaml \\
        --set num_tokens=128 \\
        --set features_train_dir=/my/features/train

Debug run with tiny dataset::

    python scripts/train_ctclip.py \\
        --config configs/ctclip_stage2.yaml \\
        --set llm_model_name=facebook/opt-125m \\
        --set max_samples=20 \\
        --set num_epochs=1 \\
        --set use_wandb=false
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Config helpers ─────────────────────────────────────────────────────────────


def _load_config(path: str) -> Dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _apply_overrides(cfg: Dict, overrides: Optional[List[str]]) -> Dict:
    """Apply --set KEY=VALUE pairs to cfg (supports nested keys with '.')."""
    if not overrides:
        return cfg
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"--set requires KEY=VALUE format, got: {kv!r}")
        key, _, val = kv.partition("=")
        # Attempt type coercion
        for coerce in (int, float):
            try:
                val = coerce(val)
                break
            except ValueError:
                pass
        if val == "true":
            val = True
        elif val == "false":
            val = False
        elif val == "null":
            val = None
        cfg[key.strip()] = val
    return cfg


# ── Checkpoint helpers ─────────────────────────────────────────────────────────


def _save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    val_loss: float,
) -> None:
    state = {
        "projector": model.projector.state_dict(),
        "visual_proj": model.visual_proj.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
    }
    # Save LoRA adapter weights separately when present
    lora_state = {
        k: v for k, v in model.llm.state_dict().items() if "lora_" in k
    }
    if lora_state:
        state["llm_lora"] = lora_state
    torch.save(state, path)
    logger.info("Checkpoint saved → %s", path)


def _load_checkpoint(path: str, model, optimizer, scheduler, device: str):
    ckpt = torch.load(path, map_location=device)
    model.projector.load_state_dict(ckpt["projector"])
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if "llm_lora" in ckpt:
        model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    logger.info(
        "Resumed from %s (step=%d, epoch=%d, val_loss=%.4f)",
        path, ckpt["step"], ckpt["epoch"], ckpt.get("val_loss", float("inf")),
    )
    return ckpt["step"], ckpt["epoch"], ckpt.get("val_loss", float("inf"))


# ── Validation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def _validate(model, val_loader, device: str, max_batches: Optional[int] = None) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        features = batch["features"].to(device)
        instructions = batch["instruction"]
        target_enc = model.tokenizer(
            batch["target"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).input_ids.to(device)
        out = model(features, instructions, report_tokens=target_enc, training=True)
        total_loss += out["loss"].item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CT-CLIP Stage 2 VLM")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--set", nargs="*", dest="overrides", metavar="KEY=VALUE",
        help="Override config keys, e.g. --set num_tokens=128 use_wandb=false",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg = _apply_overrides(cfg, args.overrides)

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA unavailable — falling back to CPU.")
        device = "cpu"
        cfg["device"] = "cpu"

    # ── Datasets ─────────────────────────────────────────────────────────────
    from aadp.data.ctclip_feature_dataset import CTCLIPFeatureDataset, ctclip_collate_fn

    max_samples = cfg.get("max_samples")
    tasks = cfg.get("tasks", ["T1", "T2", "T3"])
    task_weights = cfg.get("task_weights", {"T1": 0.6, "T2": 0.3, "T3": 0.1})

    logger.info("Loading training dataset from %s", cfg["features_train_dir"])
    train_ds = CTCLIPFeatureDataset(
        features_dir=cfg["features_train_dir"],
        csv_path=cfg["ctrate_csv_train"],
        tasks=tasks,
        task_weights=task_weights,
        max_samples=max_samples,
    )
    logger.info("Loading validation dataset from %s", cfg["features_valid_dir"])
    val_ds = CTCLIPFeatureDataset(
        features_dir=cfg["features_valid_dir"],
        csv_path=cfg["ctrate_csv_valid"],
        tasks=tasks,
        task_weights=task_weights,
        max_samples=max_samples,
    )

    batch_size = cfg.get("batch_size", 8)
    num_workers = min(cfg.get("num_workers", 8), os.cpu_count() or 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(num_workers > 0),
        collate_fn=ctclip_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(num_workers > 0),
        collate_fn=ctclip_collate_fn,
    )

    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    # ── Model ─────────────────────────────────────────────────────────────────
    from aadp.models.ctclip_vlm import CTCLIPStage2VLM

    logger.info("Building CTCLIPStage2VLM (llm=%s)...", cfg.get("llm_model_name"))
    model = CTCLIPStage2VLM(
        ctclip_dim=cfg.get("ctclip_dim", 512),
        embed_dim=cfg.get("embed_dim", 512),
        num_tokens=cfg.get("num_tokens", 64),
        num_heads=cfg.get("num_heads", 8),
        cond_dim=cfg.get("cond_dim", 2048),
        use_film=cfg.get("use_film", True),
        max_depth=cfg.get("max_depth", 24),
        dropout=cfg.get("dropout", 0.0),
        llm_model_name=cfg.get("llm_model_name", "facebook/opt-1.3b"),
        llm_frozen=cfg.get("llm_frozen", False),
        llm_lora=cfg.get("llm_lora"),
        instruction_encoder_model=cfg.get("instruction_encoder_model", "facebook/opt-1.3b"),
        device=device,
    )
    model.train()

    # ── Trainable parameters ──────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info("Trainable parameters: %s", f"{n_trainable:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    from aadp.training.scheduler import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    num_epochs = cfg.get("num_epochs", 5)
    steps_per_epoch = max(len(train_loader), 1)
    num_training_steps = num_epochs * steps_per_epoch
    warmup_steps = cfg.get("warmup_steps", 500)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp = cfg.get("mixed_precision", True) and device != "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.cuda.amp.autocast if use_amp else (lambda: __import__("contextlib").nullcontext())

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = cfg.get("use_wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="ctclip-stage2",
                name=cfg.get("experiment_name", "ctclip_stage2"),
                config=cfg,
            )
        except Exception as exc:
            logger.warning("WandB init failed (%s) — continuing without logging.", exc)
            use_wandb = False

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints/ctclip_stage2/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and Path(args.resume).exists():
        global_step, start_epoch, best_val_loss = _load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    val_every = cfg.get("val_every_n_steps", 500)
    save_every = cfg.get("save_every_n_steps", 1000)
    patience = cfg.get("patience", 5)
    patience_counter = 0

    logger.info(
        "Starting training: %d epochs, %d steps/epoch, grad_accum=%d, amp=%s",
        num_epochs, steps_per_epoch, grad_accum, use_amp,
    )

    optimizer.zero_grad()

    for epoch in range(start_epoch, num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            instructions = batch["instruction"]

            # Tokenize targets (report text / entity answer / yes-no)
            target_enc = model.tokenizer(
                batch["target"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).input_ids.to(device)

            with autocast_ctx():
                out = model(
                    features, instructions, report_tokens=target_enc, training=True
                )
                loss = out["loss"] / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(trainable_params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                raw_loss = loss.item() * grad_accum
                lr_now = scheduler.get_last_lr()[0]

                if global_step % 10 == 0:
                    logger.info(
                        "epoch=%d  step=%d  loss=%.4f  lr=%.2e",
                        epoch + 1, global_step, raw_loss, lr_now,
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({"train/loss": raw_loss, "train/lr": lr_now,
                                   "step": global_step, "epoch": epoch + 1})

                # ── Validation ────────────────────────────────────────────────
                if global_step % val_every == 0:
                    val_loss = _validate(model, val_loader, device, max_batches=50)
                    logger.info(
                        "  [val] step=%d  val_loss=%.4f  best=%.4f",
                        global_step, val_loss, best_val_loss,
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({"val/loss": val_loss, "step": global_step})

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        _save_checkpoint(
                            str(ckpt_dir / "checkpoint_best.pt"),
                            model, optimizer, scheduler,
                            global_step, epoch, val_loss,
                        )
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(
                                "Early stopping: val_loss did not improve for %d checks.",
                                patience,
                            )
                            _save_checkpoint(
                                str(ckpt_dir / "checkpoint_latest.pt"),
                                model, optimizer, scheduler,
                                global_step, epoch, val_loss,
                            )
                            if use_wandb:
                                import wandb
                                wandb.finish()
                            sys.exit(0)

                # ── Periodic checkpoint ───────────────────────────────────────
                if global_step % save_every == 0:
                    _save_checkpoint(
                        str(ckpt_dir / f"checkpoint_step_{global_step}.pt"),
                        model, optimizer, scheduler,
                        global_step, epoch, best_val_loss,
                    )

        # End of epoch checkpoint
        _save_checkpoint(
            str(ckpt_dir / "checkpoint_latest.pt"),
            model, optimizer, scheduler,
            global_step, epoch, best_val_loss,
        )
        logger.info("Epoch %d/%d complete. Best val_loss=%.4f", epoch + 1, num_epochs, best_val_loss)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
