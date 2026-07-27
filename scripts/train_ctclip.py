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
    if model.cls_head is not None:
        state["cls_head"] = model.cls_head.state_dict()
    # Save LoRA adapter weights separately when present
    lora_state = {
        k: v for k, v in model.llm.state_dict().items() if "lora_" in k
    }
    if lora_state:
        state["llm_lora"] = lora_state
    torch.save(state, path)
    logger.info("Checkpoint saved → %s", path)


def _load_checkpoint(path: str, model, optimizer, scheduler, device: str):
    """Full resume (same run/config): restores optimizer + scheduler state too."""
    ckpt = torch.load(path, map_location=device)
    model.projector.load_state_dict(ckpt["projector"])
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if model.cls_head is not None and "cls_head" in ckpt:
        model.cls_head.load_state_dict(ckpt["cls_head"])
    if "llm_lora" in ckpt:
        model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    logger.info(
        "Resumed from %s (step=%d, epoch=%d, val_loss=%.4f)",
        path, ckpt["step"], ckpt["epoch"], ckpt.get("val_loss", float("inf")),
    )
    return ckpt["step"], ckpt["epoch"], ckpt.get("val_loss", float("inf"))


def _warm_start_weights(path: str, model, device: str) -> None:
    """Weights-only load for a stage transition (e.g. Stage 1 → Stage 2):
    projector/visual_proj/cls_head/LoRA weights carry over, but optimizer
    and scheduler start fresh since Stage 2 trains a different set of
    parameters at different LRs on a different schedule.

    strict=False + explicit key audit: attention-mechanism fixes added
    InterSliceAggregator parameters (attn_temperature, gamma_v_proj,
    beta_v_proj) that don't exist in checkpoints saved before those fixes.
    These are expected-missing (correctly initialized fresh, per each
    fix's own init scheme) — anything else missing is a real error.
    """
    ckpt = torch.load(path, map_location=device)
    result = model.projector.load_state_dict(ckpt["projector"], strict=False)
    allowed_missing_suffixes = [
        "attn_temperature",
        "gamma_v_proj.weight", "gamma_v_proj.bias",
        "beta_v_proj.weight", "beta_v_proj.bias",
    ]
    real_missing = [
        k for k in result.missing_keys
        if not any(k.endswith(s) for s in allowed_missing_suffixes)
    ]
    if real_missing:
        raise RuntimeError(f"Unexpected missing keys in projector checkpoint: {real_missing}")
    if result.unexpected_keys:
        logger.warning("Unexpected keys in projector checkpoint (ignored): %s", result.unexpected_keys)
    if result.missing_keys:
        logger.info("Expected-missing projector keys (fresh init, new since checkpoint was saved): %s",
                     result.missing_keys)
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if model.cls_head is not None and "cls_head" in ckpt:
        model.cls_head.load_state_dict(ckpt["cls_head"])
    if "llm_lora" in ckpt:
        model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    logger.info(
        "Warm-started weights from %s (step=%d, epoch=%d, val_loss=%.4f) — "
        "optimizer/scheduler NOT restored (stage transition).",
        path, ckpt["step"], ckpt["epoch"], ckpt.get("val_loss", float("inf")),
    )


# ── Validation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def _validate(
    model,
    val_loader,
    device: str,
    stage: Optional[int] = None,
    cls_loss_weight: float = 0.3,
    bce_loss_fn: Optional[torch.nn.Module] = None,
    max_batches: Optional[int] = None,
    max_length: int = 1024,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        features = batch["features"].to(device)
        instructions = batch["instruction"]

        if stage == 1:
            # Stage 1: aggregator-only, cls_loss is the whole objective — the
            # LLM (LoRA frozen) is never invoked, so an LM "val_loss" here
            # would be meaningless.
            out = model(features, instructions, training=True, compute_lm=False)
            labels_batch = batch["labels"].to(device)
            loss = bce_loss_fn(out["cls_logits"], labels_batch)
        else:
            target_enc = model.tokenizer(
                batch["target"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).input_ids.to(device)
            out = model(features, instructions, report_tokens=target_enc, training=True)
            loss = out["loss"]
            if model.cls_head is not None:
                labels_batch = batch["labels"].to(device)
                loss = loss + cls_loss_weight * bce_loss_fn(out["cls_logits"], labels_batch)

        total_loss += loss.item()
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
    # CT-RATE reports average ~201 tokens, max ~824 (see verification run) —
    # 256 was silently truncating most reports. Argus Appendix B uses 1024.
    max_length = cfg.get("max_length", 1024)

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

    # ── Two-stage training (Argus Section 5.1) ─────────────────────────────────
    # stage=1: aggregator + visual_proj + cls_head only, LoRA frozen, cls_loss
    #          only (LLM never invoked — see forward()'s compute_lm gate).
    # stage=2: everything unfrozen, lm_loss + cls_loss_weight * cls_loss.
    # stage unset (legacy configs, e.g. ctclip_stage2.yaml): unchanged joint
    # LM-only training, no cls_head at all.
    stage = cfg.get("stage")
    use_cls_head = stage in (1, 2)
    num_labels = None
    if use_cls_head:
        num_labels = len(train_ds.label_cols)
        cfg_num_labels = cfg.get("num_labels")
        assert cfg_num_labels is None or cfg_num_labels == num_labels, (
            f"num_labels mismatch: config says {cfg_num_labels} but the "
            f"dataset detected {num_labels} label columns: {train_ds.label_cols}"
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
        top_k=cfg.get("aggregator_top_k", 128),
        llm_model_name=cfg.get("llm_model_name", "facebook/opt-1.3b"),
        llm_frozen=cfg.get("llm_frozen", False),
        llm_lora=cfg.get("llm_lora"),
        instruction_encoder_model=cfg.get("instruction_encoder_model", "facebook/opt-1.3b"),
        num_labels=num_labels,
        device=device,
    )
    model.train()

    # ── Stage transition: warm-start weights from a prior stage's checkpoint ──
    resume_from = cfg.get("resume_from")
    if resume_from and Path(resume_from).exists():
        _warm_start_weights(resume_from, model, device)

    # ── Stage-specific LoRA freezing ────────────────────────────────────────────
    if stage == 1:
        for n, p in model.llm.named_parameters():
            if "lora_" in n:
                p.requires_grad = False
        logger.info("Stage 1: LoRA frozen — training aggregator + visual_proj + cls_head only.")
    elif stage == 2:
        for n, p in model.llm.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        logger.info("Stage 2: LoRA unfrozen.")

    # ── Trainable parameters ──────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info("Trainable parameters: %s", f"{n_trainable:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    from aadp.training.scheduler import get_cosine_schedule_with_warmup

    # Differential LR: the diagnostic in scripts/diagnose_mode_collapse.py
    # found LoRA's gradient norm ~19x larger than the InterSliceAggregator's
    # on the same backward pass — the aggregator (projector.stage2) +
    # visual_proj get their own, much higher LR so they aren't left behind
    # while LoRA races ahead and the LLM learns to ignore a still-weak
    # visual signal. aggregator_learning_rate defaults to 1e-3; "learning_rate"
    # continues to mean the LoRA/LLM-side rate, unchanged in meaning.
    aggregator_params = list(model.projector.stage2.parameters()) + list(
        model.visual_proj.parameters()
    )
    if model.cls_head is not None:
        aggregator_params += list(model.cls_head.parameters())
    aggregator_param_ids = {id(p) for p in aggregator_params}
    other_params = [p for p in trainable_params if id(p) not in aggregator_param_ids]

    aggregator_lr = cfg.get("aggregator_learning_rate", 1e-3)
    llm_lr = cfg.get("learning_rate", 1e-4)
    logger.info(
        "Optimizer param groups: aggregator+visual_proj+cls_head=%d tensors @ lr=%.1e, "
        "other (LoRA etc.)=%d tensors @ lr=%.1e",
        len(aggregator_params), aggregator_lr, len(other_params), llm_lr,
    )

    # Stage 1 has no "other" (LoRA) params at all — they're frozen out of
    # trainable_params above — so only build that param group when non-empty
    # (an empty-params AdamW group is at best pointless, at worst an error).
    param_groups = [{"params": aggregator_params, "lr": aggregator_lr}]
    if other_params:
        param_groups.append({"params": other_params, "lr": llm_lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    num_epochs = cfg.get("num_epochs", 5)
    steps_per_epoch = max(len(train_loader), 1)
    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    # scheduler.step() only fires once per grad_accum batches (see the
    # training loop below), not once per batch — num_training_steps must
    # count real optimizer steps, or a warmup_ratio-derived warmup_steps
    # would be grad_accum times too large relative to how often
    # scheduler.step() actually gets called.
    optimizer_steps_per_epoch = max(steps_per_epoch // grad_accum, 1)
    num_training_steps = num_epochs * optimizer_steps_per_epoch

    warmup_ratio = cfg.get("warmup_ratio")
    if warmup_ratio is not None:
        warmup_steps = int(warmup_ratio * num_training_steps)
    else:
        warmup_steps = cfg.get("warmup_steps", 500)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ── Classification loss (cls_head only) ─────────────────────────────────────
    cls_loss_weight = cfg.get("cls_loss_weight", 0.3)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss() if model.cls_head is not None else None

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
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    val_every = cfg.get("val_every_n_steps", 500)
    save_every = cfg.get("save_every_n_steps", 1000)
    patience = cfg.get("patience", 5)
    patience_counter = 0

    logger.info(
        "Starting training: %d epochs, %d steps/epoch, grad_accum=%d, amp=%s, "
        "warmup_steps=%d, num_training_steps=%d",
        num_epochs, steps_per_epoch, grad_accum, use_amp,
        warmup_steps, num_training_steps,
    )

    optimizer.zero_grad()

    for epoch in range(start_epoch, num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            instructions = batch["instruction"]

            with autocast_ctx():
                if stage == 1:
                    # Aggregator-only: skip the LLM entirely, cls_loss is the
                    # whole objective (LoRA frozen ⇒ LM loss would be
                    # meaningless and wasted compute).
                    out = model(features, instructions, training=True, compute_lm=False)
                    labels_batch = batch["labels"].to(device)
                    raw_loss_t = bce_loss_fn(out["cls_logits"], labels_batch)
                else:
                    # Tokenize targets (report text / entity answer / yes-no)
                    target_enc = model.tokenizer(
                        batch["target"],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).input_ids.to(device)
                    out = model(
                        features, instructions, report_tokens=target_enc, training=True
                    )
                    raw_loss_t = out["loss"]
                    if model.cls_head is not None:
                        labels_batch = batch["labels"].to(device)
                        raw_loss_t = raw_loss_t + cls_loss_weight * bce_loss_fn(
                            out["cls_logits"], labels_batch
                        )
                loss = raw_loss_t / grad_accum

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
                    val_loss = _validate(
                        model, val_loader, device,
                        stage=stage, cls_loss_weight=cls_loss_weight,
                        bce_loss_fn=bce_loss_fn, max_batches=50,
                        max_length=max_length,
                    )
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
