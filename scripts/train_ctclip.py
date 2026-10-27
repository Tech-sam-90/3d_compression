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


def _warm_start_weights(
    path: str, model, device: str, drop_keys: Optional[List[str]] = None
) -> None:
    """Weights-only load for a stage transition (e.g. Stage 1 → Stage 2):
    projector/visual_proj/cls_head/LoRA weights carry over, but optimizer
    and scheduler start fresh since Stage 2 trains a different set of
    parameters at different LRs on a different schedule.

    strict=False + explicit key audit: attention-mechanism fixes added
    InterSliceAggregator parameters (attn_temperature, gamma_v_proj,
    beta_v_proj) that don't exist in checkpoints saved before those fixes.
    These are expected-missing (correctly initialized fresh, per each
    fix's own init scheme) — anything else missing is a real error.

    Also covers warm-starting the attention-conditioned ablation
    (conditioning="attention", AttentionConditionedInterSliceAggregator)
    from a FiLM-based checkpoint: cross_attn/depth_queries/depth_pos_enc/
    norm_kv/attn_temperature are shared by both aggregator classes and
    transfer correctly, but text_proj/cond_cross_attn/norm_q/norm_text are
    genuinely new (expected-missing, fresh init) while the old checkpoint's
    film.*/gamma_v_proj.*/beta_v_proj.* keys become unexpected (already
    tolerated below — the attention-conditioned model has no FiLM at all).

    Args:
        drop_keys: Optional list of substrings. Any projector state_dict
            key containing one of these is dropped from the checkpoint
            before loading, so that parameter keeps its fresh __init__
            value instead of the checkpoint's — it still shows up in
            missing_keys below (already-allowed for gamma_v_proj/
            beta_v_proj) rather than being silently overwritten. Used by
            configs/ctclip_vfilm_fix.yaml (warm_start_drop_keys:
            ["gamma_v_proj"]) so gamma_v_proj — trained under the old,
            unbounded gamma+beta V-FiLM formula that
            check_vfilm_scan_diversity.py showed collapses cross-scan
            direction — restarts from its new zero-init under the bounded
            tanh formula (aadp/models/projector/stage2.py) instead of
            carrying over stale weights, while beta_v_proj (removed from
            the architecture entirely) is already dropped automatically
            via unexpected_keys below, and everything else (LoRA,
            visual_proj, cross_attn, Q-FiLM, cosine attention temperature)
            still warm-starts normally.
    """
    ckpt = torch.load(path, map_location=device)
    projector_state = ckpt["projector"]
    if drop_keys:
        projector_state = {
            k: v for k, v in projector_state.items()
            if not any(dk in k for dk in drop_keys)
        }
    result = model.projector.load_state_dict(projector_state, strict=False)
    allowed_missing_suffixes = [
        "attn_temperature",
        "gamma_v_proj.weight", "gamma_v_proj.bias",
        "beta_v_proj.weight", "beta_v_proj.bias",
        "text_proj.weight", "text_proj.bias",
        "cond_cross_attn.in_proj_weight", "cond_cross_attn.in_proj_bias",
        "cond_cross_attn.out_proj.weight", "cond_cross_attn.out_proj.bias",
        "norm_q.weight", "norm_q.bias",
        "norm_text.weight", "norm_text.bias",
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


# ── End-to-end (raw-volume, DeepSpeed ZeRO-3) training ─────────────────────────


@torch.no_grad()
def _validate_e2e(model_engine, model, val_loader, max_length: int, max_batches: Optional[int] = 20) -> float:
    """Collective — every rank must call this and iterate val_loader together,
    since ZeRO-3's parameter all-gather inside model_engine() is a collective
    op regardless of which rank "owns" the result."""
    model_engine.eval()
    total_loss = 0.0
    n_batches = 0
    _TASKS = ("T1", "T2", "T3")
    task_loss_sums = {t: 0.0 for t in _TASKS}
    task_counts = {t: 0 for t in _TASKS}
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        volume = batch["volume"].to(device=model_engine.device, dtype=torch.bfloat16)
        instructions = batch["instruction"]
        report_tokens = model.tokenizer(
            batch["target"], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).input_ids.to(model_engine.device)

        out = model_engine(volume=volume, instructions=instructions,
                            report_tokens=report_tokens, training=True)
        loss_val = out["loss"].item()
        total_loss += loss_val
        n_batches += 1
        task = batch["task"][0]
        if task in task_loss_sums:
            task_loss_sums[task] += loss_val
            task_counts[task] += 1
    model_engine.train()

    # val_loader's DistributedSampler(shuffle=False) gives every rank a
    # DIFFERENT quarter of the val set — without this all_reduce, each rank
    # returns its own local average and the caller's `if val_loss <
    # best_val_loss` decision can disagree across ranks. Since that decision
    # gates a collective save_checkpoint() call, disagreement means some
    # ranks enter the collective while others don't — a rank-count mismatch
    # that hangs until NCCL's watchdog kills the job. This caused two
    # reproducible crashes (jobs 66999932, 67025026), confirmed by
    # checkpoint_best's on-disk shards showing different ranks writing at
    # different steps despite rank 0's log claiming no improvement at the
    # later step. Sum-then-divide (not mean-of-means) so per-rank batch
    # count differences (val_ds not evenly divisible by world size) don't
    # bias the result.
    #
    # Extended with 6 more entries (per-task sum + count) purely for logging
    # visibility below — val_loss itself (returned to the caller, gates the
    # best_val_loss/checkpoint_best decision) is computed exactly as before
    # from the first two entries only; this doesn't change that value.
    stats = torch.tensor(
        [total_loss, float(n_batches)]
        + [task_loss_sums[t] for t in _TASKS]
        + [float(task_counts[t]) for t in _TASKS],
        device=model_engine.device,
    )
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    global_total_loss, global_n_batches = stats[0].item(), stats[1].item()
    global_task_sums = {t: stats[2 + i].item() for i, t in enumerate(_TASKS)}
    global_task_counts = {t: int(stats[5 + i].item()) for i, t in enumerate(_TASKS)}

    if model_engine.global_rank == 0:
        task_str = "  ".join(
            f"{t.lower()}={global_task_sums[t] / global_task_counts[t]:.4f}(n={global_task_counts[t]})"
            if global_task_counts[t] > 0 else f"{t.lower()}=n/a(n={global_task_counts[t]})"
            for t in _TASKS
        )
        logger.info("  [val] per-task: %s", task_str)

    return global_total_loss / max(global_n_batches, 1)


def _train_e2e(cfg: Dict, args) -> None:
    """DeepSpeed ZeRO-3 training loop for CTCLIPStage2VLM's raw-volume
    (volume=) path — CTRawDataset + set_training_stage()/get_parameter_groups()
    from aadp/models/ctclip_vlm.py, not the pre-extracted-features path above.
    """
    import deepspeed
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    from aadp.data.ctclip_raw_dataset import CTRawDataset, collate_fn
    from aadp.models.ctclip_vlm import CTCLIPStage2VLM

    # CTViT hardcodes device=torch.device('cuda') internally in several spots
    # (aadp/models/ctclip_vlm.py's _CTVIT_HPARAMS comment) — no CPU fallback
    # is possible for this path.
    assert torch.cuda.is_available(), "e2e path requires CUDA (CTViT hardcodes it internally)."

    # DeepSpeed's launcher spawns one process per GPU and sets LOCAL_RANK per
    # process, but never hit this until scaling past 1 GPU: with a bare
    # "cuda" device string, torch.device("cuda") resolves to the process's
    # *current default* device, which is GPU 0 for every rank until
    # something calls set_device() — so all N ranks piled their full model
    # onto GPU 0 simultaneously and OOM'd instantly on a real 4-GPU
    # submission (CUDA out of memory citing all 4 PIDs' allocations on "GPU
    # 0"), even though this is a 1-GPU-per-rank workload that should never
    # come close to the 40GB limit per GPU. Must set_device() before any
    # CUDA allocation happens.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device_str = f"cuda:{local_rank}"

    training_stage = cfg.get("training_stage", "stage2")
    max_length = cfg.get("max_length", 1024)
    task_weights = cfg.get("task_weights", {"T1": 0.5, "T2": 0.35, "T3": 0.15})
    target_shape = tuple(cfg.get("target_shape", [240, 480, 480]))

    # ── Datasets ─────────────────────────────────────────────────────────────
    logger.info("Loading raw-volume training dataset from %s", cfg["ct_volume_root"])
    train_ds = CTRawDataset(
        merged_csv=cfg["ctrate_csv_train"],
        ct_volume_root=cfg["ct_volume_root"],
        split="train",
        target_shape=target_shape,
        task_weights=task_weights,
    )
    val_ds = CTRawDataset(
        merged_csv=cfg["ctrate_csv_valid"],
        ct_volume_root=cfg["ct_volume_root"],
        split="val",   # CTRawDataset's split values are "train"/"val", not "valid"
        target_shape=target_shape,
        task_weights=task_weights,
    )
    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building CTCLIPStage2VLM (llm=%s, training_stage=%s)...",
                 cfg.get("llm_model_name"), training_stage)
    model = CTCLIPStage2VLM(
        ctclip_dim=cfg.get("ctclip_dim", 512),
        embed_dim=cfg.get("embed_dim", 512),
        num_tokens=cfg.get("num_tokens", 64),
        num_heads=cfg.get("num_heads", 8),
        cond_dim=cfg.get("cond_dim", 3072),
        use_film=cfg.get("use_film", True),
        max_depth=cfg.get("max_depth", 24),
        dropout=cfg.get("dropout", 0.0),
        top_k=cfg.get("aggregator_top_k", 128),
        conditioning=cfg.get("aggregator_conditioning", "film"),
        llm_model_name=cfg.get("llm_model_name", "facebook/opt-1.3b"),
        llm_frozen=cfg.get("llm_frozen", False),
        llm_lora=cfg.get("llm_lora"),
        instruction_encoder_model=cfg.get("instruction_encoder_model", "facebook/opt-1.3b"),
        device=device_str,
        training_stage=training_stage,
        ctclip_checkpoint=cfg.get("ctclip_checkpoint"),
    )
    # set_training_stage(training_stage) already ran inside __init__ — do NOT
    # manually freeze/unfreeze anything here.

    # ── Stage transition: warm-start from stage1's consolidated checkpoint ────
    # DeepSpeed ZeRO-3 checkpoints (save_checkpoint() below) are sharded
    # directories, not directly loadable via plain torch.load()+
    # load_state_dict(). gather_16bit_weights_on_model_save (configs/
    # ds_zero3.json) doesn't make save_checkpoint() write a flat file
    # automatically — it only enables the separate save_16bit_model() call,
    # which _train_e2e calls explicitly below at end of stage1. resume_from
    # must point at THAT consolidated file (see configs/ctclip_e2e_stage2.yaml).
    #
    # visual_encoder is loaded independently above from ctclip_checkpoint=
    # (same source both stages, frozen and unchanged throughout stage1) — so
    # it's never actually missing from this warm-start's state dict at all;
    # the split below is kept as a defensive/informational check, not because
    # it's expected to find anything today.
    warm_start_ckpt = cfg.get("resume_from")
    if warm_start_ckpt and Path(warm_start_ckpt).exists():
        ckpt = torch.load(warm_start_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        result = model.load_state_dict(state, strict=False)
        vit_missing = [k for k in result.missing_keys if "visual_encoder" in k]
        other_missing = [k for k in result.missing_keys if "visual_encoder" not in k]
        if other_missing:
            raise RuntimeError(f"Unexpected missing keys in warm-start checkpoint: {other_missing}")
        logger.info(
            "Warm-start OK from %s — ViT keys not in ckpt (expected, independently loaded "
            "from ctclip_checkpoint): %d, unexpected keys (ignored): %d",
            warm_start_ckpt, len(vit_missing), len(result.unexpected_keys),
        )
    elif warm_start_ckpt:
        logger.warning(
            "resume_from=%s does not exist — skipping warm-start, projector/visual_proj "
            "keep their fresh random init. Expected if stage1 hasn't produced a "
            "consolidated checkpoint yet (e.g. a --dry-run, which never saves).",
            warm_start_ckpt,
        )

    # ── Optimizer (per-component LR via get_parameter_groups()) ────────────────
    # DeepSpeedCPUAdam, not torch.optim.AdamW: ds_zero3.json offloads the
    # optimizer to CPU, and DeepSpeed refuses a plain client-provided
    # optimizer under CPU offload (raises ZeRORuntimeException) unless
    # explicitly overridden — DeepSpeedCPUAdam is the fused, CPU-offload-aware
    # replacement it expects; get_parameter_groups()'s output (a plain list of
    # {"params", "lr"} dicts) works with either, so this doesn't touch
    # aadp/models/ctclip_vlm.py at all.
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    param_groups = model.get_parameter_groups(
        stage1_lr=cfg.get("stage1_lr", 1e-4),
        vit_lr=cfg.get("vit_lr", 1e-5),
        llm_lr=cfg.get("llm_lr", 2e-5),
    )
    optimizer = DeepSpeedCPUAdam(param_groups, weight_decay=cfg.get("weight_decay", 0.01))

    # ── DeepSpeed engine ─────────────────────────────────────────────────────
    ds_config_path = cfg.get("deepspeed_config", "configs/ds_zero3.json")
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        training_data=train_ds,
        collate_fn=collate_fn,
        config=ds_config_path,
    )

    # HuggingFace's built-in gradient checkpointing (torch.utils.checkpoint,
    # enabled unconditionally in CTCLIPStage2VLM.__init__ for the feature-
    # based path's memory budget) is not ZeRO-3-partition-aware: ZeRO-3
    # frees/reshards a parameter right after its forward use, so
    # checkpointing's recompute-on-backward pass can see a different
    # (emptied) tensor than the one saved during forward
    # (torch.utils.checkpoint.CheckpointError: Recomputed values ... have
    # different metadata). DeepSpeed's own deepspeed.checkpointing API is
    # partition-aware and is the real fix, but isn't wired up here — for now
    # rely on ZeRO-3's own memory savings (bf16 weights + CPU-offloaded
    # optimizer state) instead of layering incompatible checkpointing on top.
    model_engine.module.llm.gradient_checkpointing_disable()

    # DeepSpeed's bf16 mode sweeps every floating buffer in the model,
    # including CTViT's VQ codebook (visual_encoder.vq._codebook.embed /
    # cluster_size). vector_quantize_pytorch's CosineSimCodebook.forward()
    # unconditionally does x = x.float() on its input but uses self.embed
    # as-is (vector_quantize_pytorch/vector_quantize_pytorch.py, class
    # CosineSimCodebook) — so a bf16 embed always mismatches the forced-fp32
    # activation inside that function, regardless of what dtype the caller
    # passes in. This buffer was never meant to be swept: fp32 is the
    # library's own assumption for it, not a training-precision choice, so
    # restore it (not a workaround for a bug in our code — the bug is in
    # applying blanket bf16 casting to a buffer a third-party library
    # hardcodes to fp32 internally).
    codebook = model_engine.module.visual_encoder.vq._codebook
    codebook.embed.data = codebook.embed.data.float()
    if hasattr(codebook, "cluster_size"):
        codebook.cluster_size.data = codebook.cluster_size.data.float()

    val_batch_size = cfg.get("train_micro_batch_size_per_gpu", 1)
    val_sampler = DistributedSampler(val_ds, shuffle=False) if dist.is_initialized() else None
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, sampler=val_sampler, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints/ctclip_e2e/"))
    if model_engine.global_rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    val_every = cfg.get("val_every_n_steps", 500)
    save_every = cfg.get("save_every_n_steps", 1000)
    num_epochs = cfg.get("num_epochs", 1)
    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0

    # ── Resume mid-stage progress after a crash ─────────────────────────────
    # DeepSpeed writes ckpt_dir/latest itself on every save_checkpoint() call
    # (pointing at the most-recently-saved tag) — its presence is exactly the
    # signal that there's real progress here to resume rather than start
    # over. Skipped for --dry-run so smoke tests stay isolated from whatever
    # real training state happens to already live in the same checkpoint_dir.
    # Must be called here, right after deepspeed.initialize() while the model
    # is still pristine — DeepSpeed's own load_checkpoint() docstring warns
    # against calling it right after save_checkpoint() on a partitioned
    # (post-training-step) ZeRO-3 module.
    if not args.dry_run and (ckpt_dir / "latest").exists():
        _, client_sd = model_engine.load_checkpoint(str(ckpt_dir))
        # model_engine.global_steps is DeepSpeed's own counter, incremented
        # once per gradient-accumulation boundary (same granularity as this
        # loop's own global_step) and restored automatically by
        # load_checkpoint() regardless of client_state — the reliable
        # fallback for checkpoints saved before client_state (below) existed,
        # e.g. a checkpoint_best saved by an earlier version of this script.
        global_step = model_engine.global_steps
        if client_sd:
            best_val_loss = client_sd.get("best_val_loss", best_val_loss)
            start_epoch = client_sd.get("epoch", start_epoch)
        if model_engine.global_rank == 0:
            logger.info(
                "Resumed from %s: global_step=%d, best_val_loss=%.4f, start_epoch=%d "
                "(best_val_loss/start_epoch fall back to defaults when the checkpoint "
                "predates client_state tracking)",
                ckpt_dir, global_step, best_val_loss, start_epoch,
            )

    logger.info(
        "Starting e2e training: stage=%s, %d epoch(s), max_steps=%s, dry_run=%s, "
        "start_epoch=%d, global_step=%d",
        training_stage, num_epochs, args.max_steps, args.dry_run, start_epoch, global_step,
    )

    model_engine.train()
    stop = False
    # Per-task loss visibility only — does not touch backward()/step().
    # Each micro-batch belongs to exactly one task (CTRawDataset samples one
    # task per __getitem__ call), so there is no existing "sum of T1+T2+T3"
    # to decompose; instead this tracks which task each of the
    # gradient_accumulation_steps micro-batches in a window belonged to and
    # reports the per-task mean over whichever tasks actually appeared.
    # Per-rank-local (not all-reduced across ranks), matching the existing
    # loss= log line's own convention.
    _TASKS = ("T1", "T2", "T3")
    window_loss_sum = 0.0
    window_micro_count = 0
    task_loss_sums = {t: 0.0 for t in _TASKS}
    task_micro_counts = {t: 0 for t in _TASKS}
    for epoch in range(start_epoch, num_epochs):
        if stop:
            break
        for batch in train_loader:
            # DeepSpeed's bf16 mode (ds_zero3.json) casts the model's own
            # weights to bf16 but does not auto-cast raw input tensors —
            # CTViT's first LayerNorm otherwise sees a bf16 weight against
            # an fp32 input and raises.
            volume = batch["volume"].to(device=model_engine.device, dtype=torch.bfloat16)
            instructions = batch["instruction"]
            report_tokens = model.tokenizer(
                batch["target"], return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).input_ids.to(model_engine.device)

            out = model_engine(volume=volume, instructions=instructions,
                                report_tokens=report_tokens, training=True)
            loss = out["loss"]

            model_engine.backward(loss)
            model_engine.step()

            # Extra .item() call vs. before (was only called once per
            # accumulation window, at the log line) — needed to attribute
            # each micro-batch's loss to its task. Logging only; the tensor
            # used for backward() above is untouched.
            loss_val = loss.item()
            window_loss_sum += loss_val
            window_micro_count += 1
            task = batch["task"][0]
            if task in task_loss_sums:
                task_loss_sums[task] += loss_val
                task_micro_counts[task] += 1

            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1
                if model_engine.global_rank == 0:
                    mean_loss = window_loss_sum / max(window_micro_count, 1)
                    task_str = "  ".join(
                        f"{t.lower()}={task_loss_sums[t] / task_micro_counts[t]:.4f}(n={task_micro_counts[t]})"
                        if task_micro_counts[t] > 0 else f"{t.lower()}=n/a(n={task_micro_counts[t]})"
                        for t in _TASKS
                    )
                    logger.info("epoch=%d  step=%d  loss=%.4f(mean)  %s",
                                 epoch + 1, global_step, mean_loss, task_str)
                # Reset for the next accumulation window — every rank resets
                # its own local accumulators regardless of which rank logs.
                window_loss_sum = 0.0
                window_micro_count = 0
                task_loss_sums = {t: 0.0 for t in _TASKS}
                task_micro_counts = {t: 0 for t in _TASKS}

                if not args.dry_run and val_every and global_step % val_every == 0:
                    val_loss = _validate_e2e(model_engine, model, val_loader, max_length)
                    if model_engine.global_rank == 0:
                        logger.info("  [val] step=%d  val_loss=%.4f  best=%.4f",
                                     global_step, val_loss, best_val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model_engine.save_checkpoint(
                            str(ckpt_dir), tag="checkpoint_best",
                            client_state={"best_val_loss": best_val_loss, "epoch": epoch},
                        )
                        # Flat consolidated state dict alongside the ZeRO-3
                        # shards so eval scripts can load without spinning up
                        # a DeepSpeed engine — same two-arg (dir, filename)
                        # convention as the stage1-end save below. Collective
                        # call: every rank must enter, only rank 0 writes.
                        model_engine.save_16bit_model(
                            str(ckpt_dir / "checkpoint_best"), "pytorch_model.bin"
                        )

                if (not args.dry_run and save_every and global_step % save_every == 0
                        and global_step % val_every != 0):
                    # Skip when val_every also just fired this step (both
                    # divide evenly into step 1000 with the current config) —
                    # stacking a full validation pass immediately followed by
                    # a ~13GB, 4-rank collective checkpoint write is what
                    # caused the reproducible hangs above; checkpoint_best
                    # already covers this step if val_loss improved, and
                    # skipping this one specific periodic snapshot if it
                    # didn't is a fine tradeoff against reproducibly hanging
                    # the whole job.
                    model_engine.save_checkpoint(
                        str(ckpt_dir), tag=f"step_{global_step}",
                        client_state={"best_val_loss": best_val_loss, "epoch": epoch},
                    )

                if args.max_steps is not None and global_step >= args.max_steps:
                    logger.info("Reached --max-steps=%d, stopping.", args.max_steps)
                    stop = True
                    break
        if stop:
            break

    if not args.dry_run:
        model_engine.save_checkpoint(
            str(ckpt_dir), tag="checkpoint_latest",
            client_state={"best_val_loss": best_val_loss, "epoch": num_epochs},
        )

        if training_stage == "stage1":
            # Consolidated flat state dict for stage2's warm-start (see the
            # resume_from comment above) — save_checkpoint() alone only
            # writes ZeRO-3 shards, not something plain torch.load() can
            # feed to model.load_state_dict(). Collective call (all ranks
            # must enter, even though only rank 0 writes).
            model_engine.save_16bit_model(str(ckpt_dir), "checkpoint_latest_consolidated.pt")
            if model_engine.global_rank == 0:
                logger.info(
                    "Saved consolidated stage1 checkpoint for stage2 warm-start → %s",
                    ckpt_dir / "checkpoint_latest_consolidated.pt",
                )

    logger.info("E2E training complete (stage=%s). Best val_loss=%.4f", training_stage, best_val_loss)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CT-CLIP Stage 2 VLM")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--set", nargs="*", dest="overrides", metavar="KEY=VALUE",
        help="Override config keys, e.g. --set num_tokens=128 use_wandb=false",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Stop after this many optimizer steps (e2e/DeepSpeed path only). For smoke-testing.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="e2e/DeepSpeed path only: skip validation and checkpoint saving, just verify the loop runs.",
    )
    # DeepSpeed's launcher injects --local_rank; unused here (DeepSpeedEngine reads
    # LOCAL_RANK from the environment itself), accepted so argparse doesn't reject it.
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg = _apply_overrides(cfg, args.overrides)

    # ── End-to-end (raw-volume, DeepSpeed ZeRO-3) path ──────────────────────────
    # Distinguished from the feature-based path below by ct_volume_root — only
    # configs/ctclip_e2e_stage{1,2}.yaml set it; every existing config uses
    # features_train_dir instead. Kept as a fully separate function rather than
    # branching deep inside the existing loop: set_training_stage() supersedes
    # that loop's stage==1/stage==2 LoRA-freeze logic entirely here (llm_lora is
    # unset in the e2e configs — full fine-tuning), and DeepSpeed owns the
    # optimizer step / grad accumulation / checkpoint sharding, so interleaving
    # it with the existing AdamW+GradScaler+manual-accum loop would just be two
    # different training mechanisms fighting over the same model.
    if "ct_volume_root" in cfg:
        _train_e2e(cfg, args)
        return

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
        conditioning=cfg.get("aggregator_conditioning", "film"),
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
        _warm_start_weights(resume_from, model, device, drop_keys=cfg.get("warm_start_drop_keys"))

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
