"""Trainer — gradient-accumulation training loop for MedVLM."""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aadp.models.vlm import MedVLM, variable_depth_collate_fn
from aadp.training.losses import CombinedLoss, NextTokenLoss

log = logging.getLogger(__name__)

_DEFAULT_INSTRUCTION = "Generate a radiology report for this CT scan."


class Trainer:
    """Training loop for MedVLM with gradient accumulation and mixed precision.

    Args:
        model:         Assembled ``MedVLM``.
        optimizer:     Optimizer over projector (and optional visual_proj) params.
        scheduler:     ``LambdaLR`` returned by ``get_cosine_schedule_with_warmup``.
        loss_fn:       ``NextTokenLoss`` for explicit logit-level loss computation.
        train_dataset: Iterable dataset yielding ``(vol, report_text, labels, pid)``
                       tuples — same interface as ``CTRATEDataset``.
        val_dataset:   Same interface as ``train_dataset``.
        config:        Dict of hyperparameters (see key list below).
        device:        Training device. Default ``"cuda"``.
        use_wandb:     Log to Weights & Biases if available. Default ``True``.

    Config keys consumed:
        batch_size, gradient_accumulation_steps, max_grad_norm, num_epochs,
        val_every_n_steps, save_every_n_steps, checkpoint_dir, use_attn_loss,
        mixed_precision, patience, max_report_length.
    """

    def __init__(
        self,
        model: MedVLM,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        loss_fn: NextTokenLoss,
        train_dataset,
        val_dataset,
        config: Dict[str, Any],
        device: torch.device = torch.device("cuda"),
        use_wandb: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.use_wandb = use_wandb

        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self._no_improve_rounds: int = 0

        # Optional aux loss for use_attn_loss=True runs
        lambda_attn = config.get("lambda_attn", 0.1)
        self._combined_loss = CombinedLoss(lambda_attn=lambda_attn)

    # ── DataLoader ────────────────────────────────────────────────────────────

    def _make_collate_fn(self):
        """Convert (vol, report, labels, pid) tuples into MedVLM-ready dicts."""
        tokenizer = self.model._llm_tokenizer
        max_len = self.config.get("max_report_length", 256)

        def _collate(batch):
            items = []
            for vol, report_text, label_dict, patient_id in batch:
                enc = tokenizer(
                    report_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                    padding=False,
                )
                report_ids = enc["input_ids"][0]  # (L,)
                items.append(
                    {
                        "volumes": vol,
                        "instructions": _DEFAULT_INSTRUCTION,
                        "report_tokens": report_ids,
                        "depth_spacing_mm": None,
                        "label_dict": label_dict,
                        "patient_id": patient_id,
                    }
                )
            return variable_depth_collate_fn(items)

        return _collate

    def _build_loader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 2),
            collate_fn=self._make_collate_fn(),
        )

    # ── WandB helper ──────────────────────────────────────────────────────────

    def _log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.use_wandb:
            return
        try:
            import wandb
            wandb.log(metrics, step=step if step is not None else self.global_step)
        except ImportError:
            pass

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop.

        Returns:
            History dict with ``"train_loss"`` and ``"val_loss"`` lists.
        """
        grad_accum = self.config.get("gradient_accumulation_steps", 1)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        num_epochs = self.config.get("num_epochs", 10)
        val_every = self.config.get("val_every_n_steps", 250)
        save_every = self.config.get("save_every_n_steps", 500)
        patience = self.config.get("patience", 5)
        mixed_precision = self.config.get("mixed_precision", False)
        use_attn_loss = self.config.get("use_attn_loss", False)

        train_loader = self._build_loader(self.train_dataset)
        scaler = torch.amp.GradScaler("cuda") if mixed_precision else None

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            for batch_idx, batch in enumerate(train_loader):
                volumes = batch["volumes"].to(self.device)
                instructions = batch["instructions"]
                report_tokens = batch.get("report_tokens")
                if report_tokens is not None:
                    report_tokens = report_tokens.to(self.device)
                depth_spacing_mm = batch.get("depth_spacing_mm")

                # ── Forward ──────────────────────────────────────────────────
                with torch.amp.autocast("cuda", enabled=mixed_precision):
                    output = self.model(
                        volumes, instructions, report_tokens, depth_spacing_mm
                    )
                    lm_loss = output.loss

                    # Optional attention alignment auxiliary loss
                    if use_attn_loss:
                        try:
                            attn = self.model.projector.get_slice_attention()
                        except (AttributeError, RuntimeError):
                            attn = None
                        lm_loss = self._combined_loss(
                            lm_loss,
                            attn_weights=attn,
                            gt_slice_indices=None,  # no GT at CT-RATE stage
                            use_attn_loss=False,    # skip without GT indices
                        )

                    loss = lm_loss / grad_accum

                # ── Backward ─────────────────────────────────────────────────
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # ── Optimizer step every grad_accum micro-batches ─────────
                if (batch_idx + 1) % grad_accum == 0:
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    step_loss = loss.item() * grad_accum
                    history["train_loss"].append(step_loss)
                    current_lr = self.scheduler.get_last_lr()[0]

                    log.info(
                        "epoch=%d step=%d loss=%.4f lr=%.2e",
                        epoch, self.global_step, step_loss, current_lr,
                    )
                    self._log(
                        {"train/loss": step_loss, "train/lr": current_lr},
                        step=self.global_step,
                    )

                    # ── Validation ───────────────────────────────────────────
                    if self.global_step % val_every == 0:
                        val_metrics = self.validate()
                        val_loss = val_metrics["val_loss"]
                        history["val_loss"].append(val_loss)

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._no_improve_rounds = 0
                            self.save_checkpoint(filename="checkpoint_best.pt")
                        else:
                            self._no_improve_rounds += 1
                            if self._no_improve_rounds >= patience:
                                log.warning(
                                    "Val loss has not improved for %d consecutive "
                                    "validation rounds (best=%.4f, current=%.4f). "
                                    "Consider stopping early.",
                                    patience, self.best_val_loss, val_loss,
                                )

                    # ── Checkpoint ───────────────────────────────────────────
                    if self.global_step % save_every == 0:
                        self.save_checkpoint()

        return history

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> Dict[str, float]:
        """Run one pass over val_dataset and return ``{"val_loss": float}``."""
        val_loader = self._build_loader(self.val_dataset)

        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                volumes = batch["volumes"].to(self.device)
                instructions = batch["instructions"]
                report_tokens = batch.get("report_tokens")
                if report_tokens is not None:
                    report_tokens = report_tokens.to(self.device)
                depth_spacing_mm = batch.get("depth_spacing_mm")

                output = self.model(
                    volumes, instructions, report_tokens, depth_spacing_mm
                )
                total_loss += output.loss.item()
                count += 1

        self.model.train()

        val_loss = total_loss / max(count, 1)
        log.info("step=%d val_loss=%.4f", self.global_step, val_loss)
        self._log({"val/loss": val_loss}, step=self.global_step)
        return {"val_loss": val_loss}

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, filename: Optional[str] = None) -> str:
        """Persist projector weights, optimizer, and scheduler state.

        Args:
            filename: Override checkpoint filename.  Default is
                      ``checkpoint_step_{step}.pt``.

        Returns:
            Absolute path of the saved file as a string.
        """
        ckpt_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt: Dict[str, Any] = {
            "projector": self.model.projector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.global_step,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
        }
        if not isinstance(self.model.visual_proj, nn.Identity):
            ckpt["visual_proj"] = self.model.visual_proj.state_dict()

        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"
        path = ckpt_dir / filename
        torch.save(ckpt, path)
        log.info("Saved checkpoint: %s", path)
        return str(path)

    def load_checkpoint(self, path: str) -> int:
        """Load projector weights, optimizer, and scheduler from a checkpoint.

        Args:
            path: Path to ``.pt`` checkpoint file.

        Returns:
            The training step at which the checkpoint was saved.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.projector.load_state_dict(ckpt["projector"])

        if "visual_proj" in ckpt and not isinstance(
            self.model.visual_proj, nn.Identity
        ):
            self.model.visual_proj.load_state_dict(ckpt["visual_proj"])

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["step"]
        self.current_epoch = ckpt["epoch"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        log.info("Loaded checkpoint from %s (step=%d)", path, self.global_step)
        return self.global_step
