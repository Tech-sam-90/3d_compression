"""Trainer — gradient-accumulation training loop for MedVLM."""

import logging
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aadp.data.instruction_builder import build_instructions
from aadp.models.vlm import MedVLM, variable_depth_collate_fn
from aadp.training.losses import CombinedLoss, NextTokenLoss

log = logging.getLogger(__name__)


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
        on_checkpoint_saved: Optional callback invoked with the saved checkpoint's
                       path after every ``save_checkpoint()`` call (step-based,
                       best-val, and end-of-epoch alike). Lets a caller mirror
                       checkpoints to external storage (e.g. an rclone copy to a
                       Drive remote) without this module needing to know
                       anything about where that storage lives. A raised
                       exception from the callback is logged and swallowed —
                       a sync failure must never abort training.

    Config keys consumed:
        batch_size, gradient_accumulation_steps, max_grad_norm, num_epochs,
        val_every_n_steps, save_every_n_steps, checkpoint_dir, use_attn_loss,
        mixed_precision, patience, max_report_length.

    Checkpointing:
        In addition to the step-based ``save_every_n_steps`` / ``val_every_n_steps``
        cadence, ``train()`` unconditionally validates and checkpoints at the end
        of every epoch — writing ``checkpoint_epoch_{N}.pt`` and overwriting
        ``checkpoint_latest.pt`` — so a run always has a clean, resumable
        checkpoint after each completed epoch regardless of how the step-based
        cadence is configured. ``current_epoch`` is advanced to ``N + 1`` before
        that save, so resuming from it via ``load_checkpoint()`` + ``train()``
        continues with the next epoch rather than repeating the one just saved.
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
        on_checkpoint_saved: Optional[Callable[[str], None]] = None,
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
        self.on_checkpoint_saved = on_checkpoint_saved

        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self._no_improve_rounds: int = 0

        # Optional aux loss for use_attn_loss=True runs
        lambda_attn = config.get("lambda_attn", 0.1)
        self._combined_loss = CombinedLoss(lambda_attn=lambda_attn)

    # ── DataLoader ────────────────────────────────────────────────────────────

    def _make_collate_fn(self):
        """Convert dataset samples into MedVLM-ready dicts.

        Accepts two sample formats:
          * ``MultiTaskCTRATEDataset`` dicts with ``volume``/``instruction``/
            ``target`` keys (per-sample instruction already chosen), and
          * raw ``CTRATEDataset`` ``(vol, report, labels, pid)`` tuples, which
            are converted on the fly by sampling one of the four instruction
            types via :func:`build_instructions`.

        Either way each sample contributes one ``(instruction, target)`` pair,
        so a batch contains a mix of instruction types.
        """
        tokenizer = self.model._llm_tokenizer
        max_len = self.config.get("max_report_length", 256)

        def _collate(batch):
            items = []
            gt_slice_indices: List[Optional[List[int]]] = []
            for sample in batch:
                if isinstance(sample, dict):
                    vol = sample["volume"]
                    instruction = sample["instruction"]
                    target_text = sample["target"]
                    patient_id = sample.get("patient_id")
                    gt_slice_indices.append(sample.get("gt_slice_indices"))
                else:
                    vol, report_text, label_dict, patient_id = sample
                    instruction, target_text = random.choice(
                        build_instructions(report_text, label_dict or {})
                    )
                    gt_slice_indices.append(None)

                enc = tokenizer(
                    target_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                    padding=False,
                )
                target_ids = enc["input_ids"][0]  # (L,)
                items.append(
                    {
                        "volumes": vol,
                        "instructions": instruction,
                        "report_tokens": target_ids,
                        "label_dict": None,
                        "patient_id": patient_id,
                    }
                )
            collated = variable_depth_collate_fn(items)
            # Pass GT slice indices through as a per-sample list (one entry per
            # batch item, ``None`` where RadGenome grounding is unavailable).
            # Consumed by the auxiliary attention-alignment loss (A4).
            collated["gt_slice_indices"] = gt_slice_indices
            return collated

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

                # ── Forward ──────────────────────────────────────────────────
                with torch.amp.autocast("cuda", enabled=mixed_precision):
                    output = self.model(volumes, instructions, report_tokens)
                    lm_loss = output.loss

                    # Optional attention alignment auxiliary loss (A4).  Fires
                    # only when enabled and at least one sample in the batch
                    # carries RadGenome GT slice indices; otherwise the combined
                    # loss returns lm_loss unchanged.
                    if use_attn_loss:
                        gt_slice_indices = self._prepare_gt_slice_indices(
                            batch.get("gt_slice_indices")
                        )
                        try:
                            attn = self.model.projector.get_slice_attention()
                        except (AttributeError, RuntimeError):
                            attn = None
                        lm_loss = self._combined_loss(
                            lm_loss,
                            attn_weights=attn,
                            gt_slice_indices=gt_slice_indices,
                            use_attn_loss=gt_slice_indices is not None,
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

            # ── End-of-epoch: guaranteed validation + checkpoint ──────────────
            # Runs regardless of val_every_n_steps/save_every_n_steps, so a
            # completed epoch is never lost to an unlucky step-count alignment.
            val_metrics = self.validate()
            val_loss = val_metrics["val_loss"]
            history["val_loss"].append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._no_improve_rounds = 0
                self.save_checkpoint(filename="checkpoint_best.pt")
            else:
                self._no_improve_rounds += 1

            # Mark this epoch complete before saving, so a fresh Trainer that
            # loads this checkpoint and calls train() resumes at epoch+1
            # instead of repeating the epoch just finished.
            self.current_epoch = epoch + 1
            self.save_checkpoint(filename=f"checkpoint_epoch_{epoch}.pt")
            self.save_checkpoint(filename="checkpoint_latest.pt")
            log.info("Completed epoch %d (val_loss=%.4f)", epoch, val_loss)

        return history

    # ── Auxiliary-loss helpers ────────────────────────────────────────────────

    @staticmethod
    def _prepare_gt_slice_indices(
        raw: Optional[List[Optional[List[int]]]],
    ) -> Optional[List[List[int]]]:
        """Normalise per-sample GT slice indices for the attention loss.

        ``AttentionAlignmentLoss`` expects a length-B list of index lists, with
        an empty list for samples that have no grounding (those are skipped
        internally).  Returns ``None`` when the whole batch lacks grounding, so
        the caller can disable the auxiliary loss entirely for that step.
        """
        if not raw or all(idx is None for idx in raw):
            return None
        return [list(idx) if idx else [] for idx in raw]

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

                output = self.model(volumes, instructions, report_tokens)
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

        # Save LoRA adapter weights only (base LLM weights are unchanged and
        # reloaded from the pretrained checkpoint), keeping the file small.
        if hasattr(self.model.llm, "peft_config"):
            ckpt["llm_lora"] = {
                k: v
                for k, v in self.model.llm.state_dict().items()
                if "lora_" in k
            }

        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"
        path = ckpt_dir / filename
        torch.save(ckpt, path)
        log.info("Saved checkpoint: %s", path)

        if self.on_checkpoint_saved is not None:
            try:
                self.on_checkpoint_saved(str(path))
            except Exception:
                log.exception(
                    "on_checkpoint_saved callback failed for %s — training "
                    "continues, but this checkpoint may not be mirrored "
                    "externally.", path,
                )
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

        # Restore LoRA adapters (partial state dict → strict=False leaves the
        # frozen base LLM weights untouched).
        if "llm_lora" in ckpt and hasattr(self.model.llm, "peft_config"):
            self.model.llm.load_state_dict(ckpt["llm_lora"], strict=False)

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["step"]
        self.current_epoch = ckpt["epoch"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        log.info("Loaded checkpoint from %s (step=%d)", path, self.global_step)
        return self.global_step
