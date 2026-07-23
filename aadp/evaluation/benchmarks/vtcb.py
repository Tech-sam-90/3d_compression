"""VTCB — Visual Token Compression Benchmark.

Sweeps over a set of token budgets M, rebuilds Stage 2 at each M, and
evaluates four task families:

    T1 — Report Generation          (RadGraph-F1, RaTEScore)
    T2 — Multi-Abnormality Clf.     (AUROC macro, F1 macro)
    T3 — Slice-Specific Lesion Recall (Recall@K) — requires radgenome_dataset
    T4 — Anatomical Localisation    (Dice overlap) — requires totalseg_dataset

Passing ``None`` for either optional dataset skips that task family.
"""

import json
import logging
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VTCBRunner:
    """Run the VTCB benchmark for one or more token-budget levels.

    Args:
        model:               MedVLM to evaluate.
        val_dataset:         Iterable yielding ``(vol, report_text,
                             label_dict, patient_id)`` 4-tuples.
        radgenome_dataset:   Dataset for T3 (Recall@K).  ``None`` skips T3.
        totalseg_dataset:    Dataset for T4 (Dice).      ``None`` skips T4.
        token_budgets:       List of M values to sweep.
                             Default [16,32,64,128,256,512].
        primary_budget:      Budget highlighted in ``compare()``. Default 64.
        batch_size:          Samples per inference batch. Default 4.
        max_samples:         Cap on ``val_dataset`` items (``None`` = all).
        max_new_tokens:      Max tokens generated per sample in T1. Default 128.
        default_instruction: Instruction string used when none is in the dataset.
        device:              Target device. Default ``"cuda"`` if available.
        results_dir:         Directory where result JSONs are saved.
    """

    DEFAULT_INSTRUCTION = "Describe the findings in this CT scan."

    def __init__(
        self,
        model,
        val_dataset,
        radgenome_dataset=None,
        totalseg_dataset=None,
        token_budgets: Optional[List[int]] = None,
        primary_budget: int = 64,
        batch_size: int = 4,
        max_samples: Optional[int] = None,
        max_new_tokens: int = 128,
        default_instruction: str = DEFAULT_INSTRUCTION,
        device: Optional[str] = None,
        results_dir: str = "results/",
        probe_train_dataset=None,
        probe_max_steps: int = 1000,
    ) -> None:
        self.model = model
        self.val_dataset = val_dataset
        self.radgenome_dataset = radgenome_dataset
        self.totalseg_dataset = totalseg_dataset
        self.token_budgets = token_budgets if token_budgets is not None else [16, 32, 64, 128, 256, 512]
        self.primary_budget = primary_budget
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self.default_instruction = default_instruction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # T2 classification probe: trained on a CT-RATE training split and cached
        # per token budget M (see _run_classification).
        self.probe_train_dataset = probe_train_dataset
        self.probe_max_steps = probe_max_steps
        self._probe_cache: Dict[int, Any] = {}

    # Localisation instruction used to elicit slice-focused attention for T3/T4.
    LOCALISATION_INSTRUCTION = "Describe the location of the finding in this CT scan."

    # ── Dataset helpers ───────────────────────────────────────────────────────

    def _iter_limited(self, dataset: Iterable, max_samples: Optional[int] = None):
        count = 0
        for item in dataset:
            if max_samples is not None and count >= max_samples:
                break
            yield item
            count += 1

    def _make_batches(self):
        batch: list = []
        for item in self._iter_limited(self.val_dataset, self.max_samples):
            batch.append(item)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _unpack_item(self, item):
        if isinstance(item, (list, tuple)):
            vol, report, label_dict, pid = item[0], item[1], item[2], item[3]
        else:
            # Support both CTRATEDataset-style ("vol") and collate-fn-style ("volumes")
            vol = item.get("vol") if item.get("vol") is not None else item.get("volumes")
            report = item.get("report") if item.get("report") is not None else item.get("instructions", "")
            label_dict = item.get("label_dict") if item.get("label_dict") is not None else item.get("label_dicts", {})
            pid = item.get("patient_id") if item.get("patient_id") is not None else item.get("patient_ids", "")
        return vol, report, label_dict, pid

    def _stack_batch(self, raw_batch):
        vols, reports, label_dicts, pids = [], [], [], []
        for item in raw_batch:
            vol, report, ld, pid = self._unpack_item(item)
            vols.append(vol)
            reports.append(report)
            label_dicts.append(ld)
            pids.append(pid)

        max_D = max(v.shape[0] for v in vols)
        padded = []
        for v in vols:
            D = v.shape[0]
            if D < max_D:
                pad = torch.zeros(max_D - D, *v.shape[1:], dtype=v.dtype)
                v = torch.cat([v, pad], dim=0)
            padded.append(v)

        return torch.stack(padded, dim=0), reports, label_dicts, pids

    # ── T1: Report generation ─────────────────────────────────────────────────

    def _run_report_generation(self) -> Dict[str, float]:
        all_preds: List[str] = []
        all_refs: List[str] = []

        self.model.eval()
        with torch.no_grad():
            for raw_batch in self._make_batches():
                volumes, reports, _, _ = self._stack_batch(raw_batch)
                B = volumes.shape[0]
                volumes = volumes.to(self.device)
                instructions = [self.default_instruction] * B
                try:
                    generated_ids = self.model(
                        volumes, instructions,
                        report_tokens=None,
                        max_new_tokens=self.max_new_tokens,
                    )
                    preds = self.model._llm_tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                except Exception as e:
                    logger.warning("[T1] inference failed for batch: %s", e)
                    preds = [""] * B
                all_preds.extend(preds)
                all_refs.extend(reports)

        if not all_preds:
            return {
                "radgraph_f1": float("nan"),
                "radgraph_precision": float("nan"),
                "radgraph_recall": float("nan"),
                "ratescore_mean": float("nan"),
                "ratescore_std": float("nan"),
                "cider": float("nan"),
                "green": float("nan"),
                "avg_nlp": float("nan"),
            }

        result: Dict[str, float] = {}

        try:
            from aadp.evaluation.metrics.radgraph_f1 import RadGraphF1
            rg_out = RadGraphF1(use_xl=True).compute(all_preds, all_refs)
            result["radgraph_f1"] = rg_out["f1"]
            result["radgraph_precision"] = rg_out["precision"]
            result["radgraph_recall"] = rg_out["recall"]
        except Exception as e:
            logger.warning("[T1] RadGraph F1 failed: %s", e)
            result["radgraph_f1"] = float("nan")
            result["radgraph_precision"] = float("nan")
            result["radgraph_recall"] = float("nan")

        try:
            from aadp.evaluation.metrics.ratescore import RaTEScore
            rs_out = RaTEScore().compute(all_preds, all_refs)
            result["ratescore_mean"] = rs_out["ratescore_mean"]
            result["ratescore_std"] = rs_out["ratescore_std"]
        except Exception as e:
            logger.warning("[T1] RaTEScore failed: %s", e)
            result["ratescore_mean"] = float("nan")
            result["ratescore_std"] = float("nan")

        try:
            from aadp.evaluation.metrics.cider import compute_cider
            result["cider"] = compute_cider(all_preds, all_refs)["cider"]
        except Exception as e:
            logger.warning("[T1] CIDEr failed: %s", e)
            result["cider"] = float("nan")

        try:
            from aadp.evaluation.metrics.green import compute_green
            green_val = compute_green(all_preds, all_refs)
            result["green"] = green_val if green_val is not None else float("nan")
        except Exception as e:
            logger.warning("[T1] GREEN failed: %s", e)
            result["green"] = float("nan")

        # Avg. NLP = mean(BLEU-4, ROUGE-L, METEOR, CIDEr) — Argus Table 2
        # BLEU-4, ROUGE-L, METEOR are computed in the sweep script; here we
        # include CIDEr in the composite only when all four are present.
        # If callers also populate bleu_4/rouge_l/meteor in result, avg_nlp will
        # be correct; otherwise it is set to NaN so it does not mislead.
        nlp_keys = ("bleu_4", "rouge_l", "meteor", "cider")
        nlp_parts = [result[k] for k in nlp_keys if k in result and not math.isnan(result[k])]
        result["avg_nlp"] = float(sum(nlp_parts) / len(nlp_parts)) if len(nlp_parts) == 4 else float("nan")

        return result

    # ── Shared model forward ──────────────────────────────────────────────────

    def _forward_features(self, volumes, instructions):
        """Run ViT → projector and return ``(visual_tokens, slice_attention)``.

        Args:
            volumes:      ``(B, D, H, W)`` CT volumes (moved to device here).
            instructions: One instruction string per batch item.

        Returns:
            ``visual_tokens`` ``(B, M, C)`` and per-slice attention ``(B, D)``
            (or ``None`` if the projector does not expose slice attention, e.g.
            the Perceiver / MedPruner baselines).
        """
        B_, D, H, W = volumes.shape
        volumes = volumes.to(self.device)

        slices = volumes.reshape(B_ * D, 1, H, W)
        patch_tokens = self.model.vit(slices).float()
        N = patch_tokens.shape[1]
        C_vit = patch_tokens.shape[2]

        if self.model.vit.resize_to is not None:
            H_p = self.model.vit.resize_to // self.model.vit.patch_size
            W_p = self.model.vit.resize_to // self.model.vit.patch_size
        else:
            H_p = H // self.model.vit.patch_size
            W_p = W // self.model.vit.patch_size

        patch_tokens = patch_tokens.reshape(B_, D, N, C_vit)
        etext = self.model.instruction_encoder(instructions).float()
        visual_tokens = self.model.projector(patch_tokens, etext, H_p, W_p)

        attn = None
        getter = getattr(self.model.projector, "get_slice_attention", None)
        if getter is not None:
            try:
                attn = getter()
            except Exception:
                attn = None
        return visual_tokens, attn

    def _make_batches_from(self, dataset, max_samples: Optional[int] = None):
        """Yield stacked batches from an arbitrary dataset (used for the probe)."""
        batch: list = []
        for item in self._iter_limited(dataset, max_samples):
            batch.append(item)
            if len(batch) >= self.batch_size:
                yield self._stack_batch(batch)
                batch = []
        if batch:
            yield self._stack_batch(batch)

    def _current_budget(self) -> int:
        """Token budget M the model is currently rebuilt at."""
        return int(
            getattr(self.model, "_num_tokens", None)
            or self.model.projector.num_tokens
        )

    # ── T2: Multi-abnormality classification (linear probe) ────────────────────

    def _run_classification(self) -> Dict[str, float]:
        from aadp.data.ctrate_dataset import LABEL_COLUMNS
        from aadp.evaluation.metrics.auroc_f1 import compute_auroc_f1
        from aadp.evaluation.probes.classification_probe import (
            train_classification_probe,
        )

        label_names = list(LABEL_COLUMNS)
        M = self._current_budget()

        def feature_fn(vols, instrs):
            visual, _ = self._forward_features(vols, instrs)
            return visual.mean(dim=1)

        # Train (or reuse a cached) probe for this budget M.  A random head would
        # score ~0.5 AUROC, so a trained linear probe is required for valid T2.
        probe = self._probe_cache.get(M)
        if probe is None:
            if self.probe_train_dataset is None:
                logger.warning(
                    "[T2] No probe_train_dataset provided — cannot train the "
                    "classification probe; returning NaN."
                )
                return {"auroc_macro": float("nan"), "f1_macro": float("nan")}

            self.model.eval()
            train_batches = list(
                self._make_batches_from(self.probe_train_dataset, self.max_samples)
            )
            if not train_batches:
                return {"auroc_macro": float("nan"), "f1_macro": float("nan")}

            with torch.no_grad():
                sample_vol = train_batches[0][0]
                feats0 = feature_fn(
                    sample_vol, [self.default_instruction] * sample_vol.shape[0]
                )
                embed_dim = feats0.shape[1]

            probe = train_classification_probe(
                feature_fn=feature_fn,
                train_batches=train_batches,
                label_names=label_names,
                instruction=self.default_instruction,
                device=torch.device(self.device),
                embed_dim=embed_dim,
                max_steps=self.probe_max_steps,
            )
            self._probe_cache[M] = probe

        # Evaluate the trained probe on the val split.
        all_probs: List[torch.Tensor] = []
        all_labels: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for raw_batch in self._make_batches():
                volumes, _, label_dicts, _ = self._stack_batch(raw_batch)
                instructions = [self.default_instruction] * volumes.shape[0]
                feats = feature_fn(volumes, instructions)
                probs = torch.sigmoid(probe(feats)).cpu()
                all_probs.append(probs)
                for ld in label_dicts:
                    ld = ld or {}
                    all_labels.append([float(ld.get(k, 0.0)) for k in label_names])

        if not all_probs:
            return {"auroc_macro": float("nan"), "f1_macro": float("nan")}

        preds = torch.cat(all_probs, dim=0)
        labels_t = torch.tensor(all_labels, dtype=torch.float32)
        try:
            out = compute_auroc_f1(preds, labels_t, label_names)
            return {
                "auroc_macro": out.get("auroc_macro", float("nan")),
                "f1_macro": out.get("f1_macro", float("nan")),
            }
        except Exception as e:
            logger.warning("[T2] AUROC/F1 computation failed: %s", e)
            return {"auroc_macro": float("nan"), "f1_macro": float("nan")}

    # ── T3: Slice-specific lesion recall ──────────────────────────────────────

    def _run_lesion_recall(self) -> Dict[str, float]:
        if self.radgenome_dataset is None:
            return {}
        from aadp.evaluation.metrics.recall_at_k import compute_recall_at_k_curve

        nan_result = {
            "recall@1": float("nan"),
            "recall@3": float("nan"),
            "recall@5": float("nan"),
        }

        attn_rows: List[torch.Tensor] = []
        gt_rows: List[List[int]] = []

        self.model.eval()
        with torch.no_grad():
            for raw_batch in self._make_batches():
                volumes, _, _, pids = self._stack_batch(raw_batch)
                for b, pid in enumerate(pids):
                    findings = self.radgenome_dataset.get_by_patient(pid)
                    if not findings:
                        continue
                    gt = sorted({i for f in findings for i in f.gt_slice_indices})
                    if not gt:
                        continue

                    _, attn = self._forward_features(
                        volumes[b : b + 1], [self.LOCALISATION_INSTRUCTION]
                    )
                    if attn is None:
                        raise RuntimeError(
                            "[T3] projector exposes no slice attention "
                            "(get_slice_attention returned None); recall@k "
                            "cannot be computed for this model."
                        )
                    attn_rows.append(attn[0].detach().cpu())
                    gt_rows.append(gt)

        if not attn_rows:
            return nan_result

        # Pad per-slice attention rows to a common depth so they stack.
        max_D = max(a.shape[0] for a in attn_rows)
        padded = torch.stack(
            [
                torch.cat([a, a.new_zeros(max_D - a.shape[0])])
                if a.shape[0] < max_D
                else a
                for a in attn_rows
            ],
            dim=0,
        )
        curve = compute_recall_at_k_curve(padded, gt_rows, k_values=[1, 3, 5])
        return {
            "recall@1": curve["recall_at_1"],
            "recall@3": curve["recall_at_3"],
            "recall@5": curve["recall_at_5"],
        }

    # ── T4: Anatomical localisation ───────────────────────────────────────────

    def _run_anatomical_dice(self) -> Dict[str, Any]:
        if self.totalseg_dataset is None:
            return {}
        from aadp.evaluation.metrics.dice_overlap import compute_dice_overlap

        device = torch.device(self.device)
        dice_all: List[float] = []
        per_structure: Dict[str, List[float]] = {}
        n_evaluated = 0

        self.model.eval()
        with torch.no_grad():
            for raw_batch in self._make_batches():
                volumes, _, _, pids = self._stack_batch(raw_batch)
                for b, pid in enumerate(pids):
                    structures = self.totalseg_dataset.get_structures_for_patient(pid)
                    if not structures:
                        continue

                    _, attn = self._forward_features(
                        volumes[b : b + 1], [self.LOCALISATION_INSTRUCTION]
                    )
                    if attn is None:
                        raise RuntimeError(
                            "[T4] projector exposes no slice attention "
                            "(get_slice_attention returned None); Dice cannot "
                            "be computed for this model."
                        )
                    D = attn.shape[1]

                    for structure in structures:
                        mask = self.totalseg_dataset.load_mask(pid, structure).to(device)
                        mask = _match_depth(mask, D)                 # (D, H, W) bool
                        res = compute_dice_overlap(attn, mask.unsqueeze(0))
                        d = res["dice_mean"]
                        per_structure.setdefault(structure, []).append(d)
                        dice_all.append(d)
                        n_evaluated += 1

        if n_evaluated == 0:
            raise RuntimeError(
                "[T4] No TotalSegmentator masks were available for any evaluated "
                "volume — check the masks_root path and patient-id join."
            )

        valid = [d for d in dice_all if not math.isnan(d)]
        dice_macro = sum(valid) / len(valid) if valid else float("nan")
        per_structure_mean = {
            s: (
                sum(v for v in vals if not math.isnan(v))
                / max(len([v for v in vals if not math.isnan(v)]), 1)
            )
            for s, vals in per_structure.items()
        }
        return {"dice_macro": dice_macro, "dice_per_structure": per_structure_mean}

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, model_name: str = "model") -> Dict[int, Dict[str, Any]]:
        """Run the full VTCB sweep and write a result JSON to ``results_dir``.

        Args:
            model_name: Identifier used in the output JSON filename.

        Returns:
            ``{M: {"report_generation": {...}, ...}}`` for each budget M.
        """
        all_results: Dict[int, Dict[str, Any]] = {}

        for M in self.token_budgets:
            logger.info("VTCB [%s]: evaluating at M=%d", model_name, M)
            t0 = time.time()

            self.model.rebuild_projector_at_budget(M)

            task_results: Dict[str, Any] = {}

            try:
                task_results["report_generation"] = self._run_report_generation()
            except Exception as e:
                logger.warning("[T1] Uncaught error at M=%d: %s", M, e)
                task_results["report_generation"] = {}

            try:
                task_results["abnormality_classification"] = self._run_classification()
            except Exception as e:
                logger.warning("[T2] Uncaught error at M=%d: %s", M, e)
                task_results["abnormality_classification"] = {}

            if self.radgenome_dataset is not None:
                try:
                    task_results["lesion_recall"] = self._run_lesion_recall()
                except Exception as e:
                    logger.warning("[T3] Uncaught error at M=%d: %s", M, e)
                    task_results["lesion_recall"] = {}

            if self.totalseg_dataset is not None:
                try:
                    task_results["anatomical_localisation"] = self._run_anatomical_dice()
                except Exception as e:
                    logger.warning("[T4] Uncaught error at M=%d: %s", M, e)
                    task_results["anatomical_localisation"] = {}

            task_results["_elapsed_seconds"] = time.time() - t0
            all_results[M] = task_results
            logger.info(
                "VTCB [%s]: M=%d done in %.1fs",
                model_name, M, task_results["_elapsed_seconds"],
            )

        # Persist JSON
        json_path = os.path.join(self.results_dir, f"{model_name}_vtcb.json")
        payload = {
            "model_name": model_name,
            "primary_budget": self.primary_budget,
            "token_budgets": self.token_budgets,
            "results": {str(M): v for M, v in all_results.items()},
        }
        with open(json_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=_json_safe)
        logger.info("VTCB results saved → %s", json_path)

        return all_results

    # ── Comparison and plotting ───────────────────────────────────────────────

    @classmethod
    def compare(
        cls,
        result_jsons: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        """Compare models at their ``primary_budget`` across all metrics.

        Args:
            result_jsons: ``{model_name: path_to_json}``.

        Returns:
            ``{metric_name: {model_name: value_at_primary_budget}}``.
        """
        comparison: Dict[str, Dict[str, float]] = {}

        for model_name, json_path in result_jsons.items():
            with open(json_path) as fh:
                data = json.load(fh)

            primary = str(data.get("primary_budget", 64))
            results_at_primary = data.get("results", {}).get(primary, {})

            for metric, value in cls._flatten_results(results_at_primary).items():
                if metric not in comparison:
                    comparison[metric] = {}
                comparison[metric][model_name] = value

        return comparison

    @classmethod
    def plot_compression_curves(
        cls,
        result_jsons: Dict[str, str],
        metric_names: List[str],
        save_dir: str,
    ) -> None:
        """Plot metric-vs-M curves for multiple models; save one PNG per metric.

        Args:
            result_jsons: ``{model_name: path_to_json}``.
            metric_names: Metrics to plot (e.g. ``["radgraph_f1"]``).
            save_dir:     Output directory for PNG files.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)

        loaded: Dict[str, Any] = {}
        for model_name, json_path in result_jsons.items():
            with open(json_path) as fh:
                loaded[model_name] = json.load(fh)

        _colors = [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        ]

        for metric_name in metric_names:
            fig, ax = plt.subplots(figsize=(7, 4))

            for i, (model_name, data) in enumerate(loaded.items()):
                results = data.get("results", {})
                budgets = sorted(
                    int(k) for k in results.keys() if not k.startswith("_")
                )
                values = [
                    cls._flatten_results(results.get(str(M), {})).get(
                        metric_name, float("nan")
                    )
                    for M in budgets
                ]
                ax.plot(
                    budgets, values,
                    marker="o",
                    color=_colors[i % len(_colors)],
                    label=model_name,
                    linewidth=1.5,
                )

            title = metric_name.replace("_", " ").title()
            ax.set_xlabel("Token budget M")
            ax.set_ylabel(title)
            ax.set_title(f"{title} vs Token Budget")
            ax.legend(loc="best")
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()

            safe = metric_name.replace("/", "_")
            fig.savefig(os.path.join(save_dir, f"{safe}.png"), dpi=100)
            plt.close(fig)

    @staticmethod
    def report_parameter_counts(
        projectors: Dict[str, nn.Module],
    ) -> Dict[str, Dict[str, int]]:
        """Return trainable parameter counts per projector.

        Args:
            projectors: ``{name: nn.Module}``.

        Returns:
            ``{name: {"stage1": int, "stage2": int, "total": int}}``.
        """
        result: Dict[str, Dict[str, int]] = {}
        for name, proj in projectors.items():
            if hasattr(proj, "num_parameters"):
                result[name] = proj.num_parameters()
            else:
                total = sum(p.numel() for p in proj.parameters() if p.requires_grad)
                result[name] = {"total": total}
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _flatten_results(task_results: Dict[str, Any]) -> Dict[str, float]:
        """Recursively flatten ``{task: {metric: value}}`` → ``{metric: value}``."""
        flat: Dict[str, float] = {}
        for k, v in task_results.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                for mk, mv in v.items():
                    if not mk.startswith("_") and mv is not None:
                        try:
                            flat[mk] = float(mv)
                        except (TypeError, ValueError):
                            pass
            elif isinstance(v, (int, float)):
                flat[k] = float(v)
        return flat


def _json_safe(obj: Any) -> Any:
    """JSON serialiser fallback: converts NaN/Inf floats to ``null``."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def _match_depth(mask: torch.Tensor, D: int) -> torch.Tensor:
    """Zero-pad or crop a ``(D', H, W)`` mask along depth to match ``D`` slices."""
    d = mask.shape[0]
    if d == D:
        return mask
    if d > D:
        return mask[:D]
    pad = torch.zeros(D - d, *mask.shape[1:], dtype=mask.dtype, device=mask.device)
    return torch.cat([mask, pad], dim=0)
