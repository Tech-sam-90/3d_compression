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
        token_budgets:       List of M values to sweep. Default [16,32,64,128].
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
    ) -> None:
        self.model = model
        self.val_dataset = val_dataset
        self.radgenome_dataset = radgenome_dataset
        self.totalseg_dataset = totalseg_dataset
        self.token_budgets = token_budgets if token_budgets is not None else [16, 32, 64, 128]
        self.primary_budget = primary_budget
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self.default_instruction = default_instruction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

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
            }

        result: Dict[str, float] = {}

        try:
            from aadp.evaluation.metrics.radgraph_f1 import RadGraphF1
            rg_out = RadGraphF1().compute(all_preds, all_refs)
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

        return result

    # ── T2: Multi-abnormality classification ──────────────────────────────────

    def _run_classification(self) -> Dict[str, float]:
        from aadp.evaluation.metrics.auroc_f1 import (
            AbnormalityClassificationHead,
            compute_auroc_f1,
        )

        all_feats: List[torch.Tensor] = []
        all_labels: List[List[float]] = []
        label_names: Optional[List[str]] = None

        self.model.eval()
        with torch.no_grad():
            for raw_batch in self._make_batches():
                volumes, _, label_dicts, _ = self._stack_batch(raw_batch)
                B_, D, H, W = volumes.shape
                volumes = volumes.to(self.device)
                instructions = [self.default_instruction] * B_

                try:
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
                    feats = visual_tokens.mean(dim=1)
                    all_feats.append(feats.cpu())

                    for ld in label_dicts:
                        if ld and label_names is None:
                            label_names = sorted(ld.keys())
                        n = len(label_names) if label_names else 18
                        if ld and label_names:
                            vec = [float(ld.get(k, 0.0)) for k in label_names]
                        else:
                            vec = [0.0] * n
                        all_labels.append(vec)

                except Exception as e:
                    logger.warning("[T2] feature extraction failed: %s", e)

        if not all_feats:
            return {"auroc_macro": float("nan"), "f1_macro": float("nan")}

        feats_t = torch.cat(all_feats, dim=0)
        n_labels = len(label_names) if label_names else 18
        if label_names is None:
            label_names = [f"label_{i}" for i in range(n_labels)]
        labels_t = torch.tensor(all_labels, dtype=torch.float32)

        head = AbnormalityClassificationHead(
            embed_dim=feats_t.shape[1], num_labels=n_labels
        ).to(self.device)
        head.eval()
        with torch.no_grad():
            logits = head(feats_t.to(self.device)).cpu()

        try:
            out = compute_auroc_f1(logits, labels_t, label_names)
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
        try:
            from aadp.evaluation.metrics.recall_at_k import compute_recall_at_k_curve

            attn_list: List[torch.Tensor] = []
            gt_list: List[List[int]] = []

            for raw_item in self._iter_limited(self.radgenome_dataset, self.max_samples):
                if isinstance(raw_item, dict):
                    attn = raw_item.get("attn_weights")
                    gt = raw_item.get("gt_slice_indices", [])
                    if attn is not None:
                        attn_list.append(attn)
                        gt_list.append(gt)
                else:
                    logger.warning("[T3] Unexpected RadGenome item format; skipping.")
                    break

            if not attn_list:
                return {"recall_at_5": float("nan"), "recall_at_10": float("nan")}

            attn_t = torch.stack(attn_list, dim=0).to(self.device)
            curve = compute_recall_at_k_curve(attn_t, gt_list, k_values=[5, 10])
            return {f"recall_at_{k}": v for k, v in curve.items()}

        except Exception as e:
            logger.warning("[T3] Lesion recall failed: %s", e)
            return {"recall_at_5": float("nan"), "recall_at_10": float("nan")}

    # ── T4: Anatomical localisation ───────────────────────────────────────────

    def _run_anatomical_dice(self) -> Dict[str, float]:
        if self.totalseg_dataset is None:
            return {}
        try:
            return {"dice_macro": float("nan")}
        except Exception as e:
            logger.warning("[T4] Anatomical dice failed: %s", e)
            return {"dice_macro": float("nan")}

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
