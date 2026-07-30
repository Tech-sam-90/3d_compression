"""CTCLIPFeatureDataset — loads pre-extracted CT-CLIP features from .pt files.

Instead of running the ViT + Stage 1 encoder at training time, this dataset
reads pre-extracted (24, 24, 24, 512) feature tensors saved as fp16 .pt files
and serves them as (24, 576, 512) float32 tensors ready for Stage 2.

Expected .pt filename convention: <VolumeName stem>.pt
e.g. "train_1_a_1.nii.gz" → "train_1_a_1.pt"
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from aadp.data.instruction_builder import (
    build_instructions,
    extract_entities_from_report,
    sentences_containing,
    _prettify_label,
)

logger = logging.getLogger(__name__)

# Columns that are never label columns. Includes the free-text report columns
# from CT-RATE's train_reports.csv / validation_reports.csv (which the merged
# CSV produced by scripts/download_ctrate_labels.py still carries alongside
# the binary label columns from multi_abnormality_labels/).
_NON_LABEL_COLS = {
    "VolumeName", "ClinicalInformation_EN", "Technique_EN",
    "Findings_EN", "Impressions_EN",
    "split", "Split", "PatientID", "StudyDate",
}

# Fixed CT-CLIP feature spatial dims
_CTCLIP_D = 24
_CTCLIP_H = 24
_CTCLIP_W = 24
_CTCLIP_C = 512
_CTCLIP_K = _CTCLIP_H * _CTCLIP_W  # 576 tokens per slice after flatten


def _t3_positive_target(report: str, abn: str) -> str:
    """Evidence-grounded T3 positive target: "Yes." plus the first sentence
    of the report that actually mentions the abnormality, capped at one
    sentence / 150 chars. Falls back to a bare "Yes." when abn's prettified
    name (e.g. "lung nodule") doesn't appear verbatim in the report text —
    sentences_containing() would otherwise silently fall back to the WHOLE
    report, which is exactly the multi-sentence, unbounded target this
    function exists to avoid."""
    if abn.lower() not in report.lower():
        return "Yes."
    matched = sentences_containing(report, abn)
    evidence = matched.split(". ", 1)[0].strip()
    if not evidence.endswith("."):
        evidence += "."
    if len(evidence) > 150:
        evidence = evidence[:150].rstrip()
        if not evidence.endswith("."):
            evidence += "."
    return f"Yes. {evidence}"


def build_instruction_for_task(
    task: str,
    report: str,
    label_dict: Dict[str, int],
) -> Tuple[str, str]:
    """Return a single (instruction, target) pair for the given task type.

    Args:
        task:       "T1", "T2", or "T3".
        report:     Full radiology report text.
        label_dict: Dict mapping abnormality name → binary label (0 or 1).

    Returns:
        Tuple of (instruction_str, target_str).
    """
    report = (report or "").strip()

    if task == "T1":
        return ("Generate a radiology report for this CT scan.", report)

    if task == "T2":
        entities = extract_entities_from_report(report)
        entity = random.choice(entities) if entities else "lung"
        return (
            f"Describe the findings related to {entity} in this CT scan.",
            sentences_containing(report, entity),
        )

    if task == "T3":
        positives = [k for k, v in label_dict.items() if v == 1]
        negatives = [k for k, v in label_dict.items() if v == 0]
        # Prefer a positive example when available, else negative, else T1 fallback.
        # 70/30 (not 50/50) positive/negative: docs/QUALITATIVE_EVAL_V2.md found
        # the model defaulted to "No." for 8/10 T3 questions regardless of the
        # true label — oversampling positives gives more gradient signal against
        # that bias. Positive targets are now evidence-grounded ("Yes. <sentence
        # from the report>") for the same reason: a bare "Yes." gives the model
        # nothing to learn to associate with the abnormality being present.
        if positives and negatives:
            if random.random() < 0.7:
                abn = _prettify_label(random.choice(positives))
                return (f"Is there evidence of {abn} in this scan? Answer yes or no.",
                        _t3_positive_target(report, abn))
            else:
                abn = _prettify_label(random.choice(negatives))
                return (f"Is there evidence of {abn} in this scan? Answer yes or no.", "No.")
        elif positives:
            abn = _prettify_label(random.choice(positives))
            return (f"Is there evidence of {abn} in this scan? Answer yes or no.",
                    _t3_positive_target(report, abn))
        elif negatives:
            abn = _prettify_label(random.choice(negatives))
            return (f"Is there evidence of {abn} in this scan? Answer yes or no.", "No.")
        # Fall back to T1 if no label info
        return ("Generate a radiology report for this CT scan.", report)

    raise ValueError(f"Unknown task: {task!r}. Expected one of 'T1', 'T2', 'T3'.")


class CTCLIPFeatureDataset(Dataset):
    """Dataset of pre-extracted CT-CLIP visual features paired with reports.

    Each sample is a (24, 576, 512) float32 tensor (D slices × K spatial tokens
    per slice × C channels), loaded from a .pt file and paired with a randomly
    sampled instruction/target pair from tasks T1–T3.

    Args:
        features_dir:  Local directory containing one .pt file per volume.
        csv_path:      CT-RATE reports CSV with columns: VolumeName,
                       Findings_EN, Impressions_EN, + 18 binary label columns.
        tasks:         Which task types to sample from.
        task_weights:  Relative sampling probabilities for each task.
        max_samples:   If set, truncate the valid sample list to this count.
    """

    def __init__(
        self,
        features_dir: str,
        csv_path: str,
        tasks: List[str] = None,
        task_weights: Dict[str, float] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()

        if tasks is None:
            tasks = ["T1", "T2", "T3"]
        if task_weights is None:
            task_weights = {"T1": 0.6, "T2": 0.3, "T3": 0.1}

        self.features_dir = Path(features_dir)
        self.tasks = tasks

        # Normalise task weights to sum to 1.0
        total_w = sum(task_weights[t] for t in tasks)
        self.task_probs = np.array([task_weights[t] / total_w for t in tasks], dtype=np.float64)

        # ── Load CSV ──────────────────────────────────────────────────────────
        df = pd.read_csv(csv_path)

        # Detect 18 binary label columns (all int-valued non-metadata columns)
        meta_cols = _NON_LABEL_COLS | set()
        candidate_label_cols = [
            c for c in df.columns
            if c not in meta_cols
            and pd.api.types.is_integer_dtype(df[c])
        ]
        # Also accept columns that are 0/1 floats (can happen with NaN coercion)
        for c in df.columns:
            if c not in meta_cols and c not in candidate_label_cols:
                try:
                    unique_vals = df[c].dropna().unique()
                    # len() guard: an all-NaN column has an empty unique set,
                    # which is (vacuously) a subset of {0, 1} and would
                    # otherwise be misdetected as a label column.
                    if len(unique_vals) > 0 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                        candidate_label_cols.append(c)
                except Exception:
                    pass

        self.label_cols: List[str] = sorted(candidate_label_cols)
        logger.info("Detected %d label columns: %s", len(self.label_cols), self.label_cols)

        # ── Filter to rows where the .pt file exists ──────────────────────────
        total_rows = len(df)
        valid_rows = []
        missing = 0
        for _, row in df.iterrows():
            stem = Path(str(row["VolumeName"])).stem  # "train_1_a_1.nii" → "train_1_a_1"
            # Handle double extension (.nii.gz): Path.stem only strips last ext
            if stem.endswith(".nii"):
                stem = stem[:-4]
            pt_path = self.features_dir / f"{stem}.pt"
            if pt_path.exists():
                valid_rows.append(row.to_dict())
            else:
                missing += 1

        logger.info(
            "CTCLIPFeatureDataset: %d / %d rows have matching .pt files "
            "(%d dropped — missing features).",
            len(valid_rows), total_rows, missing,
        )

        if max_samples is not None and max_samples < len(valid_rows):
            valid_rows = valid_rows[:max_samples]
            logger.info("Truncated to %d samples (max_samples=%d).", len(valid_rows), max_samples)

        self.samples = valid_rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        row = self.samples[idx]

        # ── Derive stem and load features ────────────────────────────────────
        stem = Path(str(row["VolumeName"])).stem
        if stem.endswith(".nii"):
            stem = stem[:-4]

        feat_path = self.features_dir / f"{stem}.pt"
        feat = torch.load(feat_path, weights_only=True)   # (24, 24, 24, 512) fp16
        feat = feat.float()                                # → float32
        feat = feat.reshape(_CTCLIP_D, _CTCLIP_K, _CTCLIP_C)  # (24, 576, 512)

        # ── Build report text ────────────────────────────────────────────────
        findings = str(row.get("Findings_EN") or "").strip()
        impressions = str(row.get("Impressions_EN") or "").strip()
        report = (findings + " " + impressions).strip()

        # ── Label dict ───────────────────────────────────────────────────────
        label_dict: Dict[str, int] = {}
        for col in self.label_cols:
            val = row.get(col, 0)
            label_dict[col] = int(val) if not pd.isna(val) else 0

        # ── Sample task and build instruction/target ─────────────────────────
        task = str(np.random.choice(self.tasks, p=self.task_probs))
        instruction, target = build_instruction_for_task(task, report, label_dict)

        return {
            "features":    feat,          # (24, 576, 512) float32
            "instruction": instruction,   # str
            "target":      target,        # str
            "patient_id":  stem,          # str
            "label_dict":  label_dict,    # Dict[str, int]
        }


def ctclip_collate_fn(batch: List[Dict]) -> Dict:
    """Collate a batch of CTCLIPFeatureDataset samples.

    Features are always fixed shape (24, 576, 512), so no padding is needed.
    Label dicts are stacked into a float32 tensor of shape (B, num_labels).

    Args:
        batch: List of dicts from CTCLIPFeatureDataset.__getitem__.

    Returns:
        Dict with keys:
            "features":     (B, 24, 576, 512) float32
            "instruction":  List[str]
            "target":       List[str]
            "patient_id":   List[str]
            "labels":       (B, num_labels) float32
            "label_keys":   List[str] — column names in tensor column order
    """
    features = torch.stack([item["features"] for item in batch], dim=0)

    # Consistent column order from first item (all items share the same keys
    # because they come from the same dataset's self.label_cols list)
    label_keys = list(batch[0]["label_dict"].keys())
    labels = torch.tensor(
        [[item["label_dict"][k] for k in label_keys] for item in batch],
        dtype=torch.float32,
    )

    return {
        "features":    features,
        "instruction": [item["instruction"] for item in batch],
        "target":      [item["target"] for item in batch],
        "patient_id":  [item["patient_id"] for item in batch],
        "labels":      labels,
        "label_keys":  label_keys,
    }
