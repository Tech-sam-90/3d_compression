"""CTRawDataset — loads raw CT-RATE NIfTI volumes for end-to-end ICTC training
with an unfrozen ViT (see docs/DATA_AUDIT.md for the raw-data availability
audit this class was built against).

NIfTI path resolution/loading comes from aadp/data/ctrate_dataset.py
unchanged. T2 instruction sampling (entity blacklist, sentence extraction)
comes from aadp/data/ctclip_feature_dataset.py's build_instruction_for_task()
— imported, not copied, so this file can never silently drift out of sync
with the feature-based pipeline's task logic. T1 (30-paraphrase sampling,
see T1_INSTRUCTIONS) and T3 (evidence grounding, positive/negative ratio)
are reimplemented locally in _build_instruction_for_task_raw() — see that
function's docstring for exactly why each deviates from the imported one.

Volume preprocessing (HU window, isotropic resampling, crop/pad) is
DELIBERATELY NOT aadp/data/preprocessing.py's window_and_normalize()/
pad_or_crop_depth() — this class feeds the real pretrained CT-CLIP ViT
(ibrahimethemhamamci/CT-CLIP's CTViT, checkpoint
models/CT-CLIP-Related/CT-CLIP_v2.pt from the ibrahimhamamci/CT-RATE HF
dataset repo), and that encoder is only valid on inputs matching its own
training preprocessing. Traced from the reference repo's own
scripts/data_inference_nii.py (ground truth, not inferred): HU clip to
[-1000, 1000] (not [-1000, 400]), resample to ANISOTROPIC spacing
z=1.5mm / xy=0.75mm (aadp/data/preprocessing.py's resample_to_isotropic()
only supports a single isotropic target, so it can't express this — the
resize below reimplements the reference script's own torch.F.interpolate
approach instead of forcing that function to fit), normalize via /1000.0
(range ~[-1, 1], not [0, 1]), then centre-crop/pad all three axes to
(240, 480, 480) with pad_value=-1.0. This shape is architecture-mandated,
not a preprocessing choice: CTViT's checkpoint (patch_size=20,
temporal_patch_size=10, spatial_depth=4, temporal_depth=4) requires
exactly image_size=(480,480) and produces (240/10) * (480/20)**2 * 512 =
(24, 576, 512) tokens, the convention already used throughout
aadp/models/projector/.

Expected on-disk layout (confirmed against the real downloaded tree in
docs/DATA_AUDIT.md):
    {ct_volume_root}/dataset/{train_fixed,valid_fixed}/{patient}/{scan}/{volume_name}.nii.gz
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from aadp.data.ctrate_dataset import LABEL_COLUMNS, _local_nifti_path, _nifti_to_array
from aadp.data.ctclip_feature_dataset import _t3_positive_target, build_instruction_for_task
from aadp.data.instruction_builder import _prettify_label

logger = logging.getLogger(__name__)

_SPLIT_TO_FOLDER = {"train": "train_fixed", "val": "valid_fixed"}

# T1 instruction paraphrases. Index 0 is exactly
# ctclip_feature_dataset.py's build_instruction_for_task()'s original
# hardcoded T1 string — kept verbatim (not just similar) so it stays the
# fixed, deterministic instruction used at eval time (val split) and stays
# directly comparable to checkpoints trained before this paraphrase list
# existed (e.g. the stage1 job already running as this was written trained
# under the single fixed string via the then-current code — editing this
# file doesn't retroactively change an already-running process's behavior,
# but does mean anything trained afterward sees these paraphrases while
# eval always asks the one canonical phrasing).
T1_INSTRUCTIONS = [
    "Generate a radiology report for this CT scan.",
    # Formal clinical tone
    "Please provide a comprehensive radiology report describing the findings in this CT scan.",
    "Kindly generate a detailed radiological interpretation of this CT examination.",
    "Provide a formal radiology report summarizing all findings for this CT scan.",
    "Please compose a thorough radiological report for this computed tomography scan.",
    "Generate a complete diagnostic radiology report for the following CT examination.",
    "Provide a detailed clinical radiology report covering all findings in this scan.",
    # Concise command form
    "Describe the findings in this chest CT.",
    "Report the findings for this CT scan.",
    "Summarize this CT scan.",
    "Describe this CT examination.",
    "List the findings in this scan.",
    "Report on this CT scan.",
    # Question form
    "What does this CT scan show?",
    "What are the findings in this CT scan?",
    "What abnormalities, if any, are present in this scan?",
    "What does the radiology report for this scan say?",
    "What is shown in this CT examination?",
    "What findings are present on this chest CT?",
    # Implicit task
    "Radiology report:",
    "Impression and findings:",
    "Findings:",
    "CT report:",
    "Radiological findings:",
    "Report:",
    # Different specificity
    "Note any abnormalities.",
    "Report on all lobes.",
    "Describe all structures visible in this CT scan in detail.",
    "Provide a complete assessment of every organ system visible in this scan.",
    "Generate a detailed radiology report for this CT scan.",
]
assert len(T1_INSTRUCTIONS) == 30

# CT-CLIP's own reference preprocessing (scripts/data_inference_nii.py in
# ibrahimethemhamamci/CT-CLIP) — see module docstring for why these values,
# not aadp/data/preprocessing.py's defaults, are correct here.
_CTCLIP_HU_MIN = -1000.0
_CTCLIP_HU_MAX = 1000.0
_CTCLIP_TARGET_SPACING = (1.5, 0.75, 0.75)  # (z, y, x) mm, matching (D, H, W)
_CTCLIP_PAD_VALUE = -1.0


def _resize_to_spacing(
    volume: np.ndarray, current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Trilinear-resize a (D, H, W) volume so its voxel spacing matches
    target_spacing. Mirrors CT-CLIP's own resize_array() in
    scripts/data_inference_nii.py exactly, including its use of int()
    truncation (not round()) for the output size."""
    scaling = [current_spacing[i] / target_spacing[i] for i in range(3)]
    new_shape = [max(1, int(volume.shape[i] * scaling[i])) for i in range(3)]
    t = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(t, size=new_shape, mode="trilinear", align_corners=False)
    return resized[0, 0].numpy()


def _crop_or_pad_3d(volume: np.ndarray, target_shape: Tuple[int, int, int], pad_value: float) -> np.ndarray:
    """Centre-crop/pad all three axes to target_shape (CT-CLIP's reference
    script does this in (H, W, D) order for readability then permutes back;
    cropping/padding is per-axis-independent, so doing it directly in
    (D, H, W) order here is mathematically identical without the redundant
    transpose round-trip)."""
    out = volume
    for axis in range(3):
        size, target = out.shape[axis], target_shape[axis]
        if size > target:
            start = (size - target) // 2
            slicer = [slice(None)] * 3
            slicer[axis] = slice(start, start + target)
            out = out[tuple(slicer)]
    pad_width = []
    for axis in range(3):
        size, target = out.shape[axis], target_shape[axis]
        before = max((target - size) // 2, 0)
        after = max(target - size - before, 0)
        pad_width.append((before, after))
    return np.pad(out, pad_width, mode="constant", constant_values=pad_value)


def _build_instruction_for_task_raw(
    task: str, report: str, label_dict: Dict[str, int], positive_oversample_ratio: float,
    split: str = "train",
) -> Tuple[str, str]:
    """Same as ctclip_feature_dataset.build_instruction_for_task(), except:
      - T1 samples from T1_INSTRUCTIONS (random on "train", always index 0 —
        the canonical original string — on "val") instead of the single
        hardcoded string.
      - T3's positive/negative split ratio is a parameter instead of a
        hardcoded 0.7.

    T2 is delegated to the imported function completely unchanged. T1/T3 are
    reimplemented here — not imported — T1 for the paraphrase sampling above,
    T3 for the same reason as before (the original hardcodes its 0.7/0.3
    split inline; this mirrors that logic exactly — same _t3_positive_target
    evidence grounding, same _prettify_label, same T1-fallback-when-no-labels
    behavior, now itself also using the paraphrase list) with only the
    comparison threshold parameterized. Defaults (positive_oversample_ratio=
    0.7, split="train") match ctclip_feature_dataset.py's original behavior.
    """
    report = (report or "").strip()

    if task == "T1":
        instruction = T1_INSTRUCTIONS[0] if split == "val" else random.choice(T1_INSTRUCTIONS)
        return (instruction, report)

    if task != "T3":
        return build_instruction_for_task(task, report, label_dict)
    positives = [k for k, v in label_dict.items() if v == 1]
    negatives = [k for k, v in label_dict.items() if v == 0]
    if positives and negatives:
        if random.random() < positive_oversample_ratio:
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
    return ("Generate a radiology report for this CT scan.", report)


class CTRawDataset(Dataset):
    """Raw-NIfTI CT-RATE dataset for end-to-end (unfrozen ViT) ICTC training.

    Args:
        merged_csv:                Path to train_merged.csv or valid_merged.csv
                                    (VolumeName, Findings_EN, Impressions_EN,
                                    + 18 binary label columns).
        ct_volume_root:             Root of the downloaded NIfTI tree, e.g.
                                    /scratch/sadeniji/ct_rate_raw.
        split:                     "train" or "val".
        target_shape:               (D, H, W) after resize. Default (240, 480,
                                    480) is CT-CLIP's own CTViT input
                                    requirement (architecture-mandated — see
                                    module docstring), not an arbitrary choice.
        task_weights:               Sampling weights over T1/T2/T3. Defaults
                                    to {"T1": 0.6, "T2": 0.3, "T3": 0.1},
                                    matching ctclip_feature_dataset.py exactly.
        positive_oversample_ratio: T3 positive/negative split. Default 0.7
                                    matches ctclip_feature_dataset.py's
                                    hardcoded ratio exactly (see
                                    _build_instruction_for_task_raw's
                                    docstring for why this one parameter is
                                    reimplemented rather than imported).

    Deliberately returns raw ``instruction``/``target`` strings, not
    pre-tokenized ids — mirrors CTCLIPFeatureDataset/ctclip_collate_fn
    (aadp/data/ctclip_feature_dataset.py) exactly, because that's the
    contract CTCLIPStage2VLM.forward() actually expects: it takes
    ``instructions: List[str]`` and tokenizes them itself internally, and
    expects ``report_tokens`` to be produced by the caller (e.g.
    ``model.tokenizer(batch["target"], padding=True, ...)``, exactly as
    scripts/train_ctclip.py's training loop already does for the
    feature-based path). An earlier version of this class did its own
    combined-sequence instruction+target tokenization with -100 masking —
    that was modeled on the wrong reference and doesn't match
    CTCLIPStage2VLM.forward()'s real signature at all.
    """

    def __init__(
        self,
        merged_csv: str,
        ct_volume_root: str,
        split: str,
        target_shape: Tuple[int, int, int] = (240, 480, 480),
        task_weights: Optional[Dict[str, float]] = None,
        positive_oversample_ratio: float = 0.7,
    ) -> None:
        super().__init__()

        if split not in _SPLIT_TO_FOLDER:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.ct_volume_root = Path(ct_volume_root)
        self.split = split
        self.folder = _SPLIT_TO_FOLDER[split]
        self.target_shape = target_shape
        self.positive_oversample_ratio = positive_oversample_ratio

        if task_weights is None:
            task_weights = {"T1": 0.6, "T2": 0.3, "T3": 0.1}
        self.tasks = list(task_weights.keys())
        total_w = sum(task_weights[t] for t in self.tasks)
        self.task_probs = np.array([task_weights[t] / total_w for t in self.tasks], dtype=np.float64)

        # ── Load CSV ──────────────────────────────────────────────────────────
        df = pd.read_csv(merged_csv)
        logger.info("CTRawDataset(%s): loaded %s, columns=%s", split, merged_csv, list(df.columns))

        missing_label_cols = [c for c in LABEL_COLUMNS if c not in df.columns]
        if missing_label_cols:
            logger.warning("CTRawDataset(%s): missing expected label columns: %s", split, missing_label_cols)
        self.label_cols = [c for c in LABEL_COLUMNS if c in df.columns]

        # ── Path cache: one os.walk instead of a Path.exists() stat() per
        # sample on a shared 50K-file Lustre filesystem. _local_nifti_path()
        # is still what actually computes the path used in __getitem__ (it's
        # O(1) pure string construction, no I/O) — this cache exists purely
        # to filter the CSV down to rows with a real file, once, up front.
        split_root = self.ct_volume_root / "dataset" / self.folder
        logger.info("CTRawDataset(%s): building path cache via os.walk(%s) ...", split, split_root)
        found_names = set()
        for dirpath, _dirnames, filenames in os.walk(split_root):
            for fn in filenames:
                if fn.endswith(".nii.gz"):
                    found_names.add(fn)
        logger.info("CTRawDataset(%s): path cache found %d volumes under %s", split, len(found_names), split_root)

        valid_rows = []
        missing = 0
        for _, row in df.iterrows():
            volume_name = str(row["VolumeName"])
            if volume_name in found_names:
                valid_rows.append(row.to_dict())
            else:
                missing += 1
        logger.info(
            "CTRawDataset(%s): %d / %d CSV rows have a matching NIfTI file (%d dropped — missing volume).",
            split, len(valid_rows), len(df), missing,
        )
        self.samples: List[dict] = valid_rows

    def __len__(self) -> int:
        return len(self.samples)

    # ── Volume loading ───────────────────────────────────────────────────────

    def _load_volume(self, volume_name: str) -> torch.Tensor:
        path = _local_nifti_path(self.ct_volume_root, volume_name, self.folder)

        # _nifti_to_array (aadp/data/ctrate_dataset.py) returns a (D, H, W)
        # float32 numpy array of RAW HU values — confirmed from its own
        # source: nib.load(...).dataobj is NIfTI-native (X, Y, Z), then
        # .transpose(2, 0, 1) -> (Z, X, Y) = (D, H, W). CT-RATE's "_fixed"
        # (v2) volumes have RescaleSlope/Intercept already baked into the
        # header, so windowing these raw voxels directly is valid (see
        # ctrate_dataset.py's _SPLIT_TO_FOLDER comment).
        arr = _nifti_to_array(path)  # (D, H, W) float32, raw HU

        # Voxel spacing isn't in _nifti_to_array's return value (pixel data
        # only), so read it directly from the header. nibabel zooms follow
        # the pre-transpose (X, Y, Z) axis order, matching arr's (D, H, W) =
        # (Z, X, Y) axes as (z, x, y) — i.e. exactly (D, H, W) order already.
        zooms = nib.load(str(path)).header.get_zooms()[:3]
        current_spacing = (float(zooms[2]), float(zooms[0]), float(zooms[1]))  # (z, x, y) -> (D, H, W)

        arr = np.clip(arr, _CTCLIP_HU_MIN, _CTCLIP_HU_MAX).astype(np.float32)
        arr = _resize_to_spacing(arr, current_spacing, _CTCLIP_TARGET_SPACING)
        arr = arr / 1000.0  # CT-CLIP's own normalization: HU/1000, range ~[-1, 1]
        arr = _crop_or_pad_3d(arr, self.target_shape, pad_value=_CTCLIP_PAD_VALUE)

        return torch.from_numpy(arr).unsqueeze(0)  # (1, D, H, W) float32

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict:
        row = self.samples[idx]
        volume_name = str(row["VolumeName"])

        try:
            volume = self._load_volume(volume_name)
        except Exception as exc:
            # Path-cache filtering at __init__ should prevent this in the
            # normal case, but a file can still become unreadable between
            # init and iteration (e.g. a transient Lustre hiccup — this
            # project hit exactly that failure mode earlier this session).
            # Don't crash the DataLoader over one bad sample.
            logger.warning(
                "CTRawDataset(%s): failed to load %r (%s) — using next sample instead.",
                self.split, volume_name, exc,
            )
            return self.__getitem__((idx + 1) % len(self))

        findings = str(row.get("Findings_EN") or "").strip()
        impressions = str(row.get("Impressions_EN") or "").strip()
        report = (findings + " " + impressions).strip()

        label_dict: Dict[str, int] = {}
        for col in self.label_cols:
            val = row.get(col, 0)
            label_dict[col] = int(val) if not pd.isna(val) else 0

        task = str(np.random.choice(self.tasks, p=self.task_probs))
        instruction, target = _build_instruction_for_task_raw(
            task, report, label_dict, self.positive_oversample_ratio, split=self.split,
        )

        return {
            "volume": volume,             # (1, D, H, W) float32
            "instruction": instruction,   # str — CTCLIPStage2VLM.forward()'s `instructions` arg
            "target": target,             # str — caller tokenizes this into `report_tokens`
            "label_dict": label_dict,     # Dict[str, int]
            "task": task,                 # "T1" | "T2" | "T3"
            "volume_name": volume_name,   # for debugging
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Mirrors aadp/data/ctclip_feature_dataset.py's ctclip_collate_fn exactly
    (same keys, same label_dict-to-tensor stacking), with "volume" in place
    of "features" — so CTRawDataset batches are a drop-in replacement for
    CTCLIPFeatureDataset batches in scripts/train_ctclip.py's training loop.
    """
    volumes = torch.stack([item["volume"] for item in batch], dim=0)  # (B, 1, D, H, W)

    label_keys = list(batch[0]["label_dict"].keys())
    labels = torch.tensor(
        [[item["label_dict"][k] for k in label_keys] for item in batch],
        dtype=torch.float32,
    )

    return {
        "volume": volumes,
        "instruction": [item["instruction"] for item in batch],
        "target": [item["target"] for item in batch],
        "labels": labels,
        "label_keys": label_keys,
        "task": [item["task"] for item in batch],
        "volume_name": [item["volume_name"] for item in batch],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = CTRawDataset(
        merged_csv="/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/train_merged.csv",
        ct_volume_root="/scratch/sadeniji/ct_rate_raw",
        split="train",
    )

    print(f"Dataset size: {len(ds)}")
    print("Expected per-sample volume shape: (1, 240, 480, 480) — CT-CLIP's CTViT input requirement.")
    for i in [0, 100, 1000]:
        sample = ds[i]
        print(f"[{i}] volume={tuple(sample['volume'].shape)}  task={sample['task']}  "
              f"instruction={sample['instruction']!r}  "
              f"target={sample['target'][:60]!r}...  "
              f"volume_name={sample['volume_name']}")

    print("\n20-sample task diversity check:")
    seen_tasks = set()
    for i in range(20):
        sample = ds[i]
        seen_tasks.add(sample["task"])
    print(f"Tasks seen across 20 samples: {sorted(seen_tasks)}")
    missing = {"T1", "T2", "T3"} - seen_tasks
    if missing:
        print(f"WARNING: {missing} did not appear in 20 samples (may just be low-probability luck at low n).")
    else:
        print("OK: all three tasks appeared.")

    print("\nCollate function check (batch of 4):")
    batch = [ds[i] for i in range(4)]
    collated = collate_fn(batch)
    for k, v in collated.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v}")
