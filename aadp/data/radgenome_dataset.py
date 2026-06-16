import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from torch.utils.data import Dataset


@dataclass(frozen=True)
class RadGenomeSample:
    """One grounded finding annotation from RadGenome-Chest CT.

    patient_id matches the CT-RATE volume naming convention
    (e.g. "train_1_a_1") so it can be joined back to CT-RATE samples
    during evaluation.
    """

    patient_id: str
    finding_text: str
    gt_slice_indices: List[int]


class RadGenomeDataset(Dataset):
    """PyTorch Dataset for RadGenome-Chest CT slice-level grounding annotations.

    Used exclusively for evaluation — never as a training signal.

    Supports loading from a local JSON or CSV annotation file.
    File format is detected automatically from the file extension.

    JSON format — a list of objects::

        [
          {
            "patient_id": "train_1_a_1",
            "finding_text": "lung nodule in right lower lobe",
            "gt_slice_indices": [10, 11, 12]
          },
          ...
        ]

    CSV format — header row + one row per finding::

        patient_id,finding_text,gt_slice_indices
        train_1_a_1,lung nodule in right lower lobe,"10,11,12"
        ...

    ``gt_slice_indices`` in CSV may be bare comma-separated integers
    (``10,11,12``) or bracket-wrapped (``[10,11,12]``); both are accepted.
    """

    def __init__(self, annotations_path: Union[str, Path]) -> None:
        path = Path(annotations_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            self._samples: List[RadGenomeSample] = _load_json(path)
        elif suffix == ".csv":
            self._samples = _load_csv(path)
        else:
            raise ValueError(
                f"Unsupported annotation format '{suffix}'. Expected .json or .csv."
            )

        # Pre-build patient_id → sample indices for O(1) get_by_patient lookup.
        self._patient_index: Dict[str, List[int]] = {}
        for i, sample in enumerate(self._samples):
            self._patient_index.setdefault(sample.patient_id, []).append(i)

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
    def load_from_huggingface(cls, token: str = None) -> "RadGenomeDataset":
        raise NotImplementedError(
            "RadGenome-Chest CT is not yet publicly available via HuggingFace. "
            "Pass annotations_path to load from a local file."
        )

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> RadGenomeSample:
        return self._samples[idx]

    # ── Evaluation helper ─────────────────────────────────────────────────────

    def get_by_patient(self, patient_id: str) -> List[RadGenomeSample]:
        """Return all findings for a given patient_id.

        Returns an empty list if the patient is not in the dataset.
        Called during evaluation to look up ground-truth slices for a
        CT-RATE volume being processed.
        """
        return [self._samples[i] for i in self._patient_index.get(patient_id, [])]


# ── File loaders ──────────────────────────────────────────────────────────────


def _load_json(path: Path) -> List[RadGenomeSample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        RadGenomeSample(
            patient_id=str(row["patient_id"]),
            finding_text=str(row["finding_text"]),
            gt_slice_indices=[int(i) for i in row["gt_slice_indices"]],
        )
        for row in data
    ]


def _load_csv(path: Path) -> List[RadGenomeSample]:
    samples: List[RadGenomeSample] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["gt_slice_indices"].strip().strip("[]")
            indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
            samples.append(
                RadGenomeSample(
                    patient_id=str(row["patient_id"]),
                    finding_text=str(row["finding_text"]),
                    gt_slice_indices=indices,
                )
            )
    return samples
