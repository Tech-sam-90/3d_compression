import csv
import json
from pathlib import Path

import pytest

from aadp.data.radgenome_dataset import RadGenomeDataset, RadGenomeSample

# ── Shared synthetic data ─────────────────────────────────────────────────────

_ANNOTATIONS = [
    {
        "patient_id": "train_1_a_1",
        "finding_text": "lung nodule in right lower lobe",
        "gt_slice_indices": [10, 11, 12],
    },
    {
        "patient_id": "train_1_a_1",
        "finding_text": "pleural effusion on left",
        "gt_slice_indices": [50, 51],
    },
    {
        "patient_id": "train_2_a_1",
        "finding_text": "consolidation in right upper lobe",
        "gt_slice_indices": [20],
    },
]


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    p = tmp_path / "annotations.json"
    p.write_text(json.dumps(_ANNOTATIONS), encoding="utf-8")
    return p


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "annotations.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["patient_id", "finding_text", "gt_slice_indices"]
        )
        writer.writeheader()
        for ann in _ANNOTATIONS:
            writer.writerow(
                {
                    "patient_id": ann["patient_id"],
                    "finding_text": ann["finding_text"],
                    # Stored as bare comma-separated integers (no brackets).
                    "gt_slice_indices": ",".join(str(i) for i in ann["gt_slice_indices"]),
                }
            )
    return p


# ── __len__ ───────────────────────────────────────────────────────────────────


def test_len_json(json_file: Path) -> None:
    assert len(RadGenomeDataset(json_file)) == 3


def test_len_csv(csv_file: Path) -> None:
    assert len(RadGenomeDataset(csv_file)) == 3


# ── __getitem__ ───────────────────────────────────────────────────────────────


def test_getitem_returns_radgenome_sample_json(json_file: Path) -> None:
    ds = RadGenomeDataset(json_file)
    sample = ds[0]
    assert isinstance(sample, RadGenomeSample)


def test_getitem_fields_json(json_file: Path) -> None:
    ds = RadGenomeDataset(json_file)
    sample = ds[0]
    assert sample.patient_id == "train_1_a_1"
    assert sample.finding_text == "lung nodule in right lower lobe"
    assert sample.gt_slice_indices == [10, 11, 12]


def test_getitem_fields_csv(csv_file: Path) -> None:
    ds = RadGenomeDataset(csv_file)
    sample = ds[0]
    assert sample.patient_id == "train_1_a_1"
    assert sample.finding_text == "lung nodule in right lower lobe"
    assert sample.gt_slice_indices == [10, 11, 12]


def test_getitem_last_entry(json_file: Path) -> None:
    ds = RadGenomeDataset(json_file)
    sample = ds[2]
    assert sample.patient_id == "train_2_a_1"
    assert sample.gt_slice_indices == [20]


def test_csv_bracket_wrapped_indices(tmp_path: Path) -> None:
    """CSV with bracket-wrapped gt_slice_indices should also parse correctly."""
    p = tmp_path / "bracket.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["patient_id", "finding_text", "gt_slice_indices"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "patient_id": "valid_1_a_1",
                "finding_text": "emphysema",
                "gt_slice_indices": "[5,6,7]",
            }
        )
    ds = RadGenomeDataset(p)
    assert ds[0].gt_slice_indices == [5, 6, 7]


# ── get_by_patient ────────────────────────────────────────────────────────────


def test_get_by_patient_returns_all_findings(json_file: Path) -> None:
    ds = RadGenomeDataset(json_file)
    results = ds.get_by_patient("train_1_a_1")
    assert len(results) == 2
    assert all(s.patient_id == "train_1_a_1" for s in results)


def test_get_by_patient_finding_texts(json_file: Path) -> None:
    ds = RadGenomeDataset(json_file)
    texts = {s.finding_text for s in ds.get_by_patient("train_1_a_1")}
    assert "lung nodule in right lower lobe" in texts
    assert "pleural effusion on left" in texts


def test_get_by_patient_single_finding(json_file: Path) -> None:
    ds = RadGenomeDataset(json_file)
    results = ds.get_by_patient("train_2_a_1")
    assert len(results) == 1
    assert results[0].finding_text == "consolidation in right upper lobe"


def test_get_by_patient_unknown_returns_empty(json_file: Path) -> None:
    ds = RadGenomeDataset(json_file)
    assert ds.get_by_patient("train_999_a_1") == []


# ── Error handling ────────────────────────────────────────────────────────────


def test_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        RadGenomeDataset("/nonexistent/path/annotations.json")


def test_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "annotations.txt"
    p.write_text("data")
    with pytest.raises(ValueError, match="Unsupported annotation format"):
        RadGenomeDataset(p)


def test_load_from_huggingface_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="not yet publicly available"):
        RadGenomeDataset.load_from_huggingface()


def test_load_from_huggingface_message_mentions_local_file() -> None:
    with pytest.raises(NotImplementedError, match="annotations_path"):
        RadGenomeDataset.load_from_huggingface(token="fake")
