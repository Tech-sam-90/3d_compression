"""Tests for CTCLIPFeatureDataset and ctclip_collate_fn."""

import csv
import tempfile
from pathlib import Path

import pytest
import torch

from aadp.data.ctclip_feature_dataset import (
    CTCLIPFeatureDataset,
    ctclip_collate_fn,
    build_instruction_for_task,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_dummy_features(path: Path, stem: str) -> Path:
    """Save a (24, 24, 24, 512) fp16 tensor as <stem>.pt."""
    feat = torch.zeros(24, 24, 24, 512, dtype=torch.float16)
    pt_path = path / f"{stem}.pt"
    torch.save(feat, pt_path)
    return pt_path


def _make_dummy_csv(path: Path, stems: list) -> Path:
    """Create a minimal CT-RATE style CSV with 18 binary label columns."""
    label_cols = [
        "Medical material", "Arterial wall calcification", "Cardiomegaly",
        "Pericardial effusion", "Coronary artery wall calcification",
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
        "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
        "Pleural effusion", "Mosaic attenuation pattern",
        "Peribronchial thickening", "Consolidation", "Bronchiectasis",
        "Interstitial lung disease",
    ]
    csv_path = path / "reports.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["VolumeName", "Findings_EN", "Impressions_EN"] + label_cols,
        )
        writer.writeheader()
        for i, stem in enumerate(stems):
            row = {
                "VolumeName": f"{stem}.nii.gz",
                "Findings_EN": f"Findings for sample {i}.",
                "Impressions_EN": f"Impression {i}.",
            }
            for j, col in enumerate(label_cols):
                row[col] = i % 2  # alternating 0/1
            writer.writerow(row)
    return csv_path


@pytest.fixture
def dummy_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        feat_dir = Path(tmpdir) / "features"
        feat_dir.mkdir()
        stems = ["train_1_a_1", "train_2_a_1", "train_3_a_1"]
        for stem in stems:
            _make_dummy_features(feat_dir, stem)
        csv_path = _make_dummy_csv(Path(tmpdir), stems)
        yield feat_dir, csv_path, stems


# ── build_instruction_for_task ────────────────────────────────────────────────


def test_build_instruction_t1():
    instruction, target = build_instruction_for_task(
        "T1", "Findings: nodule. Impression: malignant.", {}
    )
    assert "radiology report" in instruction.lower()
    assert "nodule" in target


def test_build_instruction_t2():
    instruction, target = build_instruction_for_task(
        "T2", "There is a lung nodule in the right lobe.", {}
    )
    assert "Describe the findings" in instruction
    assert isinstance(target, str) and len(target) > 0


def test_build_instruction_t3_positive():
    label_dict = {"Lung nodule": 1, "Cardiomegaly": 0}
    instruction, target = build_instruction_for_task("T3", "some report", label_dict)
    assert "Answer yes or no" in instruction
    assert target in ("Yes.", "No.")


def test_build_instruction_t3_unknown_task():
    with pytest.raises(ValueError):
        build_instruction_for_task("T99", "report", {})


# ── CTCLIPFeatureDataset ──────────────────────────────────────────────────────


def test_dataset_len(dummy_data):
    feat_dir, csv_path, stems = dummy_data
    ds = CTCLIPFeatureDataset(str(feat_dir), str(csv_path))
    assert len(ds) == len(stems)


def test_dataset_getitem_shapes(dummy_data):
    feat_dir, csv_path, _ = dummy_data
    ds = CTCLIPFeatureDataset(str(feat_dir), str(csv_path))
    item = ds[0]

    assert "features" in item
    assert item["features"].shape == (24, 576, 512)
    assert item["features"].dtype == torch.float32
    assert isinstance(item["instruction"], str)
    assert isinstance(item["target"], str)
    assert isinstance(item["patient_id"], str)
    assert isinstance(item["label_dict"], dict)
    assert len(item["label_dict"]) == 18


def test_dataset_label_dict_binary(dummy_data):
    feat_dir, csv_path, _ = dummy_data
    ds = CTCLIPFeatureDataset(str(feat_dir), str(csv_path))
    item = ds[0]
    for v in item["label_dict"].values():
        assert v in (0, 1)


def test_dataset_missing_pt_dropped(dummy_data):
    feat_dir, csv_path, stems = dummy_data
    # Remove one .pt file
    (feat_dir / f"{stems[0]}.pt").unlink()
    ds = CTCLIPFeatureDataset(str(feat_dir), str(csv_path))
    assert len(ds) == len(stems) - 1


def test_dataset_max_samples(dummy_data):
    feat_dir, csv_path, stems = dummy_data
    ds = CTCLIPFeatureDataset(str(feat_dir), str(csv_path), max_samples=2)
    assert len(ds) == 2


# ── ctclip_collate_fn ─────────────────────────────────────────────────────────


def test_collate_fn_shapes(dummy_data):
    feat_dir, csv_path, _ = dummy_data
    ds = CTCLIPFeatureDataset(str(feat_dir), str(csv_path))
    batch = [ds[i] for i in range(3)]
    collated = ctclip_collate_fn(batch)

    assert collated["features"].shape == (3, 24, 576, 512)
    assert collated["features"].dtype == torch.float32
    assert len(collated["instruction"]) == 3
    assert len(collated["target"]) == 3
    assert len(collated["patient_id"]) == 3
    assert collated["labels"].shape == (3, 18)
    assert collated["labels"].dtype == torch.float32


def test_collate_fn_label_consistency(dummy_data):
    feat_dir, csv_path, _ = dummy_data
    ds = CTCLIPFeatureDataset(str(feat_dir), str(csv_path))
    batch = [ds[0], ds[0]]  # same item twice
    collated = ctclip_collate_fn(batch)
    # Both rows should be identical
    assert torch.equal(collated["labels"][0], collated["labels"][1])
