import numpy as np
import nibabel as nib
import pytest
import torch
from pathlib import Path

from aadp.data.totalseg_dataset import (
    TOTALSEG_STRUCTURES,
    TotalSegMask,
    TotalSegmentatorDataset,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_STRUCTURES_UNDER_TEST = ["liver", "spleen", "trachea"]


def _write_mask(path: Path, shape=(10, 10, 10), value: int = 1) -> None:
    """Write a small binary NIfTI mask to disk."""
    arr = np.zeros(shape, dtype=np.uint8)
    arr[2:8, 2:8, 2:8] = value          # a non-trivial foreground region
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(img, str(path))


@pytest.fixture
def masks_root(tmp_path: Path) -> Path:
    """Synthetic masks_root with 2 patients × 3 structures each."""
    for pid in ("s0001", "s0002"):
        patient_dir = tmp_path / pid
        patient_dir.mkdir()
        for struct in _STRUCTURES_UNDER_TEST:
            _write_mask(patient_dir / f"{struct}.nii.gz")
    return tmp_path


# ── Module-level constant ─────────────────────────────────────────────────────


def test_totalseg_structures_has_104() -> None:
    assert len(TOTALSEG_STRUCTURES) == 104


def test_totalseg_structures_no_duplicates() -> None:
    assert len(TOTALSEG_STRUCTURES) == len(set(TOTALSEG_STRUCTURES))


def test_known_structures_present() -> None:
    for s in ("liver", "spleen", "trachea", "aorta", "lung_upper_lobe_left"):
        assert s in TOTALSEG_STRUCTURES, f"'{s}' missing from TOTALSEG_STRUCTURES"


# ── __len__ ───────────────────────────────────────────────────────────────────


def test_len_equals_patients_times_structures(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    assert len(ds) == 2 * 3  # 2 patients × 3 structures


# ── __getitem__ ───────────────────────────────────────────────────────────────


def test_getitem_returns_totalseg_mask(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    item = ds[0]
    assert isinstance(item, TotalSegMask)


def test_getitem_patient_id_is_string(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    assert isinstance(ds[0].patient_id, str)


def test_getitem_structure_name_is_string(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    assert isinstance(ds[0].structure_name, str)


def test_getitem_mask_is_bool_tensor(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    assert ds[0].mask_tensor.dtype == torch.bool


def test_getitem_mask_shape(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    # NIfTI (10, 10, 10) stored as (X, Y, Z); transposed to (D, H, W) = (10, 10, 10)
    assert ds[0].mask_tensor.shape == (10, 10, 10)


def test_getitem_mask_has_foreground(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    assert ds[0].mask_tensor.any()


def test_getitem_all_patient_ids_known(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    pids = {ds[i].patient_id for i in range(len(ds))}
    assert pids == {"s0001", "s0002"}


def test_getitem_all_structures_known(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    structs = {ds[i].structure_name for i in range(len(ds))}
    assert structs == set(_STRUCTURES_UNDER_TEST)


# ── get_structures_for_patient ────────────────────────────────────────────────


def test_get_structures_for_patient_returns_correct_list(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    structs = ds.get_structures_for_patient("s0001")
    assert set(structs) == set(_STRUCTURES_UNDER_TEST)


def test_get_structures_for_patient_only_existing(tmp_path: Path) -> None:
    """A patient with only 1 of 3 structures should only report that one."""
    pid = tmp_path / "s0001"
    pid.mkdir()
    _write_mask(pid / "liver.nii.gz")
    # spleen and trachea deliberately omitted
    ds = TotalSegmentatorDataset(tmp_path)
    assert ds.get_structures_for_patient("s0001") == ["liver"]


def test_get_structures_for_unknown_patient(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    assert ds.get_structures_for_patient("s9999") == []


# ── load_mask ─────────────────────────────────────────────────────────────────


def test_load_mask_returns_bool_tensor(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    mask = ds.load_mask("s0001", "liver")
    assert mask.dtype == torch.bool


def test_load_mask_correct_shape(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    mask = ds.load_mask("s0001", "liver")
    assert mask.shape == (10, 10, 10)


def test_load_mask_missing_raises_file_not_found(masks_root: Path) -> None:
    ds = TotalSegmentatorDataset(masks_root)
    with pytest.raises(FileNotFoundError):
        ds.load_mask("s0001", "brain")  # not written to disk


# ── Error handling ────────────────────────────────────────────────────────────


def test_missing_masks_root_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        TotalSegmentatorDataset("/nonexistent/masks_root")


def test_load_from_huggingface_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="not available via HuggingFace"):
        TotalSegmentatorDataset.load_from_huggingface()


def test_load_from_huggingface_message_mentions_masks_root() -> None:
    with pytest.raises(NotImplementedError, match="masks_root"):
        TotalSegmentatorDataset.load_from_huggingface(token="fake")
