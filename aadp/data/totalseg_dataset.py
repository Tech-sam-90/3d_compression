from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

# ── Official TotalSegmentator v1 structures (104 total) ───────────────────────
# These are the exact filenames (without .nii.gz) produced by TotalSegmentator.
# Source: Wasserthal et al. 2023, "TotalSegmentator: Robust Segmentation of 104
# Important Anatomical Structures in CT Images."

TOTALSEG_STRUCTURES: List[str] = [
    # Organs (13)
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    # Lung lobes (5)
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    # Vertebrae — L5→C1 (24)
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
    "vertebrae_C7",
    "vertebrae_C6",
    "vertebrae_C5",
    "vertebrae_C4",
    "vertebrae_C3",
    "vertebrae_C2",
    "vertebrae_C1",
    # Airways (1)
    "trachea",
    # Cardiac (6)
    "heart_myocardium",
    "heart_atrium_left",
    "heart_ventricle_left",
    "heart_atrium_right",
    "heart_ventricle_right",
    "pulmonary_artery",
    # Brain (1)
    "brain",
    # Iliac vessels (4)
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    # Bowel (3)
    "small_bowel",
    "duodenum",
    "colon",
    # Ribs — left 1→12, right 1→12 (24)
    "rib_left_1",
    "rib_left_2",
    "rib_left_3",
    "rib_left_4",
    "rib_left_5",
    "rib_left_6",
    "rib_left_7",
    "rib_left_8",
    "rib_left_9",
    "rib_left_10",
    "rib_left_11",
    "rib_left_12",
    "rib_right_1",
    "rib_right_2",
    "rib_right_3",
    "rib_right_4",
    "rib_right_5",
    "rib_right_6",
    "rib_right_7",
    "rib_right_8",
    "rib_right_9",
    "rib_right_10",
    "rib_right_11",
    "rib_right_12",
    # Upper-limb bones (4)
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    # Clavicles (2)
    "clavicula_left",
    "clavicula_right",
    # Lower-limb bones (4)
    "femur_left",
    "femur_right",
    "hip_left",
    "hip_right",
    # Pelvis / sacrum (2)
    "sacrum",
    "face",
    # Gluteal muscles (6)
    "gluteus_maximus_left",
    "gluteus_maximus_right",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "gluteus_minimus_left",
    "gluteus_minimus_right",
    # Back muscles (2)
    "autochthon_left",
    "autochthon_right",
    # Hip flexors (2)
    "iliopsoas_left",
    "iliopsoas_right",
    # Urinary (1)
    "urinary_bladder",
]

assert len(TOTALSEG_STRUCTURES) == 104, (
    f"Expected 104 structures, got {len(TOTALSEG_STRUCTURES)}"
)

# ── Data record ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TotalSegMask:
    """One binary segmentation mask from TotalSegmentator."""

    patient_id: str
    structure_name: str
    mask_tensor: torch.Tensor  # shape (D, H, W), dtype torch.bool


# ── Dataset ───────────────────────────────────────────────────────────────────


class TotalSegmentatorDataset(Dataset):
    """PyTorch Dataset for TotalSegmentator binary segmentation masks.

    Used exclusively for evaluation (Anatomical Localisation Accuracy / Dice).
    Never used as a training signal.

    Expected directory layout::

        masks_root/
        ├── s0001/
        │   ├── liver.nii.gz
        │   ├── spleen.nii.gz
        │   └── ...
        └── s0002/
            └── ...

    Each sub-directory is treated as one patient. Only .nii.gz files whose
    stem matches a name in TOTALSEG_STRUCTURES are indexed; any extra files
    are ignored. Not all patients need all 104 structures.

    ``__len__`` returns the total number of (patient, structure) pairs that
    actually exist on disk. ``__getitem__`` loads the mask on demand.
    """

    def __init__(self, masks_root: Union[str, Path]) -> None:
        root = Path(masks_root)
        if not root.exists():
            raise FileNotFoundError(f"masks_root not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"masks_root is not a directory: {root}")

        self._root = root
        # Sorted for deterministic ordering.
        self._index: List[Tuple[str, str]] = []
        self._patient_structures: Dict[str, List[str]] = {}

        for patient_dir in sorted(root.iterdir()):
            if not patient_dir.is_dir():
                continue
            pid = patient_dir.name
            existing = [
                p.stem.replace(".nii", "")   # handle both .nii and .nii.gz stems
                for p in sorted(patient_dir.glob("*.nii.gz"))
                if p.stem.replace(".nii", "") in TOTALSEG_STRUCTURES
            ]
            self._patient_structures[pid] = existing
            for struct in existing:
                self._index.append((pid, struct))

    # ── Class methods ──────────────────────────────────────────────────────────

    @classmethod
    def load_from_huggingface(cls, token: str = None) -> "TotalSegmentatorDataset":
        _ = token  # intentionally unused; method always raises
        raise NotImplementedError(
            "TotalSegmentator masks are not available via HuggingFace. "
            "Pass masks_root to load from a local directory."
        )

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> TotalSegMask:
        patient_id, structure_name = self._index[idx]
        mask = self.load_mask(patient_id, structure_name)
        return TotalSegMask(
            patient_id=patient_id,
            structure_name=structure_name,
            mask_tensor=mask,
        )

    # ── Evaluation helpers ─────────────────────────────────────────────────────

    def get_structures_for_patient(self, patient_id: str) -> List[str]:
        """Return structure names that actually exist on disk for this patient."""
        return list(self._patient_structures.get(patient_id, []))

    def load_mask(self, patient_id: str, structure_name: str) -> torch.Tensor:
        """Load and return a single binary mask as a (D, H, W) torch.bool tensor.

        Called directly by the Dice metric during evaluation.
        """
        path = self._root / patient_id / f"{structure_name}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(
                f"Mask not found for patient '{patient_id}', "
                f"structure '{structure_name}': {path}"
            )
        arr = _load_nifti_as_bool_array(path)
        return torch.from_numpy(arr)


# ── NIfTI loading ─────────────────────────────────────────────────────────────


def _load_nifti_as_bool_array(path: Path) -> np.ndarray:
    """Load a NIfTI binary mask into a (D, H, W) bool array.

    NIfTI stores data as (X, Y, Z); we transpose to (Z, X, Y) = (D, H, W)
    to match the CT volume convention used throughout this project.
    """
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj, dtype=np.uint8)  # masks are 0/1 integers
    if arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)               # (X, Y, Z) → (D, H, W)
    return arr.astype(bool)
