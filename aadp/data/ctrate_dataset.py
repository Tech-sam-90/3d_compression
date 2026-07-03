import concurrent.futures
import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterator, Literal, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem, get_token
from torch.utils.data import IterableDataset

from aadp.data.preprocessing import window_and_normalize

# Load .env from the project root so HF_TOKEN is available without a manual login.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── Constants ────────────────────────────────────────────────────────────────

_HF_REPO = "ibrahimhamamci/CT-RATE"

# HuggingFace split name → folder name inside dataset/
# Volumes live at: dataset/{folder}/{folder}_{pid}/{folder}_{pid}_{sid}/{name}.nii.gz
_SPLIT_TO_FOLDER = {"train": "train", "valid": "valid"}

# HuggingFace dataset split name used in load_dataset calls
_SPLIT_TO_HF = {"train": "train", "valid": "validation"}

LABEL_COLUMNS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

_EMPTY_LABELS: Dict[str, int] = {col: 0 for col in LABEL_COLUMNS}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_token(token: Optional[str]) -> str:
    if token is not None:
        return token
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    cached = get_token()
    if cached:
        return cached
    raise EnvironmentError(
        "No HuggingFace token found. Set HF_TOKEN in .env, run "
        "`huggingface-cli login`, or pass token= explicitly. "
        "CT-RATE is a gated dataset and requires authentication."
    )


def _load_labels_lookup(token: str) -> Dict[str, Dict[str, int]]:
    """Pre-load both splits of the labels config into a {volume_name: label_dict} mapping."""
    lookup: Dict[str, Dict[str, int]] = {}
    for hf_split in ("train", "validation"):
        stream = load_dataset(
            _HF_REPO, "labels", split=hf_split, streaming=True, token=token
        )
        for row in stream:
            name = row.get("VolumeName") or row.get("volume_name") or ""
            if not name:
                continue
            lookup[name] = {col: int(row[col]) for col in LABEL_COLUMNS if col in row}
    return lookup


def _volume_hf_path(volume_name: str, folder: str) -> str:
    """Construct the HuggingFace filesystem path for a NIfTI file.

    Layout: dataset/{folder}/{folder}_{pid}/{folder}_{pid}_{sid}/{volume_name}
    Example: train_1_a_1.nii.gz →
             dataset/train/train_1/train_1_a/train_1_a_1.nii.gz
    """
    stem = volume_name.replace(".nii.gz", "").replace(".nii", "")
    parts = stem.split("_")          # [split, pid, sid, rid]
    patient_folder = f"{folder}_{parts[1]}"
    scan_folder = f"{folder}_{parts[1]}_{parts[2]}"
    return (
        f"datasets/{_HF_REPO}/dataset/{folder}/"
        f"{patient_folder}/{scan_folder}/{volume_name}"
    )


def _local_nifti_path(local_data_dir: Path, volume_name: str, folder: str) -> Path:
    """Mirror of _volume_hf_path but rooted at a local directory."""
    stem = volume_name.replace(".nii.gz", "").replace(".nii", "")
    parts = stem.split("_")
    patient_folder = f"{folder}_{parts[1]}"
    scan_folder = f"{folder}_{parts[1]}_{parts[2]}"
    return (
        local_data_dir / "dataset" / folder / patient_folder / scan_folder / volume_name
    )


def _nifti_to_array(src: Path) -> np.ndarray:
    """Load a NIfTI file from disk into a (D, H, W) float32 array."""
    img = nib.load(str(src))
    arr = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z)
    if arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)  # → (D, H, W)
    return arr


def _nifti_stream_to_array(fs: HfFileSystem, hf_path: str) -> np.ndarray:
    """Read a NIfTI file from HfFileSystem into a (D, H, W) float32 array.

    Writes bytes to a temp file so nibabel can handle gzip decompression.
    NIfTI convention is (X, Y, Z); we transpose to (Z, X, Y) = (D, H, W).
    """
    with fs.open(hf_path, "rb") as f:
        raw = f.read()

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        arr = _nifti_to_array(Path(tmp_path))
    finally:
        os.unlink(tmp_path)

    return arr


def _parse_patient_id(volume_name: str) -> str:
    """Return the CT-RATE volume identifier (filename stem without extension).

    This is the full stem, e.g. ``"train_1_a_1"`` for ``train_1_a_1.nii.gz`` —
    the identifier RadGenome-Chest CT and TotalSegmentator annotations are keyed
    on.  Using the full stem (rather than just the numeric patient number) is
    what lets the multi-task sampler and the VTCB evaluators join grounding /
    mask annotations back to the correct volume.
    """
    return volume_name.replace(".nii.gz", "").replace(".nii", "")


# ── Bulk pre-download utility ─────────────────────────────────────────────────


def download_subset_to_disk(
    local_data_dir: str,
    n_train: int = 1000,
    n_valid: int = 100,
    token: Optional[str] = None,
    max_workers: int = 4,
) -> None:
    """Download a fixed subset of CT-RATE NIfTI files to a local directory.

    Files are saved at:
        {local_data_dir}/dataset/{split}/{split}_{pid}/{split}_{pid}_{sid}/{name}.nii.gz

    Already-downloaded files are skipped, so this is safe to re-run after
    a Colab disconnect — it resumes from where it left off.

    Args:
        local_data_dir: Root directory (e.g. a Google Drive path on Colab).
        n_train: Number of training volumes to download.
        n_valid: Number of validation volumes to download.
        token: HuggingFace token. Resolved from env/cache if None.
        max_workers: Parallel download threads.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    resolved_token = _resolve_token(token)
    fs = HfFileSystem(token=resolved_token)
    root = Path(local_data_dir)

    def _fetch_one(volume_name: str, folder: str) -> bool:
        local_path = _local_nifti_path(root, volume_name, folder)
        if local_path.exists():
            return True
        hf_path = _volume_hf_path(volume_name, folder)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with fs.open(hf_path, "rb") as fh:
                raw = fh.read()
            local_path.write_bytes(raw)
            return True
        except Exception as exc:
            print(f"  Warning: skipping {volume_name} ({exc})")
            return False

    for split, n_samples in [("train", n_train), ("valid", n_valid)]:
        folder = _SPLIT_TO_FOLDER[split]
        hf_split = _SPLIT_TO_HF[split]

        print(f"\nCollecting {n_samples} volume names from {split} split …")
        stream = load_dataset(
            _HF_REPO, "reports", split=hf_split, streaming=True, token=resolved_token
        )
        names: list[str] = []
        for sample in stream:
            name = sample.get("VolumeName") or sample.get("volume_name") or ""
            if name:
                names.append(name)
            if len(names) >= n_samples:
                break

        already = sum(
            1 for n in names if _local_nifti_path(root, n, folder).exists()
        )
        print(f"  {already}/{len(names)} already cached — downloading {len(names) - already} more …")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, n, folder): n for n in names}
            if tqdm is not None:
                bar = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"{split} volumes",
                )
            else:
                bar = concurrent.futures.as_completed(futures)
            for future in bar:
                future.result()

    print(f"\nDone. CT-RATE subset cached at: {root}")


# ── Dataset ───────────────────────────────────────────────────────────────────


class CTRATEDataset(IterableDataset):
    """Streaming PyTorch IterableDataset for the CT-RATE dataset.

    If ``local_data_dir`` is provided, volumes that have been pre-downloaded
    (via ``download_subset_to_disk``) are loaded from disk.  Volumes not found
    locally fall back to HuggingFace streaming automatically.

    Yields:
        volume_tensor : torch.Tensor of shape (D, H, W), float32, values in [0, 1]
        report_text   : str  — concatenated Findings + Impressions
        label_dict    : Dict[str, int] — 18 binary abnormality labels
        patient_id    : str  — extracted from the volume filename
    """

    def __init__(
        self,
        split: Literal["train", "valid"] = "train",
        hu_min: float = -1000.0,
        hu_max: float = 400.0,
        shuffle: bool = False,
        shuffle_buffer_size: int = 1000,
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
        local_data_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "valid"):
            raise ValueError(f"split must be 'train' or 'valid', got '{split}'")

        self.split = split
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_samples = max_samples
        self._token = _resolve_token(token)
        self._folder = _SPLIT_TO_FOLDER[split]
        self._hf_split = _SPLIT_TO_HF[split]
        self._local_root = Path(local_data_dir) if local_data_dir else None

        # Labels are tabular — safe to materialise once at init.
        self._labels = _load_labels_lookup(self._token)
        self._fs = HfFileSystem(token=self._token)

    # ── Public ────────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, str, Dict[str, int], str]]:
        # Stream the reports config to iterate over volume names + text.
        stream = load_dataset(
            _HF_REPO,
            "reports",
            split=self._hf_split,
            streaming=True,
            token=self._token,
        )
        if self.shuffle:
            stream = stream.shuffle(buffer_size=self.shuffle_buffer_size)

        count = 0
        for sample in stream:
            if self.max_samples is not None and count >= self.max_samples:
                break

            try:
                result = self._process_sample(sample)
            except Exception:
                # Skip corrupt or inaccessible volumes without crashing the stream.
                continue

            if result is not None:
                yield result
                count += 1

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_sample(
        self, sample: dict
    ) -> Optional[Tuple[torch.Tensor, str, Dict[str, int], str]]:
        volume_name = sample.get("VolumeName") or sample.get("volume_name") or ""
        if not volume_name:
            return None

        patient_id = _parse_patient_id(volume_name)

        # Try local cache first; fall back to HF streaming.
        arr = self._load_volume(volume_name)
        volume_tensor = window_and_normalize(arr, self.hu_min, self.hu_max)

        # Concatenate Findings + Impressions as the report text.
        findings = sample.get("Findings_EN", "") or ""
        impressions = sample.get("Impressions_EN", "") or ""
        report_text = f"{findings.strip()} {impressions.strip()}".strip()

        label_dict = self._labels.get(volume_name, dict(_EMPTY_LABELS))

        return volume_tensor, report_text, label_dict, patient_id

    def _load_volume(self, volume_name: str) -> np.ndarray:
        if self._local_root is not None:
            local_path = _local_nifti_path(self._local_root, volume_name, self._folder)
            if local_path.exists():
                return _nifti_to_array(local_path)
        # Fall back to streaming from HuggingFace.
        hf_path = _volume_hf_path(volume_name, self._folder)
        return _nifti_stream_to_array(self._fs, hf_path)
