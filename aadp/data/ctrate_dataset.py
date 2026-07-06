import concurrent.futures
import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterator, Literal, Optional, Tuple

import nibabel as nib # for reading NIfTI files
import numpy as np
import torch
from datasets import load_dataset # for streaming access to HuggingFace datasets
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem, get_token
from torch.utils.data import IterableDataset

from aadp.data.preprocessing import window_and_normalize

# Load .env from the project root so HF_TOKEN is available without a manual login.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── Constants ────────────────────────────────────────────────────────────────

_HF_REPO = "ibrahimhamamci/CT-RATE"

# HuggingFace split name → top-level folder inside dataset/.
# We use the CT-RATE **v2** "fixed" volumes (train_fixed / valid_fixed). In v2
# the DICOM RescaleSlope/Intercept intensity correction is already baked into
# the NIfTI headers, so windowing the raw voxels is valid. The v1 folders
# (train/valid) are UNCORRECTED and must not be windowed directly.
_SPLIT_TO_FOLDER = {"train": "train_fixed", "valid": "valid_fixed"}

# Volumes live at:
#   dataset/{folder}/{name_split}_{pid}/{name_split}_{pid}_{sid}/{VolumeName}
# where {name_split} ("train"/"valid") comes from the VolumeName itself, NOT
# from {folder} — v2 keeps the volume names (e.g. train_1_a_1.nii.gz) and their
# nested train_1/train_1_a folders unchanged; only the top folder gains "_fixed".

# HuggingFace dataset split name used in load_dataset calls (reports/labels
# configs are shared across v1/v2 and keyed by VolumeName).
_SPLIT_TO_HF = {"train": "train", "valid": "validation"}

# Non-chest (brain) scans to exclude, per the CT-RATE data-correction note
# (752 train + 37 valid). Listed one path per line under dataset/metadata/.
_NO_CHEST_FILES = {"train": "no_chest_train.txt", "valid": "no_chest_valid.txt"}
_METADATA_DIR = f"datasets/{_HF_REPO}/dataset/metadata"

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

# this function resolves the HuggingFace token from the following sources, in order:
# 1. Explicitly passed token argument
# 2. HF_TOKEN environment variable (from .env)
# 3. Cached token from huggingface-cli login
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

# Pre-load the labels config into a {volume_name: label_dict} mapping.
# so that the IterableDataset can yield labels without streaming the entire labels dataset.
def _load_labels_lookup(token: str) -> Dict[str, Dict[str, int]]:
    """Pre-load both splits of the labels config into a {volume_name: label_dict} mapping."""
    lookup: Dict[str, Dict[str, int]] = {}
    for hf_split in ("train", "validation"):
        stream = load_dataset(
            _HF_REPO, "labels", 
            split=hf_split, 
            streaming=True, 
            token=token
        )
        for row in stream:
            name = row.get("VolumeName") or row.get("volume_name") or ""
            if not name:
                continue
            lookup[name] = {col: int(row[col]) for col in LABEL_COLUMNS if col in row}
    return lookup

# this function constructs the HuggingFace filesystem path for a 
# NIfTI file based on its volume name and folder (train/valid).
def _volume_hf_path(volume_name: str, folder: str) -> str:
    """Construct the HuggingFace filesystem path for a NIfTI file.

    Layout: dataset/{folder}/{name_split}_{pid}/{name_split}_{pid}_{sid}/{volume_name}
    Example: train_1_a_1.nii.gz (folder="train_fixed") →
             dataset/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz
    """
    stem = volume_name.replace(".nii.gz", "").replace(".nii", "")
    parts = stem.split("_")          # [name_split, pid, sid, rid]
    name_split = parts[0]            # "train"/"valid" from the VolumeName (not the top folder)
    patient_folder = f"{name_split}_{parts[1]}"
    scan_folder = f"{name_split}_{parts[1]}_{parts[2]}"
    return (
        f"datasets/{_HF_REPO}/dataset/{folder}/"
        f"{patient_folder}/{scan_folder}/{volume_name}"
    )

# this function constructs the local filesystem path for a
# NIfTI file based on its volume name and folder (train/valid).
def _local_nifti_path(local_data_dir: Path, volume_name: str, folder: str) -> Path:
    """Mirror of _volume_hf_path but rooted at a local directory."""
    stem = volume_name.replace(".nii.gz", "").replace(".nii", "")
    parts = stem.split("_")
    name_split = parts[0]            # "train"/"valid" from the VolumeName (not the top folder)
    patient_folder = f"{name_split}_{parts[1]}"
    scan_folder = f"{name_split}_{parts[1]}_{parts[2]}"
    return (
        local_data_dir / "dataset" / folder / patient_folder / scan_folder / volume_name
    )

# this function loads a NIfTI file from disk into a numpy array.
def _nifti_to_array(src: Path) -> np.ndarray:
    """Load a NIfTI file from disk into a (D, H, W) float32 array."""
    img = nib.load(str(src))
    arr = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z)
    # Transpose to (Z, X, Y) = (D, H, W).
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


def _load_no_chest_set(fs: HfFileSystem, split: str) -> set:
    """Return the set of non-chest VolumeNames to exclude for ``split``.

    The metadata lists one path per line (e.g. ``train/train_10100/.../
    train_10100_a_1.nii.gz``); we key on the basename, which equals the
    ``VolumeName`` used by the reports/labels configs.  Returns an empty set
    (i.e. no filtering) if the list can't be read.
    """
    path = f"{_METADATA_DIR}/{_NO_CHEST_FILES[split]}"
    names: set = set()
    try:
        with fs.open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.add(line.split("/")[-1])  # basename == VolumeName
    except Exception:
        pass  # metadata unavailable → proceed without filtering
    return names


# ── Bulk pre-download utility ─────────────────────────────────────────────────


def download_subset_to_disk(
    local_data_dir: str,
    n_train: int = 1000,
    n_valid: int = 100,
    token: Optional[str] = None,
    max_workers: int = 4,
    max_gb: Optional[float] = None,
    valid_ratio: float = 0.1,
) -> None:
    """Download a subset of CT-RATE **v2** NIfTI files to a local directory.

    Files mirror the HuggingFace layout:
        {local_data_dir}/dataset/{folder}/{name_split}_{pid}/…/{VolumeName}

    Already-downloaded files are skipped, so this is safe to re-run after a
    Colab disconnect — it resumes where it left off. Non-chest (brain) scans
    are excluded.

    Two stopping modes:

    * **By count** (default): download up to ``n_train`` / ``n_valid`` volumes.
    * **By size** (pass ``max_gb``): download until the cumulative on-disk size
      (already-cached + newly fetched) reaches ``max_gb`` GB across both splits,
      reserving ``valid_ratio`` of the budget for validation. In this mode
      ``n_train`` / ``n_valid`` act only as *safety upper bounds* on the count,
      so raise them if the budget isn't reached (e.g. 6000 / 600 for ~45 GB).

    Args:
        local_data_dir: Root directory (e.g. a mounted Drive path on Colab).
        n_train:        Max training volumes (upper bound in size mode).
        n_valid:        Max validation volumes (upper bound in size mode).
        token:          HuggingFace token. Resolved from env/cache if None.
        max_workers:    Parallel download threads.
        max_gb:         Total size budget in GB across both splits. ``None`` →
                        count-based mode.
        valid_ratio:    Fraction of ``max_gb`` reserved for the validation split.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    resolved_token = _resolve_token(token)
    fs = HfFileSystem(token=resolved_token)
    root = Path(local_data_dir)

    def _fetch_one(volume_name: str, folder: str) -> int:
        """Return the on-disk size in bytes (fetching if needed); 0 on failure."""
        local_path = _local_nifti_path(root, volume_name, folder)
        if local_path.exists():
            return local_path.stat().st_size
        hf_path = _volume_hf_path(volume_name, folder)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with fs.open(hf_path, "rb") as fh:
                raw = fh.read()
            local_path.write_bytes(raw)
            return len(raw)
        except Exception as exc:
            print(f"  Warning: skipping {volume_name} ({exc})")
            return 0

    # Per-split byte budgets (size mode), or None (count mode).
    if max_gb is not None:
        split_budgets = {
            "train": max_gb * (1.0 - valid_ratio) * 1e9,
            "valid": max_gb * valid_ratio * 1e9,
        }
    else:
        split_budgets = {"train": None, "valid": None}

    for split, count_cap in [("train", n_train), ("valid", n_valid)]:
        folder = _SPLIT_TO_FOLDER[split]
        hf_split = _SPLIT_TO_HF[split]
        budget = split_budgets[split]

        exclude = _load_no_chest_set(fs, split)  # skip non-chest (brain) scans
        stream = load_dataset(
            _HF_REPO, "reports", split=hf_split, streaming=True, token=resolved_token
        )

        target = (
            f"~{budget / 1e9:.1f} GB" if budget is not None
            else f"{count_cap} volumes"
        )
        print(f"\n[{split}] downloading up to {target} …")

        done_bytes = 0
        done_count = 0
        stream_iter = iter(stream)
        bar = tqdm(desc=f"{split}", unit="vol") if tqdm is not None else None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            while True:
                # Stop conditions: byte budget (size mode) or count cap.
                if budget is not None and done_bytes >= budget:
                    break
                if done_count >= count_cap:
                    break

                # Pull the next wave of (filtered) names from the reports stream.
                wave: list[str] = []
                exhausted = False
                while len(wave) < max_workers and done_count + len(wave) < count_cap:
                    try:
                        sample = next(stream_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    name = sample.get("VolumeName") or sample.get("volume_name") or ""
                    if name and name not in exclude:
                        wave.append(name)

                if not wave:
                    break

                futures = {pool.submit(_fetch_one, n, folder): n for n in wave}
                for fut in concurrent.futures.as_completed(futures):
                    size = fut.result()
                    if size > 0:
                        done_bytes += size
                        done_count += 1
                    if bar is not None:
                        bar.update(1)

                if exhausted:
                    break

        if bar is not None:
            bar.close()
        print(f"[{split}] {done_count} volumes on disk (~{done_bytes / 1e9:.2f} GB)")

    print(f"\nDone. CT-RATE subset cached at: {root}")


# ── Dataset ───────────────────────────────────────────────────────────────────


class CTRATEDataset(IterableDataset):
    """Streaming PyTorch IterableDataset for the CT-RATE dataset.

    Uses the **v2** intensity-corrected volumes (train_fixed / valid_fixed), so
    the HU windowing in ``window_and_normalize`` operates on correct voxel
    values.  Non-chest (brain) scans flagged in the CT-RATE data-correction note
    are skipped by default (``filter_non_chest=True``).

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
        filter_non_chest: bool = True,
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

        # Non-chest (brain) scans to drop from this split, if filtering is on.
        self._exclude = (
            _load_no_chest_set(self._fs, split) if filter_non_chest else set()
        )

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

        # Drop non-chest (brain) scans flagged by the CT-RATE correction note.
        if volume_name in self._exclude:
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
