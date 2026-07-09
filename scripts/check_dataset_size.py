"""Report the real on-disk size of CT-RATE (or a subfolder of it) without
downloading anything — uses HuggingFace's repo-tree metadata API, which
returns file sizes directly, so this costs a handful of API calls, not a
25TB download.

Usage:
    python scripts/check_dataset_size.py                     # summarise every top-level folder
    python scripts/check_dataset_size.py train_fixed         # just one folder
    python scripts/check_dataset_size.py train_fixed valid_fixed
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, get_token

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_ID = "ibrahimhamamci/CT-RATE"


def _resolve_token() -> str:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise EnvironmentError("No HuggingFace token found. Set HF_TOKEN in .env.")
    return token


def folder_size_bytes(api: HfApi, path_in_repo: str) -> tuple[int, int]:
    """Return (total_bytes, file_count) for a folder, via metadata only."""
    total = 0
    count = 0
    for entry in api.list_repo_tree(
        REPO_ID, repo_type="dataset", path_in_repo=path_in_repo, recursive=True
    ):
        size = getattr(entry, "size", None)
        if size:  # skips directory entries, which have size=None
            total += size
            count += 1
    return total, count


def human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PB"


def main() -> None:
    token = _resolve_token()
    api = HfApi(token=token)

    targets = sys.argv[1:] or [
        "dataset/train",
        "dataset/train_fixed",
        "dataset/valid",
        "dataset/valid_fixed",
        "dataset/ts_seg",
        "dataset/anatomy_segmentation_labels",
        "dataset/radiology_text_reports",
        "dataset/multi_abnormality_labels",
        "dataset/vqa",
        "dataset/metadata",
    ]
    targets = [t if t.startswith("dataset/") else f"dataset/{t}" for t in targets]

    print(f"Querying {REPO_ID} (metadata only - no files downloaded)\n")
    grand_total = 0
    sizes = {}  # path -> (size, count), so we never have to re-query a folder
    for path in targets:
        try:
            size, count = folder_size_bytes(api, path)
        except Exception as e:
            print(f"{path:45s}  ERROR: {e}")
            continue
        sizes[path] = (size, count)
        grand_total += size
        avg = size / count if count else 0
        print(f"{path:45s}  {human(size):>12s}  ({count:6d} files, avg {human(avg)}/file)")

    print(f"\n{'TOTAL (targets above)':45s}  {human(grand_total):>12s}")

    # A quick, concrete "what would N files cost" estimate using train_fixed's
    # average file size, since that's what our training pipeline actually reads.
    # Reuses the size/count already fetched above instead of re-querying.
    # Note: CT-RATE's "25,692 volumes" (dataset card) is the report/patient-scan
    # count, NOT the file count — most patients have multiple scan files per
    # reported "volume" (e.g. train_1_a_1.nii.gz and train_1_a_2.nii.gz), so the
    # real train_fixed file count is roughly double that (see tf_count below).
    if "dataset/train_fixed" in sizes:
        tf_size, tf_count = sizes["dataset/train_fixed"]
        avg_per_volume = tf_size / tf_count if tf_count else 0
        print(f"\nEstimate using train_fixed average ({human(avg_per_volume)}/file):")
        for n in (1000, 2000, 5000, 10000, tf_count):
            label = f"{n:>6d} files" if n != tf_count else f"{n:>6d} files (= ALL of train_fixed)"
            print(f"  {label}  ~  {human(avg_per_volume * n)}")


if __name__ == "__main__":
    main()
