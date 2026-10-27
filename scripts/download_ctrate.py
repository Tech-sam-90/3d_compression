#!/usr/bin/env python3
"""
CT-RATE Download Script (fixed splits)
=======================================
Downloads dataset/train_fixed/ and dataset/valid_fixed/ from CT-RATE.

  1. Uses train_fixed / valid_fixed instead of train / valid
  2. Reads HF token from ~/.hf_token or $HF_TOKEN — never hardcoded
  3. Downloads metadata CSVs first, then volumes

Run inside a tmux session on the Narval login node (compute nodes have
no internet access). Resumable — set --start-at to skip already-downloaded
batches if interrupted; already-downloaded files are also skipped
automatically on any rerun (checked via out_path.exists()).

Two things worth verifying before a full run (see docs/CTRATE_DOWNLOAD_NOTES.md
if present, or just check manually): (1) CT-RATE is a gated HF dataset — the
token's account must have been separately granted access on the dataset's HF
page, or every download will 401; (2) the train_fixed/valid_fixed subfolder
layout assumed by volume_path() below is reconstructed from the volume name,
not verified against the live repo tree — worth a small --start-at test run
before committing to the full ~50k-volume download.

Usage:
    python download_ctrate.py              # downloads both splits
    python download_ctrate.py --split train_fixed --start-at 500
    python download_ctrate.py --split valid_fixed
"""

import os
import sys
import shutil
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path("/scratch/sadeniji/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "ctrate_download.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID     = "ibrahimhamamci/CT-RATE"
DEST_DIR    = Path("/scratch/sadeniji/ct_rate_raw")
# Matches huggingface_hub's own snapshot_download(max_workers=8) default — a
# moderate concurrency level that speeds up the mostly-I/O-wait single-file-
# at-a-time loop below without being aggressive enough to trip HF-side rate
# limiting or saturate the login node's shared network unfairly. Tune via
# --workers if the observed throughput suggests headroom either way.
DEFAULT_WORKERS = 8


def get_hf_token() -> str:
    """Read HF token from env var or ~/.hf_token. Never hardcode."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token_file = Path("~/.hf_token").expanduser()
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token
    log.error(
        "No HF token found.\n"
        "  Option A: export HF_TOKEN=hf_...\n"
        "  Option B: echo 'hf_...' > ~/.hf_token"
    )
    sys.exit(1)


def check_disk_space(dest: Path, required_gb: float = 14_000) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(dest).free / 1e9
    log.info(f"Disk: {free_gb:.0f} GB free at {dest}  (need ~{required_gb:.0f} GB)")
    if free_gb < required_gb:
        log.error(f"Only {free_gb:.0f} GB free — need {required_gb:.0f} GB. Aborting.")
        sys.exit(1)


def download_csvs(token: str, dest: Path) -> None:
    """Download metadata and label CSVs first — small files, needed for volume list."""
    from huggingface_hub import snapshot_download
    log.info("Phase 1: downloading metadata CSVs...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        token=token,
        allow_patterns=[
            "dataset/radiology_text_reports/**",
            "dataset/multi_abnormality_labels/**",
            "dataset/metadata/**",
            "dataset/README.md",
            "dataset/data_correction_note.md",
        ],
    )
    log.info("Phase 1 complete — CSVs downloaded.\n")


def get_volume_names(dest: Path, split: str) -> list:
    """
    Read volume names from the multi_abnormality_labels CSV for the given split.
    Falls back to radiology_text_reports CSV if labels CSV not found.
    Returns a list of volume name strings like 'train_1_a_1.nii.gz'.
    """
    import pandas as pd

    # Try multi_abnormality_labels first (has VolumeName column)
    labels_dir = dest / "dataset" / "multi_abnormality_labels"
    reports_dir = dest / "dataset" / "radiology_text_reports"

    # Split name mapping: train_fixed -> train, valid_fixed -> valid for CSV lookup
    base_split = split.replace("_fixed", "")

    # Substring glob rather than exact filenames: the repo's actual naming
    # isn't consistent between splits (train_predicted_labels.csv /
    # train_reports.csv vs valid_predicted_labels.csv / validation_reports.csv
    # -- "valid" is a substring of "validation", so this still matches both).
    csv_path = None
    for d in (labels_dir, reports_dir):
        if not d.exists():
            continue
        matches = sorted(d.glob(f"*{base_split}*.csv"))
        if matches:
            csv_path = matches[0]
            break

    if csv_path is None:
        # List what CSVs actually exist to help debug
        found = list(labels_dir.glob("*.csv")) + list(reports_dir.glob("*.csv"))
        log.error(
            f"Could not find a labels/reports CSV for split '{split}'.\n"
            f"CSVs found: {[str(f) for f in found]}\n"
            "Check the multi_abnormality_labels/ and radiology_text_reports/ folders."
        )
        sys.exit(1)

    log.info(f"Reading volume names from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Column may be 'VolumeName' or 'volume_name' depending on CSV version
    col = next((c for c in df.columns if "volume" in c.lower()), None)
    if col is None:
        log.error(f"No volume name column found in {csv_path}. Columns: {list(df.columns)}")
        sys.exit(1)

    names = df[col].tolist()
    log.info(f"Found {len(names)} volumes for split '{split}'")
    return names


def volume_path(split: str, name: str) -> tuple:
    """
    Given split='train_fixed' and name='train_1_a_1.nii.gz',
    return (subfolder, filename) matching HF repo directory structure:
      dataset/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz
    """
    parts = name.replace(".nii.gz", "").split("_")
    # folder:    train_1  (parts[0] + '_' + parts[1])
    # subfolder: train_1_a (parts[0..2])
    folder    = "_".join(parts[:2])        # e.g. train_1
    subfolder = "_".join(parts[:3])        # e.g. train_1_a
    hf_subfolder = f"dataset/{split}/{folder}/{subfolder}"
    return hf_subfolder, name


def _download_one(split: str, token: str, dest: Path, name: str):
    """Download a single volume. Returns (name, status) where status is
    "skipped", "downloaded", or the caught Exception on failure. Kept as a
    free function (not a closure) so it's picklable/thread-safe to submit
    from ThreadPoolExecutor without touching any shared mutable state —
    all counters are aggregated back in the caller from return values."""
    from huggingface_hub import hf_hub_download

    subfolder, filename = volume_path(split, name)
    out_path = dest / subfolder / filename
    if out_path.exists():
        return name, "skipped"
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
            subfolder=subfolder,
            filename=filename,
            local_dir=str(dest),
        )
        return name, "downloaded"
    except Exception as e:
        return name, e


def download_volumes(split: str, token: str, dest: Path,
                     volume_names: list, start_at: int = 0,
                     max_workers: int = DEFAULT_WORKERS) -> None:
    from tqdm import tqdm

    total = len(volume_names)
    remaining = volume_names[start_at:]
    log.info(f"Phase 2: downloading {total} volumes for split '{split}' "
             f"(starting at index {start_at}, {max_workers} parallel workers)...")

    skipped = 0
    downloaded = 0
    failed = []
    done = 0

    # hf_hub_download is I/O-bound (network wait dominates) and is the same
    # thread-safe call huggingface_hub's own snapshot_download(max_workers=...)
    # uses internally for concurrent multi-file downloads within one repo —
    # this is the same mechanism, just applied to our own already-known
    # volume list instead of re-discovering it via a repo tree scan.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one, split, token, dest, name): name
            for name in remaining
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                            desc=split, unit="vol"):
            name = futures[future]
            _, status = future.result()
            done += 1
            if status == "skipped":
                skipped += 1
            elif status == "downloaded":
                downloaded += 1
            else:
                log.warning(f"  FAILED: {name}  ({status})")
                failed.append(name)

            if done % 1000 == 0:
                log.info(f"  Progress: {done}/{len(remaining)} (of {total} total)  "
                         f"downloaded={downloaded}  skipped={skipped}  failed={len(failed)}")

    log.info(f"\nSplit '{split}' done: "
             f"{downloaded} downloaded, {skipped} already existed, {len(failed)} failed")
    if failed:
        failed_path = LOG_DIR / f"failed_{split}.txt"
        failed_path.write_text("\n".join(failed))
        log.warning(f"Failed volumes written to {failed_path}")


def main():
    parser = argparse.ArgumentParser(description="Download CT-RATE fixed splits")
    parser.add_argument(
        "--split",
        choices=["train_fixed", "valid_fixed", "both"],
        default="both",
        help="Which split to download (default: both)",
    )
    parser.add_argument(
        "--start-at", type=int, default=0,
        help="Resume from this volume index (default: 0)",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEST_DIR,
        help=f"Destination directory (default: {DEST_DIR})",
    )
    parser.add_argument(
        "--skip-space-check", action="store_true",
        help="Skip the disk space check",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel download workers (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("CT-RATE Fixed Splits Download")
    log.info(f"  Split       : {args.split}")
    log.info(f"  Destination : {args.dest}")
    log.info(f"  Log file    : {log_file}")
    log.info(f"  Started at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    token = get_hf_token()

    if not args.skip_space_check:
        check_disk_space(args.dest)

    # Phase 1: CSVs
    download_csvs(token, args.dest)

    # Phase 2: volumes
    splits = (
        ["train_fixed", "valid_fixed"] if args.split == "both"
        else [args.split]
    )

    t0 = time.time()
    for split in splits:
        names = get_volume_names(args.dest, split)
        start = args.start_at if split == splits[0] else 0
        download_volumes(split, token, args.dest, names, start_at=start,
                          max_workers=args.workers)

    elapsed = time.time() - t0
    log.info(f"\nAll done in {elapsed/3600:.1f} hours.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
