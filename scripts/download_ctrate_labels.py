#!/usr/bin/env python3
"""Download CT-RATE multi-abnormality label CSVs and merge them with the
text-reports CSVs.

CT-RATE's ``dataset/radiology_text_reports/{train,validation}_reports.csv``
only carries the free-text report columns (VolumeName, ClinicalInformation_EN,
Technique_EN, Findings_EN, Impressions_EN) — the 18 binary abnormality labels
live in a separate ``dataset/multi_abnormality_labels/`` folder. This script
downloads that folder's CSVs, joins them onto the reports CSVs on VolumeName,
and writes merged CSVs that CTCLIPFeatureDataset can read directly (it
auto-detects label columns from the merged file).

Usage
-----
    python scripts/download_ctrate_labels.py \\
        --train_csv /path/to/dataset/radiology_text_reports/train_reports.csv \\
        --valid_csv /path/to/dataset/radiology_text_reports/validation_reports.csv \\
        --output_dir /path/to/dataset/merged \\
        --hf_token $HF_TOKEN
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, get_token

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_ID = "ibrahimhamamci/CT-RATE"
LABELS_FOLDER = "dataset/multi_abnormality_labels"


def _resolve_token(token: Optional[str]) -> str:
    if token:
        return token
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    cached = get_token()
    if cached:
        return cached
    raise EnvironmentError(
        "No HuggingFace token found. Pass --hf_token, set HF_TOKEN in .env, "
        "or run `huggingface-cli login`. CT-RATE is a gated dataset."
    )


def _find_label_file(api: HfApi, keyword: str) -> str:
    """Return the path (within the repo) of the labels CSV matching ``keyword``
    (e.g. "train" or "valid"), discovered by listing the repo tree rather than
    assuming a hard-coded filename.
    """
    entries = api.list_repo_tree(
        REPO_ID, repo_type="dataset", path_in_repo=LABELS_FOLDER, recursive=False
    )
    candidates = [
        e.path for e in entries
        if e.path.endswith(".csv") and keyword in Path(e.path).name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No CSV matching '{keyword}' found under {LABELS_FOLDER} in {REPO_ID}."
        )
    if len(candidates) > 1:
        print(f"Warning: multiple candidates for '{keyword}', using first: {candidates}")
    return candidates[0]


def _download_labels_csv(api: HfApi, keyword: str, token: str, output_dir: Path) -> Path:
    repo_path = _find_label_file(api, keyword)
    print(f"Found labels CSV for '{keyword}': {repo_path}")
    local_path = hf_hub_download(
        REPO_ID,
        filename=repo_path,
        repo_type="dataset",
        token=token,
        local_dir=str(output_dir / "_raw"),
    )
    return Path(local_path)


def _merge(reports_csv: Path, labels_csv: Path) -> pd.DataFrame:
    reports_df = pd.read_csv(reports_csv)
    labels_df = pd.read_csv(labels_csv)

    if "VolumeName" not in reports_df.columns:
        raise ValueError(f"'VolumeName' column missing from {reports_csv}")
    if "VolumeName" not in labels_df.columns:
        raise ValueError(f"'VolumeName' column missing from {labels_csv}")

    merged = reports_df.merge(labels_df, on="VolumeName", how="inner")
    if len(merged) == 0:
        raise ValueError(
            f"Merge of {reports_csv.name} and {labels_csv.name} produced 0 rows "
            "— check that VolumeName values match between the two CSVs."
        )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_token", default=None, help="HuggingFace token (else HF_TOKEN env/.env or cached login)")
    parser.add_argument("--train_csv", required=True, help="Path to train_reports.csv")
    parser.add_argument("--valid_csv", required=True, help="Path to validation_reports.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to write merged CSVs and raw downloads")
    args = parser.parse_args()

    token = _resolve_token(args.hf_token)
    api = HfApi(token=token)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_labels_csv = _download_labels_csv(api, "train", token, output_dir)
    valid_labels_csv = _download_labels_csv(api, "valid", token, output_dir)

    train_merged = _merge(Path(args.train_csv), train_labels_csv)
    valid_merged = _merge(Path(args.valid_csv), valid_labels_csv)

    train_out = output_dir / "train_merged.csv"
    valid_out = output_dir / "valid_merged.csv"
    train_merged.to_csv(train_out, index=False)
    valid_merged.to_csv(valid_out, index=False)

    for name, path, df in (("train", train_out, train_merged), ("valid", valid_out, valid_merged)):
        print(f"\n[{name}] wrote {path}")
        print(f"[{name}] shape: {df.shape}")
        print(f"[{name}] columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
