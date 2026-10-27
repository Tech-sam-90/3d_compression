#!/usr/bin/env python3
"""compute_full_t1_metrics.py — full Argus Table 2 metric suite (BLEU-4,
ROUGE-L, METEOR, CIDEr, Avg-NLP, GREEN, RaTEScore, RadGraph-XL F1) for the
T1 (full report) rows of an existing qualitative_eval_v3.py output CSV.

Deliberately does NOT re-run generation — qualitative_eval_v3.py's CSV
already has generated_text/ground_truth for every row; this only adds the
metric set evaluate_checkpoint.py used for the old features-based
checkpoints (aadp/evaluation/metrics/compute_all.py's compute_all_metrics(),
the same single implementation both entry points share, per that module's
own docstring) but which qualitative_eval_v3.py itself only ever computed
ROUGE-L for.

Usage:
    python scripts/compute_full_t1_metrics.py \\
        --csv /scratch/sadeniji/ictc/eval/stage2_e2e_eval_n3039.csv \\
        --output /scratch/sadeniji/ictc/eval/stage2_e2e_t1_full_metrics.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aadp.evaluation.metrics.compute_all import compute_all_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="qualitative_eval_v3.py output CSV (merged, not a .shardN file).")
    parser.add_argument("--output", required=True, help="Path to write the metrics JSON.")
    args = parser.parse_args()

    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    t1_rows = [r for r in rows if r["task"] == "T1"]
    print(f"{args.csv}: {len(rows)} total rows, {len(t1_rows)} T1 rows")

    predictions = [r["generated_text"] for r in t1_rows]
    references = [r["ground_truth"] for r in t1_rows]

    print("Computing compute_all_metrics() over T1 rows (this loads RadGraph-XL/RaTEScore models, may take a few minutes)...")
    scores = compute_all_metrics(predictions, references)
    scores["n"] = len(t1_rows)
    scores["source_csv"] = args.csv

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(scores, fh, indent=2)

    print(f"\nWrote metrics to {out_path}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
