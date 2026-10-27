#!/usr/bin/env python3
"""merge_eval_shards.py — combine qualitative_eval_v3.py --num-shards>1
output into one CSV and recompute the TRUE aggregate T1/T2/T3 metrics.

Deliberately does not average the per-shard "PARTIAL" summaries each shard
process prints — those are means over unequal-n subsets and averaging
averages does not equal the true global mean/accuracy (same class of bug
_validate_e2e() had in scripts/train_ctclip.py before its all_reduce fix).
Instead this re-scores every row directly from the merged CSV using the
exact same functions qualitative_eval_v3.py itself uses, so there is exactly
one scoring implementation, not two that can drift apart.

Usage:
    python scripts/merge_eval_shards.py \\
        --output /scratch/sadeniji/ictc/eval/stage2_e2e_eval_n3039.csv \\
        --num-shards 4
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualitative_eval_v3 import (  # noqa: E402
    extract_t2_entity, rouge_l, t3_label_from_target, t3_prediction_from_generated,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True,
                         help="Same --output path passed to each shard (without .shardN — "
                              "this script derives shard filenames itself and writes the "
                              "merged result back to this exact path).")
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    all_rows = []
    for shard_id in range(args.num_shards):
        shard_path = out_path.with_suffix(f".shard{shard_id}{out_path.suffix}")
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file: {shard_path} — did all {args.num_shards} shards finish?")
        with open(shard_path, newline="") as fh:
            shard_rows = list(csv.DictReader(fh))
        print(f"shard {shard_id}: {len(shard_rows)} rows from {shard_path}")
        all_rows.extend(shard_rows)

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["volume_name", "task", "instruction", "generated_text", "ground_truth"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} merged rows to {out_path}")

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    t1_scores, t2_scores = [], []
    t3_correct, t3_scored = 0, 0
    for row in all_rows:
        task, instruction = row["task"], row["instruction"]
        generated, target = row["generated_text"], row["ground_truth"]
        if task == "T1":
            t1_scores.append(rouge_l(scorer, generated, target))
        elif task == "T2":
            entity = extract_t2_entity(instruction)
            from aadp.data.instruction_builder import sentences_containing
            scoped_generated = sentences_containing(generated, entity) if entity else generated
            t2_scores.append(rouge_l(scorer, scoped_generated, target))
        elif task == "T3":
            gt_label = t3_label_from_target(target)
            pred_label = t3_prediction_from_generated(generated)
            if gt_label is not None:
                t3_scored += 1
                t3_correct += int(pred_label == gt_label)

    t1_mean = sum(t1_scores) / len(t1_scores) if t1_scores else float("nan")
    t2_mean = sum(t2_scores) / len(t2_scores) if t2_scores else float("nan")
    t3_acc = t3_correct / t3_scored if t3_scored else float("nan")

    print("\n=== Merged Summary (true aggregate, not an average of shard partials) ===")
    print(f"T1 (n={len(t1_scores)}): mean ROUGE-L = {t1_mean:.4f}")
    print(f"T2 (n={len(t2_scores)}): mean entity-scoped ROUGE-L = {t2_mean:.4f}")
    print(f"T3 (n={t3_scored}): binary accuracy = {t3_acc:.4f} ({t3_correct}/{t3_scored})")


if __name__ == "__main__":
    main()
