# Evaluation Metrics Report — 2026-08-15

All evaluation metrics currently present in the project, as of 2026-08-15.
Compiled from SLURM job logs (`/scratch/sadeniji/ictc/logs/`), merged eval
CSVs and full-metrics JSONs (`/scratch/sadeniji/ictc/eval/`), and
`docs/RESULTS_SUMMARY.md` (dated 2026-08-03).

## 1. New end-to-end (unfrozen-ViT, raw-volume) pipeline

### Stage 1 checkpoint (`ictc_checkpoints_e2e_stage1`)

Job 249604, 2026-08-06, n=3039 eval set.

| Task | n | Metric |
|---|---|---|
| T1 | 1536 | ROUGE-L = 0.2005 |
| T2 (entity-scoped) | 1047 | ROUGE-L = 0.3222 |
| T3 (binary) | 456 | accuracy = 0.4846 (221/456) |

Full T1 NLP metrics (`stage1_e2e_t1_full_metrics.json`, n=1536):

| BLEU-4 | METEOR | ROUGE-L | CIDEr | avg_nlp | RaTEScore | GREEN | RadGraph-XL F1 |
|---|---|---|---|---|---|---|---|
| 0.0644 | 0.2484 | 0.2005 | 0.0106 | 0.1310 | 0.7854 | NaN* | NaN* |

### Stage 2 checkpoint, 1-epoch (superseded)

Job 321603, 2026-08-06.

| Task | n | Metric |
|---|---|---|
| T1 | 1525 | ROUGE-L = 0.2180 |
| T2 | 1074 | ROUGE-L = 0.2831 |
| T3 | 440 | accuracy = 0.6727 (296/440) |

### Stage 2 checkpoint, current (latest — 2026-08-13)

Job 850923, `ictc_checkpoints_e2e_stage2/checkpoint_best`. **Current best/live numbers.**

| Task | n | Metric |
|---|---|---|
| T1 | 1562 | ROUGE-L = 0.2973 |
| T2 | 1019 | ROUGE-L = 0.2764 |
| T3 | 458 | accuracy = 0.6747 (309/458) |

Full T1 NLP metrics (`stage2_e2e_t1_full_metrics.json`, n=1525):

| BLEU-4 | METEOR | ROUGE-L | CIDEr | avg_nlp | RaTEScore | GREEN | RadGraph-XL F1 |
|---|---|---|---|---|---|---|---|
| 0.1117 | 0.2559 | 0.2180 | 0.0213 | 0.1517 | 0.7945 | NaN* | NaN* |

Note: the merged-CSV ROUGE-L above (0.2973) and this json's ROUGE-L (0.2180)
differ because they were computed from different CSV snapshots/runs at
different dates — the json is stale relative to the 08-13 rerun.

## 2. Old pre-extracted-features pipeline (frozen ViT) — legacy

T1-only, from `docs/RESULTS_SUMMARY.md` (dated 2026-08-03, via
`evaluate_checkpoint.py`):

| Job | Checkpoint | n | BLEU-4 | METEOR | ROUGE-L | CIDEr | RaTEScore | RadGraph-XL F1 |
|---|---|---|---|---|---|---|---|---|
| 66272201 | checkpoints/checkpoint_best, step 3500 | 200 | 0.2037 | 0.3579 | 0.3345 | 0.0395 | 0.8158 | 0.1868 |
| 66300376 | checkpoints/checkpoint_best, step 6000 | 200 | 0.1599 | 0.3217 | 0.2641 | 0.0145 | 0.8115 | 0.1569 |
| 66304396 | checkpoints_llama3b, step 4500 | 200 | 0.1325 | 0.2928 | 0.2829 | 0.0237 | 0.8029 | 0.1578 |
| 66320539 | checkpoints/checkpoint_best, step 5500 | 200 | 0.1880 | 0.3533 | 0.3217 | 0.0269 | 0.8129 | 0.1749 |
| 66357600 | checkpoints_stage2_final, step 2000 | 200 | 0.0824 | 0.2516 | 0.2087 | 0.0098 | 0.7928 | 0.1219 |
| 66415007 | checkpoints_stage2_final, step 2500 | 200 | 0.1517 | 0.3022 | 0.2755 | 0.0211 | 0.8009 | 0.1396 |
| 66455039 | checkpoints_stage2_final, step 3500 | 200 | 0.1564 | 0.3156 | 0.2908 | 0.0259 | 0.8049 | 0.1344 |
| 66513179 | checkpoints_stage2_attention_cond, step 3000 | 200 | 0.1674 | 0.3354 | 0.3117 | 0.0158 | 0.8122 | 0.1552 |
| 66515927 | checkpoints_stage2_attention_cond, step 3000 | 1000 | 0.1549 | 0.3315 | 0.3131 | 0.0205 | 0.8133 | 0.1571 |
| 66522737 | checkpoints_stage2_final, step 3500 | 1000 | 0.1587 | 0.3195 | 0.3177 | 0.0226 | 0.8124 | 0.1677 |
| 66559189 | checkpoints_llama3b_v2, step 5000 | 3039 | 0.1450 | 0.2994 | 0.2705 | 0.0230 | 0.8034 | 0.1436 |
| 66559190 | checkpoints_llama3b_attention_cond, step 7000 | 3039 | 0.1292 | 0.2832 | 0.2905 | 0.0170 | 0.8014 | 0.1440 |
| 66627986 | checkpoints_llama3b_biomedbert, step 4500 | 3039 | 0.1169 | 0.2778 | 0.2718 | 0.0223 | 0.7945 | 0.1194 |
| 66701363 | checkpoints_vfilm_fix, step 4500 (best val_loss=0.3995, best T3 collapse) | 3039 | 0.1277 | 0.2827 | 0.2731 | 0.0192 | 0.8011 | 0.1424 |
| 66793050 | checkpoints_t2t3_fix, step 2500 | 3039 | 0.1405 | 0.2997 | NaN† | NaN† | 0.8052 | 0.1469 |

† NaN because `rouge_score`/`pycocoevalcap` were not installed in the
environment at run time — missing-dependency artifact, not a real
zero/failure.

T2/T3 for this old pipeline — small-n qualitative checks only (no
ROUGE-L/BLEU at scale for either task):

| Job | Checkpoint | T1 distinct | T2 (lung) distinct | T3 accuracy |
|---|---|---|---|---|
| 66699593 | checkpoints_vfilm_fix | 5/5 | 1/5 | 7/10 |
| 66793046 | checkpoints_t2t3_fix | 5/5 | 0/5 | 7/10 |

"Distinct" = number of scans (out of 5) whose generated output differs from
the others under the same instruction — a collapse/diversity check, not an
accuracy metric against ground truth.

## 3. Excluded / not real results

- **5-sample smoke test** (job 67003146, `smoke_test.csv`) — run to verify
  the new e2e eval script's plumbing only, against the wrong (old,
  features-path) checkpoint. T1 (n=2) ROUGE-L = 0.0941, T2 (n=1) ROUGE-L =
  0.0429, T3 (n=2) accuracy = 1.00 (2/2). n far too small and
  checkpoint/pipeline mismatch — exclude from any paper-facing results.
- **BERTScore** — `StanfordAIMI/RadBERT` weights load in one log
  (job 66793050) but no BERTScore number is ever printed — appears unused/
  incomplete in that code path.
- **GREEN score** — stubbed as `NaN` everywhere (known upstream bug: GREEN's
  greedy decoding degenerates on this model; see
  `aadp/evaluation/metrics/compute_all.py`).
- **RadGraph-XL F1** — `NaN` for both new e2e checkpoints (stage1, stage2)
  in the `_t1_full_metrics.json` files.
- **CT-CLIP retrieval sweep** (`results/vtcb_sweep_ctclip/`) — directory
  exists but is empty; no metrics saved.

## Sources

- `/scratch/sadeniji/ictc/logs/eval_249604.out` (stage1 e2e eval)
- `/scratch/sadeniji/ictc/logs/eval_stage2_par_321603.out` (stage2 e2e eval, 1-epoch)
- `/scratch/sadeniji/ictc/logs/eval_stage2_par_850923.out` (stage2 e2e eval, current)
- `/scratch/sadeniji/ictc/eval/stage1_e2e_t1_full_metrics.json`
- `/scratch/sadeniji/ictc/eval/stage2_e2e_t1_full_metrics.json`
- `docs/RESULTS_SUMMARY.md` (2026-08-03 snapshot, legacy pipeline)
- `docs/QUALITATIVE_EVAL_V2.md`, `docs/QUALITATIVE_EVAL_T2T3_FIX.md`
