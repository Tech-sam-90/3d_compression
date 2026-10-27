# Results Summary — All Saved Evaluation Metrics

Consolidated from every metrics-bearing file found in the project as of
2026-08-03: `docs/*.md`, `/scratch/sadeniji/logs/eval_checkpoint_*.{out,err}`,
`/scratch/sadeniji/logs/qual_eval_v2*.{out,err}`,
`/scratch/sadeniji/logs/qualitative_eval_*.{out,err}`, and
`/scratch/sadeniji/ictc/eval/smoke_test.csv`. Intended as a single reference
for paper-writing — every number below is traceable to a specific job ID and
checkpoint path.

## Read this first

- **All large-sample metrics** (BLEU-4/METEOR/ROUGE-L/CIDEr/RaTEScore/
  RadGraph-XL-F1) come from `scripts/evaluate_checkpoint.py`, which evaluates
  **T1 only** (fixed instruction: "Generate a radiology report for this CT
  scan.") on the **old `features=` path** (pre-extracted CT-CLIP features,
  frozen ViT) — i.e. the pre-existing `ictc_checkpoints*` checkpoint family,
  **not** the new end-to-end unfrozen-ViT model (`ictc_checkpoints_e2e_stage1`,
  job 66999932, still training as of this writing).
- **T2 and T3 have never had a full-sample metric run.** Only small-n (5–10
  sample) qualitative/distinctness checks exist for them — no ROUGE-L/BLEU at
  scale for either task.
- The new e2e/raw-volume pipeline has **no real evaluation numbers yet** —
  only a 5-sample smoke test (see bottom section), run against the wrong
  (old, features-path) checkpoint purely to test the eval script's plumbing.

## T1 (full report) — large-sample NLP/clinical metrics

Source: `/scratch/sadeniji/logs/eval_checkpoint_*.out` (`evaluate_checkpoint.py`
JSON output blocks), cross-checked against `docs/DIAGNOSTIC_REPORT.md`'s
summary tables (numbers match).

| Job | Checkpoint (path, step, val_loss) | n | BLEU-4 | METEOR | ROUGE-L | CIDEr | RaTEScore | RadGraph-XL F1 |
|---|---|---|---|---|---|---|---|---|
| 66272201 | `ictc_checkpoints/checkpoint_best.pt`, step=3500, val_loss=0.8575 | 200 | 0.2037 | 0.3579 | 0.3345 | 0.0395 | 0.8158 | 0.1868 |
| 66300376 | `ictc_checkpoints/checkpoint_best.pt`, step=6000, val_loss=0.9094 | 200 | 0.1599 | 0.3217 | 0.2641 | 0.0145 | 0.8115 | 0.1569 |
| 66304396 | `ictc_checkpoints_llama3b/checkpoint_best.pt`, step=4500, val_loss=0.5528 | 200 | 0.1325 | 0.2928 | 0.2829 | 0.0237 | 0.8029 | 0.1578 |
| 66320539 | `ictc_checkpoints/checkpoint_best.pt`, step=5500, val_loss=0.9081 | 200 | 0.1880 | 0.3533 | 0.3217 | 0.0269 | 0.8129 | 0.1749 |
| 66357600 | `ictc_checkpoints_stage2_final/checkpoint_best.pt`, step=2000, val_loss=1.7740 | 200 | 0.0824 | 0.2516 | 0.2087 | 0.0098 | 0.7928 | 0.1219 |
| 66415007 | `ictc_checkpoints_stage2_final/checkpoint_best.pt`, step=2500, val_loss=0.9517 | 200 | 0.1517 | 0.3022 | 0.2755 | 0.0211 | 0.8009 | 0.1396 |
| 66455039 | `ictc_checkpoints_stage2_final/checkpoint_best.pt`, step=3500, val_loss=0.9466 | 200 | 0.1564 | 0.3156 | 0.2908 | 0.0259 | 0.8049 | 0.1344 |
| 66513179 | `ictc_checkpoints_stage2_attention_cond/checkpoint_best.pt`, step=3000, val_loss=0.8795 | 200 | 0.1674 | 0.3354 | 0.3117 | 0.0158 | 0.8122 | 0.1552 |
| 66515927 | `ictc_checkpoints_stage2_attention_cond/checkpoint_best.pt`, step=3000, val_loss=0.8795 | 1000 | 0.1549 | 0.3315 | 0.3131 | 0.0205 | 0.8133 | 0.1571 |
| 66522737 | `ictc_checkpoints_stage2_final/checkpoint_best.pt`, step=3500, val_loss=0.9466 | 1000 | 0.1587 | 0.3195 | 0.3177 | 0.0226 | 0.8124 | 0.1677 |
| 66559189 | `ictc_checkpoints_llama3b_v2/checkpoint_best.pt`, step=5000, val_loss=0.4225 | 3039 | 0.1450 | 0.2994 | 0.2705 | 0.0230 | 0.8034 | 0.1436 |
| 66559190 | `ictc_checkpoints_llama3b_attention_cond/checkpoint_best.pt`, step=7000, val_loss=0.4413 | 3039 | 0.1292 | 0.2832 | 0.2905 | 0.0170 | 0.8014 | 0.1440 |
| 66627986 | `ictc_checkpoints_llama3b_biomedbert/checkpoint_best.pt`, step=4500, val_loss=0.4201 | 3039 | 0.1169 | 0.2778 | 0.2718 | 0.0223 | 0.7945 | 0.1194 |
| 66701363 | `ictc_checkpoints_vfilm_fix/checkpoint_best.pt`, step=4500, val_loss=0.3995 | 3039 | 0.1277 | 0.2827 | 0.2731 | 0.0192 | 0.8011 | 0.1424 |
| 66793050 | `ictc_checkpoints_t2t3_fix/checkpoint_best.pt`, step=2500, val_loss=0.3955 | 3039 | 0.1405 | 0.2997 | NaN\* | NaN\* | 0.8052 | 0.1469 |

\* Job 66793050's ROUGE-L and CIDEr are `NaN` because `rouge_score` and
`pycocoevalcap` were not installed in the environment at the time this job
ran (`ROUGE-L scoring failed, returning NaN: No module named 'rouge_score'`) —
this is a missing-dependency artifact, not a real zero/failure of the model.

Best n=3039 checkpoint with complete metrics: **job 66701363 (V-FiLM fix,
val_loss=0.3995)**. Per `docs/DIAGNOSTIC_REPORT.md`'s own analysis, this
checkpoint has the best val_loss and best Test-3 collapse rate (0.0%) of the
session but is *not* uniformly best on the n-gram/clinical metrics — see that
doc's "best val_loss ≠ best generation quality" discussion for the full
caveat (BLEU-4/METEOR/RaTEScore/RadGraph-XL-F1 all slightly regress vs. the
pre-fix checkpoint 66559189, while ROUGE-L slightly improves).

## T2 (entity-scoped) and T3 (binary) — small-n qualitative checks only

Source: `docs/QUALITATIVE_EVAL_V2.md`, `docs/QUALITATIVE_EVAL_T2T3_FIX.md`,
raw logs `qual_eval_v2_66699593.out`, `qual_eval_v2_t2t3_66793046.out`.

| Job | Checkpoint | T1 distinct | T2 (lung) distinct | T3 binary accuracy |
|---|---|---|---|---|
| 66699593 | `ictc_checkpoints_vfilm_fix/checkpoint_best.pt` (step 4500, val_loss=0.3995) | 5/5 | 1/5 | 7/10 |
| 66793046 | `ictc_checkpoints_t2t3_fix/checkpoint_best.pt` (step 2500, val_loss=0.3955) | 5/5 | 0/5 | 7/10 |

"Distinct" = number of scans (out of 5) whose generated output differs from
the others under the same instruction — a collapse/diversity check, not an
accuracy metric against ground truth. No ROUGE-L/BLEU was ever computed for
T2 or T3 at any sample size — these are the only numbers that exist for
those two tasks.

## This session's e2e-pipeline smoke test (not a real result — do not cite)

`/scratch/sadeniji/ictc/eval/smoke_test.csv` (SLURM job 67003146),
5 samples total, run through the new `qualitative_eval_v3.py` /
`volume=` raw-CT pipeline against the **old** `checkpoint_best.pt` from
`ictc_checkpoints_vfilm_fix` (a features-path checkpoint, loaded via the
piecewise-format fallback) — purpose was solely to verify the new eval
script's plumbing (checkpoint loading, dtype handling, CSV output) works
end-to-end, not to measure model quality:

- T1 (n=2): mean ROUGE-L = 0.0941
- T2 (n=1): mean entity-scoped ROUGE-L = 0.0429
- T3 (n=2): binary accuracy = 1.00 (2/2)

n is far too small and the checkpoint doesn't match the pipeline it's being
run through — exclude from any paper-facing results.

## Metrics not found anywhere in the project

- **BERTScore** — job 66793050's log shows `StanfordAIMI/RadBERT` weights
  loading (likely in preparation for a BERTScore computation) but the log
  ends before any BERTScore number is printed; appears unused/incomplete in
  that code path.
- **GREEN score** — explicitly stubbed as `NaN` everywhere it appears
  ("known upstream generation bug: green_score's greedy decoding degenerates
  on this model" — see `aadp/evaluation/metrics/compute_all.py`).
- **Any metric for the new end-to-end unfrozen-ViT checkpoints** — stage1
  (job 66999932) is still training; no `evaluate_checkpoint.py`-equivalent
  full run has been done against a real e2e checkpoint yet. `slurm_eval.sh`
  is staged and ready to run once stage1 produces
  `checkpoint_latest_consolidated.pt`.
