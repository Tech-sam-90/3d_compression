# ICTC Diagnostic Report 3

Read-only diagnostic against pre-extracted CT-CLIP `.pt` feature files.
No model/training code or checkpoints were modified.

---

## CT-CLIP Feature Cosine Check — Collapsed vs Distinct Scans

**Goal:** distinguish whether the 13.3% Test 3 Part B text-collapse (see
`docs/DIAGNOSTIC_REPORT.md`) traces to an upstream CT-CLIP feature
bottleneck (Hypothesis A) or an aggregator failure to preserve
discriminative signal that *is* present upstream (Hypothesis B).

**Feature files:** `/project/def-uanazodo-ab/sadeniji/ctrate_features/valid/<stem>.pt`, naming convention `<VolumeName without .nii.gz>.pt` (confirmed via `find`). Raw tensor shape per scan: `(24, 24, 24, 512)` = `(D=24 depth slices, 24×24=576 spatial tokens/slice, C=512 channels)`. Global descriptor per scan: mean-pooled over all `D×576` positions → `(512,)`.

### Pairwise cosine similarity (mean-pooled global CT-CLIP descriptor)

**Collapsed cluster** (5 scans, 10 pairs):

| Pair | Cosine |
|---|---|
| valid_131_a_1 vs valid_258_a_2 | 0.832400 |
| valid_131_a_1 vs valid_634_a_1 | 0.976007 |
| valid_131_a_1 vs valid_1016_b_2 | 0.943179 |
| valid_131_a_1 vs valid_1147_b_2 | 0.785372 |
| valid_258_a_2 vs valid_634_a_1 | 0.904435 |
| valid_258_a_2 vs valid_1016_b_2 | 0.837355 |
| valid_258_a_2 vs valid_1147_b_2 | 0.967123 |
| valid_634_a_1 vs valid_1016_b_2 | 0.923274 |
| valid_634_a_1 vs valid_1147_b_2 | 0.880266 |
| valid_1016_b_2 vs valid_1147_b_2 | 0.776379 |

**Distinct group** (5 scans, 10 pairs):

| Pair | Cosine |
|---|---|
| valid_1_a_1 vs valid_382_c_2 | 0.959516 |
| valid_1_a_1 vs valid_500_d_1 | 0.978274 |
| valid_1_a_1 vs valid_890_a_2 | 0.860493 |
| valid_1_a_1 vs valid_766_a_1 | 0.874180 |
| valid_382_c_2 vs valid_500_d_1 | 0.925971 |
| valid_382_c_2 vs valid_890_a_2 | 0.897137 |
| valid_382_c_2 vs valid_766_a_1 | 0.953144 |
| valid_500_d_1 vs valid_890_a_2 | 0.905489 |
| valid_500_d_1 vs valid_766_a_1 | 0.864226 |
| valid_890_a_2 vs valid_766_a_1 | 0.920472 |

**Cross-group** (collapsed × distinct, 25 pairs): full detail in the run log; range 0.7854-0.9920.

### Summary

```
Collapsed cluster   mean cosine: 0.882579  (min 0.776379, max 0.976007)
Distinct group      mean cosine: 0.913890  (min 0.860493, max 0.978274)
Cross-group         mean cosine: 0.914029  (min 0.785422, max 0.992004)

Gap (collapsed - distinct): -0.031311
```

### Verdict: **HYPOTHESIS B — Aggregator failure**

The gap runs in the *opposite* direction from what Hypothesis A predicts. Hypothesis A required `collapsed mean > distinct mean + 0.05`; instead, the collapsed cluster's raw CT-CLIP features are **slightly less** similar to each other (0.8826) than the distinct group's are to each other (0.9139) — a gap of -0.031, not a positive gap at all. The collapsed cluster isn't even unusually self-similar relative to *outside* its own group: cross-group mean (0.9140) is essentially identical to within-distinct-group mean (0.9139) and higher than within-collapsed-group mean.

In other words, the raw CT-CLIP features for the 5 "collapsed" scans carry at least as much discriminative signal as the features for the 5 "distinct" scans — there was no upstream ceiling preventing the aggregator from telling these scans apart. Combined with `docs/DIAGNOSTIC2_REPORT.md`'s finding that the ground-truth reference reports for these scans are also clinically diverse (mean edit distance 0.72, comparable to the distinct group), this closes off both plausible "it's actually fine, the scans/reports/features really are similar" explanations. **The aggregator (top-K sparse + cosine visual cross-attention, unchanged from the FiLM pipeline in this attention-conditioned ablation) is discarding real discriminative signal that is present in its input.**

**Next step per the decision framework:** train the V-FiLM fix (value-conditioning via `gamma_v_proj`/`beta_v_proj`, already implemented and zero-initialized in `aadp/models/projector/stage2.py`, not yet exercised in the attention-conditioned ablation — the attention-conditioned aggregator in `aadp/ablations/attention_conditioned_ctclip_stage2.py` currently has no value-conditioning mechanism at all, only query-side conditioning via `cond_cross_attn`). Note this is a code change, not something this read-only diagnostic performed.
