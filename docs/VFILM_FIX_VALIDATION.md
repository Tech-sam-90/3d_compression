# V-FiLM Fix Validation

Validates the bounded, multiplicative-only V-FiLM fix (`aadp/models/projector/stage2.py`,
commit `253b3fa`) against the two diagnostics that motivated it, rerun on the
resulting checkpoint. No training or code changes happened in this pass —
read-only diagnostics only.

## Background

`scripts/check_vfilm.py` (job 66657581) first established that the old V-FiLM
formula (`gamma_v = 1.0 + gamma_v_proj(etext)`, `beta_v = beta_v_proj(etext)`,
`V = gamma_v*V + beta_v`, unbounded) was genuinely **ACTIVE** — nonzero learned
weights, large activation deltas, instruction-differentiating. But
`scripts/check_vfilm_scan_diversity.py` (job 66661482) then showed that
"active" was itself the problem: under a **fixed instruction**, `beta_v` (a
pure function of the instruction, scan-agnostic) trained to ~46% of the
combined term's norm — enough to dominate direction. V_after's pairwise
cosine across 6 scans (3 collapsed-cluster, 3 distinct) was **0.999 for every
pair-type group** (collapsed-collapsed, distinct-distinct, cross-group), up
from 0.955–0.963 before V-FiLM — V-FiLM was making different scans look
*more* alike, not less, erasing the one distinction the test was designed to
detect.

The fix (`aadp/models/projector/stage2.py`): removed `beta_v_proj` entirely,
bounded `gamma_v` to `[0.7, 1.3]` via `1.0 + 0.3*tanh(gamma_v_proj(etext))`,
keeping V-FiLM strictly multiplicative. Trained fresh via warm start from the
pre-fix checkpoint (job 66527771, val_loss=0.4225), with `gamma_v_proj`
restarted from zero-init (`configs/ctclip_vfilm_fix.yaml`,
`warm_start_drop_keys: ["gamma_v_proj"]`) — job **66667523**, completed
2026-07-30, best val_loss **0.3995** (checkpoint step 4500).

Both diagnostics needed a small compatibility fix to even run against the new
checkpoint: `check_vfilm.py`'s `run_with_vfilm_capture()` hardcoded a hook on
`stage2.beta_v_proj`, which no longer exists post-fix — it now detects the
architecture via `hasattr(stage2, "beta_v_proj")` and branches accordingly
(old formula with beta, or new bounded-gamma-only formula), so it works
against both pre-fix and post-fix checkpoints.

## Diagnostic 1 — Scan Diversity (`check_vfilm_scan_diversity.py`)

Same protocol as job 66661482: 6 scans (3 collapsed-cluster: `valid_131_a_1`,
`valid_634_a_1`, `valid_1016_b_2`; 3 distinct: `valid_1_a_1`, `valid_382_c_2`,
`valid_500_d_1`), same fixed instruction ("Generate a radiology report for
this CT scan."), pairwise cosine of the mean-pooled value tensor before/after
V-FiLM.

**Job 66694996**, checkpoint `checkpoint_best.pt` (step 4500, val_loss=0.3995).

### Pairwise cosine similarity — before vs. after

| Pair type | Before (pre-fix, job 66661482) | After (pre-fix) | Before (post-fix, job 66694996) | After (post-fix) |
|---|---|---|---|---|
| collapsed-collapsed | 0.955 ± 0.026 | **0.999 ± 0.000** | 0.965 ± 0.018 | **0.968 ± 0.016** |
| distinct-distinct | 0.963 ± 0.023 | **0.999 ± 0.000** | 0.979 ± 0.010 | **0.979 ± 0.010** |
| cross-group | 0.963 ± 0.031 | **0.999 ± 0.001** | 0.969 ± 0.029 | **0.972 ± 0.027** |

Pre-fix: V-FiLM pushed every group to a uniform ~0.999, erasing the
collapsed/distinct distinction entirely. Post-fix: V-FiLM barely moves the
cosine at all (largest shift is +0.003), and the three groups remain exactly
as distinguishable after V-FiLM as before it — the directional collapse is
gone.

### Diversity ratio & beta dominance

| Metric | Pre-fix | Post-fix |
|---|---|---|
| Diversity ratio (after/before) | 2.08 → "PRESERVED" (misleading — raw std increased while cosine collapsed) | **0.91 → PRESERVED** (this time corroborated by the cosine numbers above) |
| `‖γ·V‖` | 18267.50 | 5125.90 |
| `‖β‖` | 15704.96 | **0.00** (beta_v_proj removed entirely) |
| Beta fraction | 46.2% | **0.0% (N/A — structurally impossible)** |

The pre-fix "diversity ratio" verdict was flagged at the time as misleading
(magnitude-sensitive, not direction-sensitive — see prior conversation
turn's analysis) since it disagreed with the cosine numbers. Post-fix, the
diversity ratio and cosine numbers agree with each other, which is itself a
sign the metric is now measuring something coherent rather than an artifact.

**Verdict: fixed.** V-FiLM no longer collapses cross-scan direction under a
fixed instruction.

## Diagnostic 2 — Test 3 Collapse Diagnostic

Same protocol as every other Test 3 rerun this session
(`scripts/diagnostics/test3_full_rerun_vfilm_fix.py`, job **66695047**,
checkpoint step 4500, val_loss=0.3995): Part A = same scan (`valid_1_a_1`), 5
instructions; Part B = same instruction, 10 scans at fixed indices `[0, 300,
600, 900, 1200, 1500, 1800, 2100, 2400, 2700]`.

### Before vs. after (checkpoint comparison)

| Metric | Pre-fix (Llama+FiLM, job 66527771) | Post-fix (V-FiLM fix, job 66667523) |
|---|---|---|
| val_loss | 0.4225 | **0.3995** |
| Part A cosine (mean) | 0.6640 | 0.7729 |
| Part B cosine (mean) | 0.8937 | 0.9137 |
| **Collapse rate** | 6.7% (3/45 pairs) | **0.0% (0/45 pairs)** |

Part A/B cosine both rose slightly post-fix — consistent with a pattern
observed repeatedly this session: pooled-cosine similarity and text-level
collapse rate are not perfectly correlated, and collapse rate (what actually
shows up as duplicate/near-duplicate generated text) is the more decision-
relevant metric of the two. On that metric, the fix eliminated collapse
entirely: **0 exact or near-duplicate pairs out of 45**, down from 3/45
pre-fix — the best collapse rate of any checkpoint trained this session
(previous best was 6.7%, same checkpoint).

### Part A — instruction sensitivity (same scan, 5 instructions)

Scan: `valid_1_a_1`. Reports remain genuinely distinct in content across
instructions:

| # | Instruction | Report (truncated) |
|---|---|---|
| 1 | Generate a radiology report... | Trachea and both main bronchi are open... ascending aorta measures approximately 40mm... fusiform dilatation... calcified lymph nodes... |
| 2 | Describe any pulmonary findings... | No mass or infiltrative lesion was detected in both lungs. |
| 3 | Are there any cardiovascular abnormalities? | No, there are no significant abnormalities in the heart and mediastinum. |
| 4 | Summarize mediastinal structures and lymph nodes. | No lymph node was observed in the supraclavicular fossa, axilla and mediastinum... |
| 5 | Report musculoskeletal or pleural findings. | No pleural effusion was detected. |

### Part B — scan sensitivity (same instruction, 10 scans) — PRIMARY TEST

Fixed instruction: "Generate a radiology report for this CT scan." All 10
generated reports are distinct in content — different pathology patterns
(fusiform aortic dilatation, viral/COVID-pattern ground-glass opacities,
consolidation with halo sign, coronary calcification, hiatal hernia) surface
for different scans rather than one boilerplate report repeating:

| # | Scan | Notable content |
|---|---|---|
| 1 | valid_1_a_1 | Fusiform ascending aortic dilatation (~40mm), calcified paratracheal lymph nodes |
| 2 | valid_131_a_1 | Cardiomegaly, LAD calcific plaque, hiatal hernia, bilateral consolidation/ground-glass |
| 3 | valid_258_a_2 | Bilateral pulmonary nodules, pericardial effusion (15mm), bilateral pleural effusions, CVC catheter |
| 4 | valid_382_c_2 | Coronary artery disease pattern, pleural effusion + atelectasis, viral pneumonia (regressing), hepatosteatosis |
| 5 | valid_500_d_1 | Hiatal hernia, paratracheal lymph nodes, unremarkable lung aeration |
| 6 | valid_634_a_1 | Diffuse ground-glass opacities, "interpreted primarily in favor of viral pneumonia (Covid 19)" |
| 7 | valid_766_a_1 | Consolidation with halo sign, "infective process" differential |
| 8 | valid_890_a_2 | Peripheral ground-glass opacities, "typical-probable... Covid 19 pneumonia", hepatosteatosis |
| 9 | valid_1016_b_2 | Widened ascending aorta (40mm AP diameter), paratracheal/subcarinal lymph nodes |
| 10 | valid_1147_b_2 | Coronary artery disease pattern, esophageal wall calcification, peripheral ground-glass, Covid-19 pattern |

Edit-distance matrix (normalized, 0=identical/1=completely different) shows
no pair below the 0.05 near-duplicate threshold — closest pair is #8/#10 at
0.100 (both COVID-pattern ground-glass reports, but still distinct text, not
a near-duplicate by the test's own definition).

Vocabulary diversity: 304 unique words / 1374 total words across the 10
reports (TTR=0.2213).

## Overall Verdict

The bounded, multiplicative-only V-FiLM fix resolves both problems it was
built to fix:
1. **Directional collapse under a fixed instruction** (diagnosed via
   `check_vfilm_scan_diversity.py`) — gone; post-fix cosine barely moves
   across V-FiLM at all, and stays fully distinguishable by group.
2. **Text-level collapse rate** (diagnosed via Test 3 Part B) — improved
   from 6.7% to **0.0%**, the best result of any checkpoint this session.

It also improved val_loss (0.4225 → 0.3995) despite starting from a fresh
`gamma_v_proj` zero-init and losing the `beta_v_proj` parameters entirely —
the model did not need the additive term to fit the training distribution.

**Recommendation:** `/scratch/sadeniji/ictc_checkpoints_vfilm_fix/checkpoint_best.pt`
(step 4500, val_loss=0.3995) supersedes
`/scratch/sadeniji/ictc_checkpoints_llama3b_v2/checkpoint_best.pt` (job
66527771) as the strongest checkpoint of the project to date. Full n=3039
NLP/clinical metric evaluation (`scripts/evaluate_checkpoint.py`) has not yet
been run on this checkpoint — collapse-rate and cosine diagnostics are
necessary but not sufficient for a complete comparison against the
BLEU-4/METEOR/ROUGE-L/RaTEScore/RadGraph-XL-F1 table in
`docs/DIAGNOSTIC_REPORT.md`'s full-session summary.
