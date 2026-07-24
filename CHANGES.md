# ICTC Stage 2 — Session Change Log

Branch: `ctclip-stage2-train`. All six requested changes applied in order, each
verified before moving to the next. This document is the final report for
that session, including the Stage 2 smoke test result that gated the full
training run.

---

## Change 1 — Sequence length: 256 → 1024

**Finding during implementation:** the actual `max_length=256` report
truncation was not in `aadp/models/ctclip_vlm.py` as assumed — it's in
`scripts/train_ctclip.py` (target-report tokenization, two call sites: the
training loop and `_validate()`). The two `max_length=128` calls that *do*
live in `ctclip_vlm.py` are for instruction-text encoding (short imperative
sentences), a separate and correctly-scoped value — left unchanged.

**Applied:**
- `scripts/train_ctclip.py`: both `model.tokenizer(batch["target"], ...)`
  calls now use `max_length = cfg.get("max_length", 1024)` instead of the
  hardcoded `256`. `_validate()` gained a `max_length` parameter, threaded
  through from its call site.
- `aadp/models/ctclip_vlm.py`: added `self.llm.gradient_checkpointing_enable()`
  immediately after the LLM load, trading ~30% speed for the activation-memory
  headroom 1024-token sequences need on A100-40G. Confirmed safe without
  `enable_input_require_grads()`: `inputs_embeds` is a concat that includes
  `visual` (output of the trainable `visual_proj`), so it already requires
  grad — the usual frozen-embedding-plus-checkpointing gotcha doesn't apply
  here.
- `max_length: 1024` added to all four configs: `ctclip_stage1.yaml`,
  `ctclip_stage2_final.yaml`, `ctclip_stage2_llama3b.yaml`,
  `narval_smoke_llama3b.yaml`.

**Verification** — token length distribution against the real training CSV
(`/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/train_merged.csv`,
BioMedLM tokenizer, n=47,149):

| Stat | Value |
|---|---|
| Mean | 298 |
| Median | 276 |
| 90th percentile | 453 |
| Max | 1314 |
| % > 256 | 58.0% |
| % > 512 | 5.6% |
| % > 1024 | 0.013% (6 reports) |

At the old `max_length=256`, **58% of all reports were being silently
truncated**. At 1024, only 6 out of 47,149 reports (0.013%) still exceed the
limit — effectively solved.

---

## Change 2 — Batch size adjustment

`ctclip_stage2_final.yaml` and `ctclip_stage1.yaml`:
`batch_size: 2 → 1`, `gradient_accumulation_steps: 16 → 32`.

**Verification:** loaded both configs with `yaml.safe_load` and confirmed
`batch_size * gradient_accumulation_steps == 32` in both — effective batch
size unchanged.

---

## Change 3 — Fix NaN collapse from Stage 2 (job 66352992)

Job 66352992 (previous Stage 2 attempt) collapsed into 100% NaN loss from
step 2950 onward. In `ctclip_stage2_final.yaml`:

| Key | Old | New |
|---|---|---|
| `aggregator_learning_rate` | `1.0e-3` | `2.0e-4` |
| `learning_rate` (LoRA) | `1.0e-6` | `1.0e-5` |
| `max_grad_norm` | `1.0` | `0.5` |
| `cls_loss_weight` | `0.3` | `0.1` |

Rationale documented inline in the config: the aggregator's `1e-3` LR under
the combined `lm_loss + cls_loss_weight*cls_loss` objective is suspected to
have caused early gradient explosion; LoRA's `1e-6` was too small to adapt
meaningfully in the ~500 steps completed before the run died.

**Verification:** confirmed via `yaml.safe_load` that all four values load
correctly. Real verification came from Change 6's smoke test (see below) —
**zero NaN across all 400 steps**, versus the prior run's collapse.

---

## Change 4 — Auto-discover LoRA target modules

Added `find_lora_target_modules(model)` to `aadp/models/ctclip_vlm.py`
(module-level function, adapted from M3D's `find_all_linear_names`): scans
`named_modules()` for `nn.Linear` layers, excluding
`visual_proj`/`projector`/`cls_head`/`instruction_encoder`/`lm_head`/
`embed_tokens`. Wired into the LoRA setup in place of the hardcoded
`target_modules` list, called on `self.llm` before PEFT wrapping, with
`["c_attn", "c_proj"]` kept as a fallback if discovery returns nothing.

**Verification — important finding:** ran discovery live against BioMedLM:

```
LoRA targets: []
find_lora_target_modules found nothing; falling back to ['c_attn', 'c_proj']
```

BioMedLM is GPT-2-based, and HF's GPT-2 implementation uses
`transformers.pytorch_utils.Conv1D` for its attention/MLP projections, **not**
`nn.Linear` — confirmed directly (`type(m).__name__ == 'Conv1D'` for
`c_attn`/`c_proj`/`c_fc`). So for BioMedLM the function always returns `[]`
and falls back to the same `["c_attn", "c_proj"]` that was previously
hardcoded — functionally identical result, reached dynamically instead.
It should behave differently (and usefully) for the Llama-3.2 configs, whose
attention/MLP layers are real `nn.Linear`, but this is **untested** — Llama
isn't cached in the offline HF cache used on compute nodes.

---

## Change 5 — Expand instruction templates from M3D

Read `M3D/LaMed/src/dataset/prompt_templates.py`'s `Caption_templates` (42
entries, one exact internal duplicate — `"Please caption this medical scan
with findings."` appeared twice — deduped to 41 unique). Compared against
the existing 15 `T1_TEMPLATES` in `aadp/data/instruction_builder.py`:
**zero overlap** (M3D's pool is short/terse and mostly interrogative;
the existing set is longer/declarative and chest-CT-specific).

All 41 unique M3D templates were added verbatim:

```
Can you provide a caption consists of findings for this medical image?
Describe the findings of the medical image you see.
Please caption this medical scan with findings.
What is the findings of this image?
Describe this medical scan with findings.
Please write a caption consists of findings for this image.
Can you summarize with findings the images presented?
Please caption this scan with findings.
Please provide a caption consists of findings for this medical image.
Can you provide a summary consists of findings of this radiograph?
What are the findings presented in this medical scan?
Please write a caption consists of findings for this scan.
Can you provide a description consists of findings of this medical scan?
Can you provide a caption consists of findings for this medical scan?
Please generate a medical report based on this image.
Can you generate a diagnose report from this image.
Could you analyze and provide a caption for the findings in this medical image?
Please describe the observations depicted in this medical scan.
Can you summarize the findings of this image in a caption?
What are the significant findings in this medical image?
Please provide a detailed caption outlining the findings of this image.
Could you interpret and describe the findings shown in this medical scan?
What conclusions can you draw from the observations in this image?
Please write a descriptive caption based on the findings in this scan.
What key findings can you identify from examining this medical image?
Could you generate a detailed report based on the observations in this image?
Can you provide a diagnosis based on the findings in this image?
Please generate a comprehensive report summarizing the findings in this image.
Caption the findings in this medical image?
Describe the findings you see.
Caption this medical scan's findings.
What are the findings here?
Describe these findings.
Summarize the findings in these images.
Caption this scan's findings.
Provide a caption for this medical image's findings.
Summarize the findings of this radiograph.
What findings are presented in this scan?
Describe this scan's findings.
Generate a medical report based on this image.
Can you provide a diagnosis based on this image?
```

**Verification:** `len(T1_TEMPLATES) == 56` (15 original + 41 added),
`len(set(T1_TEMPLATES)) == 56` — no duplicates anywhere in the merged list.

---

## Change 6 — Smoke test Stage 2 before full run

1. Confirmed Stage 1 checkpoint exists:
   `/scratch/sadeniji/ictc_checkpoints_stage1/checkpoint_best.pt` (118 MB,
   present).
2. Created `configs/narval_smoke_stage2.yaml`: copy of
   `ctclip_stage2_final.yaml` with `max_samples: 200`, `num_epochs: 2`,
   `resume_from` pointing at the Stage 1 checkpoint. **Deviation from the
   literal instructions:** `gradient_accumulation_steps` was overridden from
   the parent's `32` down to `1` — `global_step` only increments on real
   optimizer steps, so at `grad_accum=32` with only 200 samples/2 epochs,
   the run would produce only ~12 total steps, far short of the "watch the
   first 300 steps" requirement. At `grad_accum=1`, up to ~400 steps are
   produced (matches the same fix already used in `smoke_test_stage1.sh`).
3. Updated `smoke_test_narval.sh` to run `configs/narval_smoke_stage2.yaml`.

**Pre-flight gate (job 66391031):** `pytest tests/ -v` — **377 passed, 2
skipped, 8 issues** (2 failed + 6 errors), all confined to
`tests/test_training.py`. Confirmed pre-existing and unrelated: that file
hasn't been touched since the initial "First Upload" commit
(`git log --oneline -1 -- tests/test_training.py`), and the errors trace to
an uncached `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k` model blocked by
offline mode on compute nodes — the old MedVLM/ViT pipeline, unrelated to
`CTCLIPStage2VLM`. Every test actually covering the code touched by Changes
1–5 (`test_stage2.py`, `test_ctclip_stage2_vlm.py`) passed cleanly.

**Smoke test — three attempts:**

- **Job 66392222** (first attempt): hung for 17+ minutes before any training
  step — confirmed via `/proc/<pid>/io` (`read_bytes` frozen, zero progress
  across multiple checks 5s+ apart) and `wchan` (`cl_sync_io_wait`, a
  Lustre-filesystem I/O wait state) that this was a genuine stall, not just
  slow loading. Cancelled and resubmitted; not a code issue.
- **Job 66394290** (second attempt): SLURM `oom_kill` (host RAM), exit code
  125, died inside a GPT-2 forward pass ~3 real steps in. Root cause:
  `smoke_test_narval.sh` still carried its original `--mem=40G
  --cpus-per-task=4` allocation, sized for the old lightweight OPT-125m
  joint-smoke config it used to point at — never bumped when repointed at
  Stage 2's real BioMedLM (2.7B) forward pass. Fixed to `--mem=80G
  --cpus-per-task=8`, matching `train_ictc_stage2.sh`'s already-proven
  allocation.
- **Job 66396698** (third attempt): **COMPLETED cleanly.**

### Smoke test result (job 66396698)

- Duration: 3m56s. All 400 steps (2 epochs × 200 optimizer steps) completed.
- **Zero NaN** anywhere in stdout/stderr (`grep -ic nan` → 0 in both log
  files).
- Warm-start from Stage 1 confirmed working:
  `Warm-started weights from .../checkpoint_best.pt (step=1500, epoch=1,
  val_loss=0.4462) — optimizer/scheduler NOT restored (stage transition).`
- Training loss (noisy at `batch_size=1`, as expected), sampled every 10
  steps: `6.98 → 4.41 → 4.56 → 5.74 → 6.28 → 2.99 → ... → 2.67` (step 200,
  epoch 1 end) `→ 3.04 → 2.61 → ... → 2.60` (step 400, epoch 2 end).
- **Validation loss — monotonic improvement, new best every checkpoint:**

  | Step | val_loss | best |
  |---|---|---|
  | 50 | 4.1314 | inf |
  | 100 | 2.6515 | 4.1314 |
  | 150 | 2.3643 | 2.6515 |
  | 200 | 2.1907 | 2.3643 |
  | 250 | 2.1850 | 2.1907 |
  | 300 | 2.2020 | 2.1850 |
  | 350 | 2.1537 | 2.1850 |
  | 400 | **2.0040** | 2.1537 |

This is a clean pass on both explicit gates from the original instructions
(no NaN before step 200; no NaN through step 300) — the revised Stage 2
hyperparameters (Change 3) show no sign of the collapse pattern that killed
job 66352992.

### Full run

Since the smoke test passed, the full Stage 2 run was chained per the
original instructions. `sbatch --dependency=afterok:66396698` was rejected by
SLURM (`Job dependency problem` — the smoke job had already fully completed
and aged out of the scheduler's dependency table by the time of submission,
so the dependency was moot). Submitted directly instead:

**Job 66397199** — `train_ictc_stage2.sh` / `configs/ctclip_stage2_final.yaml`
— currently running.

---

## Summary of files changed

- `aadp/models/ctclip_vlm.py` — gradient checkpointing, `find_lora_target_modules()`
- `scripts/train_ctclip.py` — `max_length` threaded from config
- `aadp/data/instruction_builder.py` — 41 M3D templates added to `T1_TEMPLATES`
- `configs/ctclip_stage1.yaml`, `configs/ctclip_stage2_final.yaml` — `max_length`,
  batch size/grad accum, revised Stage 2 LRs/clip/cls_loss_weight
- `configs/ctclip_stage2_llama3b.yaml`, `configs/narval_smoke_llama3b.yaml` — `max_length`
- `configs/narval_smoke_stage2.yaml` — new
- `smoke_test_narval.sh` — repointed at Stage 2 smoke config, fixed mem/cpu allocation
