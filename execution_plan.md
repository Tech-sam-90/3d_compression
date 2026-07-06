# A-ADP: Full Project Execution Plan
## Geometry- and Instruction-Aware Token Compression for 3D Medical VLMs

---

## Overview

This plan is structured as sequential **phases**, each broken into **tasks** that Claude Code can implement one by one. Each task specifies what to build, what files to create, and what the acceptance criteria are.

---

## PHASE 0 — Project Scaffold & Environment

### Task 0.1 — Repository Structure

**Layout convention (package-only):** all importable library code lives under the
`aadp/` package; `scripts/`, `tests/`, and `configs/` sit at the repository root.
The tree below reflects the current state of the repository.

```
3d-compression/
├── aadp/                              # importable package (all library code)
│   ├── __init__.py
│   ├── data/                          # dataset loaders, preprocessing, instructions
│   │   ├── preprocessing.py           # HU windowing / normalise / pad-crop depth
│   │   ├── ctrate_dataset.py          # CT-RATE volumes + reports + 18 labels (streaming)
│   │   ├── radgenome_dataset.py       # slice-level grounding (eval + T4/A4 supervision)
│   │   ├── totalseg_dataset.py        # TotalSegmentator masks (eval: Dice)
│   │   ├── instruction_encoder.py     # instruction text → etext vector
│   │   ├── instruction_builder.py     # build the four instruction types (T1–T4)
│   │   └── multitask_sampler.py       # wraps CTRATE to emit one task per sample
│   ├── models/
│   │   ├── film.py                    # FiLM conditioning layer (γ, β from etext)
│   │   ├── vlm.py                     # MedVLM: encoder → projector → LLM (LoRA)
│   │   ├── encoder/
│   │   │   └── vit_encoder.py         # frozen 2D ViT slice encoder
│   │   └── projector/
│   │       ├── pos_encoding.py        # 2D sinusoidal (patches) + learnable depth
│   │       ├── stage1.py              # intra-slice latent distillation (N→K)
│   │       ├── stage2.py              # inter-slice aggregation + FiLM (D·K→M)
│   │       └── aadp.py                # combined two-stage projector
│   ├── training/
│   │   ├── losses.py                  # next-token CE (+ optional attention loss)
│   │   ├── scheduler.py               # cosine schedule with warmup
│   │   ├── factory.py                 # build projector variant from config
│   │   └── trainer.py                 # training loop, collate, checkpointing
│   ├── evaluation/
│   │   ├── metrics/
│   │   │   ├── radgraph_f1.py
│   │   │   ├── ratescore.py
│   │   │   ├── auroc_f1.py
│   │   │   ├── recall_at_k.py
│   │   │   └── dice_overlap.py
│   │   ├── probes/
│   │   │   └── classification_probe.py  # trained linear probe for T2
│   │   └── benchmarks/
│   │       └── vtcb.py                # Volumetric Token Compression Benchmark runner
│   ├── baselines/
│   │   ├── perceiver_projector.py     # RadFM/M3D-style single-stage, instruction-blind
│   │   └── medpruner_projector.py     # similarity-based hard slice deletion
│   ├── ablations/
│   │   ├── attention_conditioned_stage2.py  # FiLM → cross-attention conditioning
│   │   ├── task_conditioned_stage1.py       # instruction conditioning in Stage 1
│   │   └── auxiliary_attention_loss.py      # KL attention-alignment loss (A4)
│   └── visualization/
│       ├── attention_maps.py
│       └── compression_curves.py
├── scripts/
│   ├── preprocess_ctrate.py
│   ├── train.py
│   ├── evaluate.py
│   ├── _integration_check.py          # manual smoke check (not part of pytest)
│   └── _step5_vtcb_smoke.py           # manual smoke check (not part of pytest)
├── configs/                           # YAML experiment configs (base, ablations, baselines)
├── tests/                             # pytest suite (test_*.py)
├── setup.py
├── environment.yml
├── requirements.txt
└── README.md
```

**Acceptance criteria:** `python -c "import aadp"` runs without error; all directories exist.

---

### Task 0.2 — Dependencies & Environment File
Create `requirements.txt` and `environment.yml` with:
- `torch >= 2.2`, `torchvision`
- `transformers` (for LLM backbone and ViT)
- `einops` (for tensor reshaping in cross-attention)
- `timm` (ViT backbone)
- `SimpleITK` or `nibabel` (CT volume loading)
- `pydicom` (DICOM handling)
- `pandas`, `numpy`, `scipy`
- `scikit-learn` (AUROC/F1)
- `monai` (medical image augmentation)
- `radgraph` (entity/relation F1)
- `nltk` (for RaTEScore text processing)
- `matplotlib`, `seaborn` (visualization)
- `pytest` (testing)
- `pyyaml` (config loading)
- `tqdm`, `wandb` (training monitoring)

---

## PHASE 1 — Data Pipeline

### Task 1.1 — CT-RATE Dataset Loader
**File:** `data/ctrate_dataset.py`

Implement a PyTorch `Dataset` class for CT-RATE that:
- Loads CT volumes from NIfTI/DICOM files given a root path
- Returns `(volume_tensor, report_text, label_dict, patient_id)`
- `volume_tensor` shape: `(D, H, W)` where D = number of axial slices
- Respects the official CT-RATE train/validation/test split (load from split CSV)
- Supports configurable slice sampling (full volume or fixed-D subsampling)
- Applies per-volume HU windowing (e.g., lung window: [-1000, 400]) and normalisation to [0, 1]

**Acceptance criteria:** DataLoader iterates without error; batch shapes are correct.

---

### Task 1.2 — Slice-Level Preprocessing
**File:** `data/preprocessing.py`

Implement:
- `window_and_normalize(volume, hu_min, hu_max)` — HU clipping + min-max norm
- `resample_to_isotropic(volume, spacing)` — optional resampling using SimpleITK
- `pad_or_crop_depth(volume, target_D)` — pads or centre-crops along depth axis to a fixed D

---

### Task 1.3 — RadGenome-Chest CT Grounding Loader
**File:** `data/radgenome_dataset.py`

Implement a loader for RadGenome-Chest CT slice-level grounding annotations:
- Returns `(patient_id, finding_text, ground_truth_slice_indices)` tuples
- Used **only** for evaluation of Slice-Specific Lesion Recall (recall@k); never as training signal

---

### Task 1.4 — TotalSegmentator Mask Loader
**File:** `data/totalseg_dataset.py`

Implement a loader for TotalSegmentator structure masks (1,204 CT exams, 104 anatomical structures):
- Returns `(patient_id, structure_name, binary_mask_tensor)` tuples
- Used **only** for evaluation of Anatomical Localisation Accuracy (Dice overlap); never as training signal

---

### Task 1.5 — Instruction Tokenizer Wrapper
**File:** `data/instruction_encoder.py`

Implement `encode_instruction(text, tokenizer, max_length)`:
- Wraps a HuggingFace tokenizer
- Returns `etext` tensor of shape `(C,)` by mean-pooling the LLM's text encoder output over instruction tokens
- Must be compatible with LLaMA / Qwen / Mistral tokenizers (configurable via config)

---

## PHASE 2 — Core Architecture

### Task 2.1 — ViT Slice Encoder Wrapper
**File:** `models/encoder/vit_encoder.py`

Wrap a pretrained 2D ViT (e.g., `ViT-B/16` from `timm`) to:
- Accept a batch of 2D slices: input shape `(B*D, C, H, W)`
- Return patch token embeddings: shape `(B*D, N, C)` where N = number of patches (e.g., 1024 for 512×512 / 16×16)
- Support freezing backbone weights (configurable)
- Attach 2D sinusoidal positional encodings to patch tokens

---

### Task 2.2 — 2D Sinusoidal Positional Encoding
**File:** `models/projector/pos_encoding.py`

Implement:
- `SinusoidalPosEnc2D(H_patches, W_patches, C)` — standard 2D sine/cosine encoding, same shape as patch grid
- `LearnableDepthEnc1D(max_D, C)` — learnable 1D depth embedding, interpolated to actual D at runtime

---

### Task 2.3 — Stage 1: Intra-Slice Latent Distillation
**File:** `models/projector/stage1.py`

Implement `IntraSliceDistiller`:
- Input: `(B*D, N, C)` patch tokens from ViT encoder
- Learnable queries `Qs`: shape `(K, C)`, default K=32
- Single Perceiver-style cross-attention block:
  - Q = Qs (broadcast over batch)
  - K, V = input patch tokens
  - Standard scaled dot-product attention: `softmax(QK^T / sqrt(C)) V`
- Output: `(B*D, K, C)` — K latents per slice, same for all D slices
- Positional encodings (2D sinusoidal) added to K and V before attention

**Acceptance criteria:** Forward pass runs; output shape is `(B*D, K, C)`.

---

### Task 2.4 — FiLM Conditioning Layer
**File:** `models/film.py`

Implement `FiLMLayer`:
- Input: query tensor `Qd` of shape `(M, C)`, instruction embedding `etext` of shape `(C,)`
- Two linear projections: `gamma_proj: C → C`, `beta_proj: C → C`
- Output: `Qd' = Qd ⊙ gamma(etext) + beta(etext)` (elementwise, broadcast over M)
- Activation on gamma/beta: configurable (default: none, matching original FiLM paper)

---

### Task 2.5 — Stage 2: Inter-Slice Axial Aggregation
**File:** `models/projector/stage2.py`

Implement `InterSliceAggregator`:
- Input:
  - `slice_latents`: `(B, D, K, C)` — reshape from Stage 1 output
  - `etext`: `(B, C)` — instruction embedding
  - `depth_pos_enc`: `(D, C)` — learnable 1D depth encoding
- Steps:
  1. Add `depth_pos_enc` to slice latents along the D dimension
  2. Flatten to `(B, D*K, C)` as keys and values
  3. FiLM-modulate learnable depth queries `Qd` (shape `M, C`) using `etext` via `FiLMLayer`
  4. Cross-attention: `softmax(Qd' · (D*K tokens)^T / sqrt(C)) · V`
  5. Output: `(B, M, C)` — M final tokens for the LLM
- Save attention weights `(B, M, D*K)` for visualisation (detached, not in computation graph)

**Acceptance criteria:** Forward pass runs; output is `(B, M, C)`; attention weights are accessible.

---

### Task 2.6 — Combined A-ADP Projector
**File:** `models/projector/aadp.py`

Implement `AADPProjector` that chains Stage 1 → Stage 2:
- Accepts `(B, D, N, C)` volume tokens + `(B, C)` instruction embedding
- Handles reshaping between stages
- Returns `(B, M, C)` tokens + stage-2 attention weights
- Expose `K` and `M` as constructor arguments

---

### Task 2.7 — Full VLM Wrapper
**File:** `models/vlm.py`

Implement `MedVLM`:
- Components: ViT encoder + A-ADP projector + LLM (e.g., LLaMA-3-8B via HuggingFace)
- Forward pass:
  1. Encode volume slices with ViT → `(B*D, N, C)` tokens
  2. Encode instruction text → `etext`
  3. Run A-ADP projector → `(B, M, C)` visual tokens
  4. Concatenate visual tokens with instruction tokens in LLM embedding space
  5. Run LLM forward pass → logits / generated text
- Support both training (teacher forcing) and inference (autoregressive generation) modes
- Configurable: which components are frozen (ViT, LLM, projector only trainable by default)

---

## PHASE 3 — Baselines

### Task 3.1 — RadFM / M3D Perceiver Projector
**File:** `baselines/perceiver_projector.py`

Implement `PerceiverProjector`:
- Single-stage Perceiver resampler (isotropic, task-agnostic)
- Learnable queries Q of shape `(M, C)`
- Single cross-attention over all `D*N` tokens flattened
- Output: `(B, M, C)`
- This replicates the RadFM / M3D projector for fair comparison

---

### Task 3.2 — MedPruner Projector
**File:** `baselines/medpruner_projector.py`

Implement `MedPrunerProjector`:
- Similarity-based slice deletion: compute cosine similarity between adjacent slice tokens; discard slices below a threshold
- Hard deletion (tokens cannot be recovered after this step)
- Reduce remaining tokens to M via mean pooling
- Output: `(B, M, C)`

---

## PHASE 4 — Ablations

### Task 4.1 — Attention-Conditioned Stage 2 (Ablation)
**File:** `ablations/attention_conditioned_stage2.py`

Implement `AttentionConditionedStage2`:
- Same as Stage 2 but instead of FiLM modulation, inject `etext` via an additional cross-attention over `Qd`:
  - `Qd' = CrossAttention(Q=Qd, K=etext, V=etext)`
- Plugs in as a drop-in replacement for Stage 2 in `AADPProjector`
- Direct ablation: FiLM vs attention conditioning

---

### Task 4.2 — Auxiliary Attention Alignment Loss (Fallback Ablation)
**File:** `ablations/auxiliary_attention_loss.py`

Implement `AttentionAlignmentLoss`:
- Input: Stage 2 attention weights `(B, M, D*K)` + ground-truth slice indices from RadGenome
- Loss: KL divergence between predicted attention distribution over slices and ground-truth slice indicator (soft target)
- Only activated as a fallback if end-to-end training does not produce slice-level reallocation
- Weighted by a configurable `lambda_attn` hyperparameter

---

## PHASE 5 — Training

### Task 5.1 — Loss Function
**File:** `training/losses.py`

Implement:
- `NextTokenLoss`: standard cross-entropy over LLM output tokens (primary training signal)
- `CombinedLoss(next_token_loss, attn_loss, lambda_attn)`: weighted sum for the fallback ablation case

---

### Task 5.2 — Trainer
**File:** `training/trainer.py`

Implement `Trainer`:
- Standard PyTorch training loop with gradient accumulation
- Mixed precision (bf16 / fp16) via `torch.cuda.amp`
- Gradient clipping (configurable max norm)
- Checkpoint saving: save best model by validation RadGraph-F1
- WandB logging: loss, LR, GPU memory, per-metric validation scores
- Configurable frozen modules (skip gradient updates for frozen params)

---

### Task 5.3 — LR Scheduler
**File:** `training/scheduler.py`

Implement cosine annealing with linear warmup:
- `warmup_steps` configurable
- Decays to `min_lr` by end of training

---

### Task 5.4 — Experiment Configs
**Directory:** `configs/`

Create YAML configs for:
- `configs/aadp_base.yaml` — main A-ADP run (K=32, M=64)
- `configs/aadp_ablation_k_m_grid.yaml` — K ∈ {16,32,64} × M ∈ {16,32,64,128,256,512}
- `configs/aadp_ablation_attention_cond.yaml` — attention-conditioned Stage 2
- `configs/aadp_ablation_aux_loss.yaml` — auxiliary attention alignment loss
- `configs/baseline_perceiver.yaml` — RadFM/M3D baseline
- `configs/baseline_medpruner.yaml` — MedPruner baseline

Each config specifies: model, dataset paths, batch size, LR, warmup steps, frozen modules, token budgets.

---

### Task 5.5 — Training Entry Point
**File:** `scripts/train.py`

CLI script:
```
python scripts/train.py --config configs/aadp_base.yaml
```
- Loads config, instantiates model + data + trainer
- Supports `--resume` from checkpoint
- Logs to WandB

---

## PHASE 6 — Evaluation Metrics

### Task 6.1 — RadGraph-Based Entity/Relation F1
**File:** `evaluation/metrics/radgraph_f1.py`

Implement `compute_radgraph_f1(predictions, references)`:
- Uses RadGraph library to parse entities and relations from generated and reference reports
- Returns precision, recall, F1 per label and macro-averaged
- Follows CT-RATE evaluation protocol

---

### Task 6.2 — RaTEScore
**File:** `evaluation/metrics/ratescore.py`

Implement `compute_ratescore(predictions, references)`:
- Wraps the RaTEScore metric (entity-aware radiology text similarity)
- Returns mean score over the evaluation set

---

### Task 6.3 — Per-Label AUROC and F1
**File:** `evaluation/metrics/auroc_f1.py`

Implement `compute_auroc_f1(predictions, labels, label_names)`:
- Per-label AUROC over CT-RATE's abnormality labels
- Per-label F1 at 0.5 threshold
- Returns dict of `{label: auroc}` and `{label: f1}` plus macro averages

---

### Task 6.4 — Slice-Specific Lesion Recall@k
**File:** `evaluation/metrics/recall_at_k.py`

Implement `compute_recall_at_k(attn_weights, gt_slice_indices, k)`:
- Input: Stage 2 attention weights `(B, M, D*K)` aggregated to per-slice attention mass `(B, D)`
- Rank slices by attention mass descending
- Recall@k = fraction of cases where ground-truth slice falls in top-k ranked slices
- k matched to M (the token budget)

---

### Task 6.5 — Anatomical Localisation Dice
**File:** `evaluation/metrics/dice_overlap.py`

Implement `compute_dice_overlap(attn_spatial_map, structure_mask)`:
- Convert Stage 2 per-slice attention mass `(D,)` into a 3D spatial attention map (broadcast over H, W within each slice)
- Threshold at mean attention to produce a binary prediction volume
- Compute Dice overlap against TotalSegmentator binary structure mask
- Return per-structure Dice and macro average across 104 structures

---

## PHASE 7 — VTCB Benchmark Runner

### Task 7.1 — Benchmark Orchestrator
**File:** `evaluation/benchmarks/vtcb.py`

Implement `VTCBRunner`:
- Loops over all four task families
- Loops over all token budgets M ∈ {16, 32, 64, 128, 256, 512}
- For each (task, M): runs inference, computes metrics, logs results to a results dict
- Saves results as `results/vtcb_{model_name}.json`
- Generates compression-quality curves (metric vs M) as matplotlib figures
- Reports parameter counts for each projector (using `sum(p.numel() for p in model.parameters())`)

---

### Task 7.2 — Evaluation Entry Point
**File:** `scripts/evaluate.py`

CLI script:
```
python scripts/evaluate.py --config configs/aadp_base.yaml --checkpoint path/to/ckpt.pt
```
- Runs VTCB benchmark for the specified model and checkpoint
- Saves JSON results and plots

---

## PHASE 8 — Visualisation & Analysis

### Task 8.1 — Stage 2 Attention Map Visualiser
**File:** `visualization/attention_maps.py`

Implement `visualize_attention(volume, attn_weights, instruction, save_path)`:
- Input: CT volume `(D, H, W)`, Stage 2 per-slice attention mass `(D,)`, instruction string
- Output: figure with two panels:
  - Left: per-slice attention mass bar chart along depth axis
  - Right: CT slice montage with attention mass overlaid as colour intensity
- Highlight top-k slices (k = M)
- Save as PNG

---

### Task 8.2 — Compression-Quality Curve Plotter
**File:** `visualization/compression_curves.py`

Implement `plot_compression_curves(results_json_paths, metric_names, save_path)`:
- Loads VTCB result JSONs for multiple models (A-ADP, baselines, ablations)
- Plots metric vs M curves for each model on the same axes
- One plot per metric (RadGraph-F1, AUROC, Recall@k, Dice)

---

## PHASE 9 — Tests

### Task 9.1 — Unit Tests
**Directory:** `tests/`

Write pytest unit tests for:
- `test_stage1.py` — IntraSliceDistiller: correct output shape `(B*D, K, C)` for varied B, D, N, C, K
- `test_film.py` — FiLMLayer: output shape; gamma=1/beta=0 when etext=0 check
- `test_stage2.py` — InterSliceAggregator: output shape `(B, M, C)`; attention weights accessible
- `test_aadp.py` — AADPProjector end-to-end: `(B, D, N, C)` + `(B, C)` → `(B, M, C)`
- `test_baselines.py` — Perceiver and MedPruner projectors: shape checks
- `test_metrics.py` — Smoke tests for all six metric functions with dummy inputs
- `test_dataset.py` — DataLoader iterates; batch shapes match spec

---

## PHASE 10 — Documentation

### Task 10.1 — README
**File:** `README.md`

Include:
- Project overview (one paragraph)
- Installation instructions
- How to preprocess CT-RATE
- How to train (with config examples)
- How to run VTCB evaluation
- How to reproduce ablations
- Citation block

---

## Execution Order for Claude Code

Feed tasks to Claude Code **in this order**:

```
Phase 0 → Phase 1 → Phase 2 (Tasks 2.1–2.7 in order) →
Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 →
Phase 8 → Phase 9 → Phase 10
```

Within Phase 2, the dependency chain is strict:
**2.2 → 2.1 → 2.3 → 2.4 → 2.5 → 2.6 → 2.7**
(pos encodings and encoder before projector stages; FiLM before Stage 2; both stages before combined projector; combined projector before VLM wrapper)

All other phases can proceed after Phase 2 completes.