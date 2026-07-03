# Anatomy-Aware Dynamic Projector (A-ADP) for 3-D Medical Vision–Language Models

A-ADP is a two-stage token-compression projector for CT volumes that conditions
slice selection on the clinical instruction. Stage 1 (IntraSliceDistiller) compresses
each axial slice's ViT patch tokens to a compact latent using Perceiver cross-attention.
Stage 2 (InterSliceAggregator) aggregates the per-slice latents into a fixed token
budget M for the LLM, guided by FiLM-modulated depth queries that focus attention on
the slices most relevant to the query. The result is a model that allocates its token
budget adaptively — attending to the chest when asked about nodules, to the abdomen
when asked about liver lesions — without any additional supervision beyond the
report-generation objective.

## Installation

```bash
pip install -r requirements.txt
huggingface-cli login          # required for gated models (Llama-3.2-1B)
```

Python 3.10+ and PyTorch 2.1+ with CUDA are required.

## Data

Training uses the [CT-RATE dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
streamed directly from HuggingFace — no local download is necessary.
The streaming dataset is implemented in [data/ctrate_dataset.py](data/ctrate_dataset.py).

## Training

All training commands share the same entry point. Pass the appropriate config YAML
and the script handles model construction, optimiser setup, and checkpointing.

**A-ADP (primary model)**
```bash
python scripts/train.py --config configs/aadp_base.yaml
```

**Perceiver baseline** (RadFM / M3D-style flat projector, no instruction conditioning)
```bash
python scripts/train.py --config configs/baseline_perceiver.yaml
```

**MedPruner baseline** (score-and-prune token selection)
```bash
python scripts/train.py --config configs/baseline_medpruner.yaml
```

**A1 ablation** (task-conditioned Stage 1 — FiLM applied to Perceiver queries)
```bash
python scripts/train.py --config configs/aadp_ablation_task_cond_stage1.yaml
```

All runs log to Weights & Biases when `use_wandb: true` is set in the config
(default on). Set `use_wandb: false` to disable.

## Evaluation

**Run the VTCB benchmark** (sweeps token budgets M, evaluates four task families):
```bash
python scripts/evaluate.py \
    --config configs/aadp_base.yaml \
    --checkpoint checkpoints/aadp_base_best.pt \
    --budgets 16 32 64 128
```

**Compare two models** at their primary budget across all metrics:
```bash
python scripts/evaluate.py \
    --compare results/aadp_vtcb.json results/perceiver_vtcb.json \
    --model_names A-ADP Perceiver
```

**Plot compression–quality curves** (generates PNGs and PDFs):
```bash
python scripts/evaluate.py \
    --plot results/aadp_vtcb.json results/perceiver_vtcb.json \
    --model_names A-ADP Perceiver \
    --metrics radgraph_f1 ratescore_mean auroc_macro
```

For paper-quality figures across all result files in a directory:
```bash
python -c "
from aadp.visualization.compression_curves import plot_paper_figures
plot_paper_figures('results/', 'figures/')
"
```

## Ablations

| Config | Projector | What it tests |
|--------|-----------|---------------|
| `configs/aadp_base.yaml` | `aadp` | A-ADP with FiLM in Stage 2 only (primary model) |
| `configs/aadp_ablation_task_cond_stage1.yaml` | `aadp_task_cond_stage1` | A1 — FiLM applied to Stage 1 Perceiver queries as well |
| `configs/aadp_ablation_attention_cond.yaml` | `attention_conditioned_aadp` | Replace FiLM with instruction-aware attention scoring in Stage 2 |
| `configs/aadp_ablation_aux_loss.yaml` | `aadp` | Same as base + auxiliary attention-alignment loss during training |

Each ablation config is otherwise identical to the base (same LR, batch size, epochs,
ViT/LLM backbone) so results are directly comparable.

## Reproducing visualisations

**Attention map for a single volume** (requires a trained checkpoint):
```python
from aadp.visualization.attention_maps import visualize_attention
import torch

# volume: (D, H, W) tensor loaded from your dataset
# attn:   (D,) per-slice attention mass from model.projector.get_slice_attention()
visualize_attention(volume, attn, instruction="Are there lung nodules?",
                    save_path="figures/attn_nodule.png",
                    gt_slice_indices=[12, 13, 14])  # optional RadGenome GT
```

**Side-by-side instruction comparison** (the key qualitative figure):
```python
from aadp.visualization.attention_maps import compare_instructions

compare_instructions(
    volume,
    instructions=["Are there lung nodules?", "Is there pleural effusion?"],
    checkpoints=[(model, "A-ADP")],
    save_path="figures/compare_instructions.png",
)
```

## Hardware

Training requires a GPU with **at least 16 GB VRAM** (A100 recommended).
The Llama-3.2-1B backbone and mixed-precision training fit comfortably on a
single A100 40 GB.

Evaluation with RadGraph-F1 loads an additional DyGIE++ model (~2 GB) on top
of the main model. This exceeds typical consumer GPU memory; run evaluation on
the A100, not locally.

## Citation

```bibtex
@techreport{adeniji2025aadp,
  title   = {Anatomy-Aware Dynamic Projector for 3-D Medical Vision--Language Models},
  author  = {Adeniji, Samuel Akinwumi},
  year    = {2025},
  institution = {Summer Research Internship},
}
```
