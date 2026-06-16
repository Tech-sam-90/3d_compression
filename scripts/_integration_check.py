"""Phases 1-7 integration check — run before Colab training."""
import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda")

# ── Step 3: five-way projector comparison ─────────────────────────────────────

print("=" * 70)
print("STEP 3: FIVE-WAY PROJECTOR COMPARISON")
print("=" * 70)

torch.manual_seed(0)
B, D, N, C, M = 2, 32, 196, 768, 64
patch_tokens = torch.randn(B, D, N, C, device=device)
etext_a = torch.randn(B, 768, device=device)
etext_b = torch.randn(B, 768, device=device)

from aadp.models.projector.aadp import AADPProjector
from ablations.attention_conditioned_stage2 import AttentionConditionedAADP
from ablations.task_conditioned_stage1 import TaskConditionedAADP
from baselines.perceiver_projector import PerceiverProjector
from baselines.medpruner_projector import MedPrunerProjector

projectors = {
    "A-ADP (FiLM Stage2 only)":   AADPProjector(C, num_latents=16, num_tokens=M, cond_dim=768, device=device),
    "A-ADP (FiLM Stage1+Stage2)":  TaskConditionedAADP(C, num_latents=16, num_tokens=M, cond_dim=768, device=device),
    "A-ADP (Attn Stage2)":         AttentionConditionedAADP(C, num_latents=16, num_tokens=M, cond_dim=768, device=device),
    "Perceiver (RadFM/M3D)":       PerceiverProjector(C, num_tokens=M, device=device),
    "MedPruner":                   MedPrunerProjector(C, num_tokens=M, device=device),
}

print(f"{'Projector':<34} {'Shape':<18} {'etext matters':<16} {'params':>12}")
print("-" * 84)
for name, proj in projectors.items():
    with torch.no_grad():
        out_a = proj(patch_tokens, etext_a, H_patches=14, W_patches=14)
        out_b = proj(patch_tokens, etext_b, H_patches=14, W_patches=14)
    shape_ok = out_a.shape == (B, M, C)
    etext_matters = not torch.allclose(out_a, out_b)
    counts = proj.num_parameters() if hasattr(proj, "num_parameters") else {
        "total": sum(p.numel() for p in proj.parameters())
    }
    assert shape_ok, f"{name}: wrong shape {out_a.shape}"
    print(f"{name:<34} {str(out_a.shape):<18} {str(etext_matters):<16} {counts.get('total', '?'):>12,}")

print()
print("Note: FiLM is identity at init so all A-ADP variants show etext_matters=False before training")
print("Five-way comparison: OK")

# ── Step 4: rebuild_at_budget ─────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 4: rebuild_at_budget CHECK")
print("=" * 70)

proj_r = AADPProjector(768, num_latents=16, num_tokens=64, cond_dim=768, device=device)
pt_r = torch.randn(2, 16, 196, 768, device=device)
et_r = torch.randn(2, 768, device=device)

s1_before = proj_r.stage1.latents.data.clone()

out_64 = proj_r(pt_r, et_r, H_patches=14, W_patches=14)
assert out_64.shape == (2, 64, 768), f"Wrong shape at M=64: {out_64.shape}"

proj_r.rebuild_at_budget(32)
out_32 = proj_r(pt_r, et_r, H_patches=14, W_patches=14)
assert out_32.shape == (2, 32, 768), f"Wrong shape at M=32: {out_32.shape}"
assert torch.allclose(proj_r.stage1.latents.data, s1_before), "Stage1 weights changed after rebuild!"

proj_r.rebuild_at_budget(128)
out_128 = proj_r(pt_r, et_r, H_patches=14, W_patches=14)
assert out_128.shape == (2, 128, 768), f"Wrong shape at M=128: {out_128.shape}"
assert torch.allclose(proj_r.stage1.latents.data, s1_before), "Stage1 weights changed after second rebuild!"

print(f"rebuild_at_budget: M=64 {out_64.shape} -> M=32 {out_32.shape} -> M=128 {out_128.shape}")
print("Stage 1 weights preserved across both rebuilds: True")
print("rebuild_at_budget: OK")

# ── Step 5: end-to-end VTCB smoke run ────────────────────────────────────────

print()
print("=" * 70)
print("STEP 5: END-TO-END VTCB SMOKE RUN")
print("=" * 70)

import shutil
from torch.utils.data import Dataset
from aadp.models.vlm import MedVLM
from aadp.evaluation.benchmarks.vtcb import VTCBRunner

torch.cuda.reset_peak_memory_stats(device)

model = MedVLM(
    vit_model_name="vit_tiny_patch16_224",
    vit_frozen=True,
    vit_resize_to=224,
    llm_model_name="facebook/opt-125m",
    llm_frozen=True,
    num_latents=8,
    num_tokens=32,
    device=device,
).to(device)


class MockValDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return {
            "volumes": torch.rand(16, 224, 224),
            "instructions": "Describe any findings",
            "report_tokens": None,
            "depth_spacing_mm": 3.0,
            "label_dicts": {"Lung nodule": 0, "Emphysema": 1},
            "patient_ids": f"test_{i}",
        }


results_dir = "results/smoke_test/"
os.makedirs(results_dir, exist_ok=True)

runner = VTCBRunner(
    model=model,
    val_dataset=MockValDataset(),
    radgenome_dataset=None,
    totalseg_dataset=None,
    token_budgets=[32],
    primary_budget=32,
    batch_size=2,
    max_new_tokens=16,
    device=str(device),
    results_dir=results_dir,
)

t0 = time.time()
results = runner.run(model_name="smoke_test")
elapsed = time.time() - t0

peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

assert 32 in results, "Token budget 32 missing from results"
assert "report_generation" in results[32], "report_generation missing"
print(f"VTCB smoke run results: {[k for k in results[32].keys() if not k.startswith('_')]}")
print(f"Elapsed: {elapsed:.1f}s")
print(f"Peak GPU memory: {peak_mb:.0f} MB")
print("VTCB end-to-end smoke run: OK")

# Clean up
shutil.rmtree(results_dir, ignore_errors=True)

print()
print("=" * 70)
print("ALL INTEGRATION STEPS PASSED")
print("=" * 70)
