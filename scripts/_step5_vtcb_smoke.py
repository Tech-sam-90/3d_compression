"""Step 5 — end-to-end VTCB smoke run for the integration check."""
import os
import sys
import time
import shutil

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset
from aadp.models.vlm import MedVLM
from aadp.evaluation.benchmarks.vtcb import VTCBRunner

print("=" * 60)
print("STEP 5: END-TO-END VTCB SMOKE RUN")
print("=" * 60)

device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats(device)

model = MedVLM(
    vit_model_name="vit_tiny_patch16_224",
    vit_pretrained=False,
    llm_model_name="facebook/opt-125m",
    llm_frozen=True,
    vit_frozen=True,
    num_latents=8,
    num_tokens=32,
    device=device,
).to(device)
print("Model loaded.")


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
assert "report_generation" in results[32], "report_generation key missing"

task_keys = [k for k in results[32] if not k.startswith("_")]
rg_keys = list(results[32]["report_generation"].keys())

print(f"Task families at M=32 : {task_keys}")
print(f"report_generation keys: {rg_keys}")
print(f"Elapsed               : {elapsed:.1f}s")
print(f"Peak GPU memory       : {peak_mb:.0f} MB")
print("VTCB smoke run: OK")

shutil.rmtree(results_dir, ignore_errors=True)

print()
print("=" * 60)
print("ALL INTEGRATION STEPS PASSED")
print("=" * 60)
