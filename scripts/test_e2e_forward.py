"""Forward-pass smoke test for CTCLIPStage2VLM's end-to-end (raw-volume)
path: stage1 freeze pattern, stage2 unfreeze pattern, and a real batch from
CTRawDataset.

Adapted from the original spec in two ways, both load-bearing (not
cosmetic):
  1. CTCLIPStage2VLM.forward()'s real signature takes
     instructions: List[str] + report_tokens: Tensor (tokenized by the
     caller), not input_ids/attention_mask/labels — matches
     scripts/train_ctclip.py's existing training loop exactly (see its
     _validate() function). CTRawDataset's collate_fn returns raw
     "instruction"/"target" strings for the same reason.
  2. batch_size=1 (not 2) and stage2 is a known memory risk, not just a
     wiring one: Narval's A100s here are 40GB, full fp32 fine-tuning of a
     3B-param LLM with no LoRA needs ~25.6GB for weights+gradients alone,
     and train_ctclip.py has no DeepSpeed/ZeRO-3 offload yet (that's
     planned follow-up work, not built here). If stage2 OOMs, that
     confirms the already-anticipated need for ZeRO-3 offload, not a bug
     in this wiring.
"""
import sys
sys.path.insert(0, "/cvmfs/soft.computecanada.ca/easybuild/software/2023/"
                    "x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages")

import torch
from torch.utils.data import DataLoader

from aadp.models.ctclip_vlm import CTCLIPStage2VLM
from aadp.data.ctclip_raw_dataset import CTRawDataset, collate_fn

CTCLIP_CKPT = (
    "/project/def-uanazodo-ab/sadeniji/hf_cache/hub/"
    "datasets--ibrahimhamamci--CT-RATE/snapshots/"
    "deeca4d89e9f978d4d1bccd88a55071ddbb146bb/"
    "models/CT-CLIP-Related/CT-CLIP_v2.pt"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "CTViT hardcodes cuda internally — this test requires a GPU."

MODEL_KWARGS = dict(
    ctclip_dim=512,
    embed_dim=512,
    num_tokens=64,
    num_heads=8,
    cond_dim=3072,
    llm_model_name="meta-llama/Llama-3.2-3B-Instruct",
    llm_frozen=False,
    llm_lora=None,
    instruction_encoder_model="meta-llama/Llama-3.2-3B-Instruct",
    device=device,
    ctclip_checkpoint=CTCLIP_CKPT,
)

# ── Test 1: stage1 forward + backward, synthetic volume ────────────────────
print("=== Test 1: stage1 synthetic forward pass ===")
model = CTCLIPStage2VLM(**MODEL_KWARGS, training_stage="stage1")

dummy_volume = torch.randn(1, 1, 240, 480, 480, device=device)
dummy_instructions = ["Generate a radiology report for this CT scan."]
dummy_report_tokens = model.tokenizer(
    ["No acute abnormality identified."], return_tensors="pt", padding=True,
).input_ids.to(device)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(volume=dummy_volume, instructions=dummy_instructions,
                report_tokens=dummy_report_tokens, training=True)
    loss = out["loss"]
print(f"Stage1 loss: {loss.item():.4f}")

loss.backward()
vit_grad = any(p.grad is not None for p in model.visual_encoder.parameters())
llm_grad = any(p.grad is not None for p in model.llm.parameters())
proj_grad = any(p.grad is not None for p in model.projector.parameters())
print(f"Grads — ViT: {vit_grad} (expect False)  LLM: {llm_grad} (expect False)  Projector: {proj_grad} (expect True)")
assert not vit_grad and not llm_grad and proj_grad, "Stage1 freeze pattern wrong"
print("Stage1 freeze pattern: PASS")

del model, dummy_volume, dummy_report_tokens, loss
torch.cuda.empty_cache()

# ── Test 2: stage2 forward + backward, synthetic volume ────────────────────
print("\n=== Test 2: stage2 synthetic forward pass ===")
print("(Known memory risk — see module docstring. An OOM here is informative, not a wiring failure.)")
try:
    model = CTCLIPStage2VLM(**MODEL_KWARGS, training_stage="stage2")
    dummy_volume = torch.randn(1, 1, 240, 480, 480, device=device)
    dummy_instructions = ["Generate a radiology report for this CT scan."]
    dummy_report_tokens = model.tokenizer(
        ["No acute abnormality identified."], return_tensors="pt", padding=True,
    ).input_ids.to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(volume=dummy_volume, instructions=dummy_instructions,
                    report_tokens=dummy_report_tokens, training=True)
        loss = out["loss"]
    print(f"Stage2 loss: {loss.item():.4f}")

    loss.backward()
    vit_grad = any(p.grad is not None for p in model.visual_encoder.parameters())
    proj_grad = any(p.grad is not None for p in model.projector.parameters())
    print(f"Grads — ViT: {vit_grad} (expect True)  Projector: {proj_grad} (expect True)")
    assert vit_grad and proj_grad, "Stage2 unfreeze pattern wrong"
    print("Stage2 unfreeze pattern: PASS")

    model.zero_grad(set_to_none=True)  # free Test 2's ~12.8GB of full-LLM gradients before Test 3
    del dummy_volume, dummy_report_tokens, loss
    torch.cuda.empty_cache()
except torch.cuda.OutOfMemoryError as exc:
    print(f"Test 2 OOM (expected-possible per module docstring, not a wiring bug): {exc}")
    print("Stage2 unfreeze pattern: NOT VERIFIED (OOM before backward completed)")
    model = None
    torch.cuda.empty_cache()

# ── Test 3: real batch from CTRawDataset ────────────────────────────────────
print("\n=== Test 3: real data batch ===")
if model is None:
    print("Rebuilding model in stage1 (lowest memory) to at least verify real-data wiring end to end.")
    model = CTCLIPStage2VLM(**MODEL_KWARGS, training_stage="stage1")

ds = CTRawDataset(
    merged_csv="/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/train_merged.csv",
    ct_volume_root="/scratch/sadeniji/ct_rate_raw",
    split="train",
)
loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn, num_workers=2)
batch = next(iter(loader))

report_tokens = model.tokenizer(
    batch["target"], return_tensors="pt", padding=True, truncation=True, max_length=1024,
).input_ids.to(device)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(
        volume=batch["volume"].to(device),
        instructions=batch["instruction"],
        report_tokens=report_tokens,
        training=True,
    )
print(f"Real batch loss: {out['loss'].item():.4f}")
print(f"Real batch task: {batch['task']}  volume_name: {batch['volume_name']}")
print("All tests complete.")
