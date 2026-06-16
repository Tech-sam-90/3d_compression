import torch
from aadp.models.encoder.vit_encoder import SliceEncoder
from aadp.models.projector.aadp import AADPProjector
from aadp.models.vlm import MedVLM, variable_depth_collate_fn
from aadp.data.instruction_encoder import InstructionEncoder

device = torch.device("cuda")
torch.manual_seed(42)
torch.cuda.reset_peak_memory_stats()

# ── 1. Simulate a batch of 2 CT volumes at native resolution ──────────────────
vol_a = torch.rand(303, 512, 512)
vol_b = torch.rand(251, 512, 512)

batch = variable_depth_collate_fn([
    {"volumes": vol_a, "instructions": "Find lung nodules",
     "report_tokens": None, "depth_spacing_mm": 3.0,
     "label_dict": {}, "patient_id": "train_1_a_1"},
    {"volumes": vol_b, "instructions": "Any pleural effusion?",
     "report_tokens": None, "depth_spacing_mm": 3.5,
     "label_dict": {}, "patient_id": "train_2_a_1"},
])

volumes = batch["volumes"].to(device)
instructions = batch["instructions"]
print(f"Collated volume shape: {volumes.shape}")
assert volumes.shape == (2, 303, 512, 512), f"Collate shape wrong: {volumes.shape}"

# ── 2. Build MedVLM ──────────────────────────────────────────────────────────
model = MedVLM(
    vit_model_name="vit_tiny_patch16_224",
    vit_frozen=True,
    vit_resize_to=224,
    llm_model_name="facebook/opt-125m",
    llm_frozen=True,
    embed_dim=192,
    num_latents=16,
    num_tokens=32,
    cond_dim=768,
    instruction_encoder_model="facebook/opt-125m",
    device=device,
).to(device)

# ── 3. Inference forward pass ─────────────────────────────────────────────────
with torch.no_grad():
    output = model(volumes, instructions, report_tokens=None,
                   depth_spacing_mm=3.0, max_new_tokens=8)
print(f"Generated token ids shape: {output.shape}")
assert output.ndim == 2, "Inference output should be (B, generated_length)"
print("Inference forward pass: OK")

peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak GPU memory (303-slice forward): {peak_gb:.3f} GB")

# ── 4. Slice attention ────────────────────────────────────────────────────────
attn = model.projector.get_slice_attention()
print(f"Slice attention shape: {attn.shape}")
assert attn.shape == (2, 303), f"Attention shape wrong: {attn.shape}"
assert attn.min() >= 0.0, "Attention weights should be non-negative"
print("Slice attention: OK")

# ── 5. Frozen/trainable audit ─────────────────────────────────────────────────
trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
frozen    = [(n, p.shape) for n, p in model.named_parameters() if not p.requires_grad]
print(f"Trainable parameter groups: {len(trainable)}")
print(f"Frozen parameter groups:    {len(frozen)}")
assert all("projector" in n or "visual_proj" in n for n, _ in trainable), \
    "Non-projector parameters are trainable — check frozen flags"
print("Frozen/trainable audit: OK")

# ── 6. Parameter counts ───────────────────────────────────────────────────────
counts = model.projector.num_parameters()
print(f"Projector parameters: {counts}")
assert counts["total"] == counts["stage1"] + counts["stage2"]
assert counts["total"] > 0
print("Parameter counts: OK")

# ── 7. FiLM conditioning active ───────────────────────────────────────────────
etext_a = torch.randn(2, 768, device=device)
etext_b = torch.randn(2, 768, device=device)
patch_tokens = torch.randn(2, 32, 196, 192, device=device)
# Perturb FiLM weights away from identity init
with torch.no_grad():
    model.projector.stage2.film.gamma_proj.weight.normal_(std=0.1)
    model.projector.stage2.film.beta_proj.weight.normal_(std=0.1)
out_a = model.projector(patch_tokens, etext_a, H_patches=14, W_patches=14)
out_b = model.projector(patch_tokens, etext_b, H_patches=14, W_patches=14)
assert not torch.allclose(out_a, out_b), "FiLM conditioning has no effect — check Stage 2"
print("FiLM conditioning active: OK")

print("\nAll Phase 2 integration checks PASSED.")
