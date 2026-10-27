"""Shape-check for CT-CLIP's real CTViT visual encoder (see docs/ for the
provenance chain: HF dataset ibrahimhamamci/CT-RATE, models/CT-CLIP-Related/
CT-CLIP_v2.pt; architecture code cloned from
github.com/ibrahimethemhamamci/CT-CLIP into /scratch/sadeniji/ct_clip_src).

Confirms the pretrained visual_transformer.* weights load cleanly into a
freshly-constructed CTViT (hyperparameters taken verbatim from the reference
repo's scripts/run_train.py) and that forward(video, return_encoded_tokens=True)
produces the (B, 24, 576, 512) convention already used throughout
aadp/models/projector/.
"""
import sys
sys.path.insert(0, "/cvmfs/soft.computecanada.ca/easybuild/software/2023/"
                    "x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages")

import torch
from transformer_maskgit import CTViT

CKPT_PATH = (
    "/project/def-uanazodo-ab/sadeniji/hf_cache/hub/"
    "datasets--ibrahimhamamci--CT-RATE/snapshots/"
    "deeca4d89e9f978d4d1bccd88a55071ddbb146bb/"
    "models/CT-CLIP-Related/CT-CLIP_v2.pt"
)

# Hyperparameters copied verbatim from ct_clip_src/scripts/run_train.py's
# CTViT(...) call — must match exactly or load_state_dict shapes mismatch.
visual_encoder = CTViT(
    dim=512,
    codebook_size=8192,
    image_size=480,
    patch_size=20,
    temporal_patch_size=10,
    spatial_depth=4,
    temporal_depth=4,
    dim_head=32,
    heads=8,
)

print("Loading checkpoint (this is a 1.65GB file, may take a moment) ...")
full_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
visual_sd = {
    k[len("visual_transformer."):]: v
    for k, v in full_sd.items()
    if k.startswith("visual_transformer.")
}
print(f"Extracted {len(visual_sd)} visual_transformer.* keys from {len(full_sd)} total checkpoint keys.")

missing, unexpected = visual_encoder.load_state_dict(visual_sd, strict=False)
print(f"Missing keys ({len(missing)}): {missing}")
print(f"Unexpected keys ({len(unexpected)}): {unexpected}")
if missing or unexpected:
    print("WARNING: non-empty missing/unexpected keys — architecture may not match the checkpoint exactly.")
else:
    print("OK: state_dict loaded with zero missing/unexpected keys.")

visual_encoder.eval()

# The reference code hardcodes device=torch.device('cuda') in several
# internal spots (attention.py's ContinuousPositionBias, ctvit.py's
# encode/decode) regardless of the input tensor's actual device, so the
# module itself must live on GPU for those internal tensors to match.
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: no CUDA available — this reference implementation hardcodes "
          "cuda internally and will not run on CPU.")
visual_encoder = visual_encoder.to(device)

print(f"\nRunning dummy forward pass: (1, 1, 240, 480, 480) on {device} ...")
dummy = torch.randn(1, 1, 240, 480, 480, device=device)
with torch.no_grad():
    out = visual_encoder(dummy, return_encoded_tokens=True)

print(f"Output type : {type(out)}")
print(f"Output shape: {tuple(out.shape)}")

b, t, h, w, d = out.shape
reshaped = out.reshape(b, t, h * w, d)
print(f"Reshaped to (B, D, K, C): {tuple(reshaped.shape)}")
expected = (1, 24, 576, 512)
if tuple(reshaped.shape) == expected:
    print(f"OK: matches expected {expected} convention.")
else:
    print(f"MISMATCH: expected {expected}, got {tuple(reshaped.shape)} — STOP, do not proceed.")
