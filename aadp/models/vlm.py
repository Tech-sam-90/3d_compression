"""MedVLM — full medical VLM: ViT slice encoder → A-ADP projector → LLM.

Variable-depth design:
    Volumes can have any D (depth) depending on the scanner protocol.  The
    forward method accepts whatever D the volume has natively.  Padding across
    a batch is handled by the caller via ``variable_depth_collate_fn`` — this
    is why ``pad_or_crop_depth`` is *optional* preprocessing rather than a
    mandatory pipeline step.  Only set ``vit_resize_to`` when GPU memory is
    the hard constraint and you are willing to trade fine spatial detail for
    memory savings.
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from aadp.data.instruction_encoder import InstructionEncoder
from aadp.models.encoder.vit_encoder import SliceEncoder
from aadp.models.projector.aadp import AADPProjector


class MedVLM(nn.Module):
    """Full medical VLM wrapper.

    Chains together:
    1. ``SliceEncoder`` — encodes each CT slice with a frozen ViT
    2. ``AADPProjector`` — two-stage FiLM-conditioned compression to M tokens
    3. LLM — generates radiology report text from M visual tokens

    Args:
        vit_model_name:            timm ViT model name.
        vit_frozen:                Freeze ViT weights. Default True.
        vit_pretrained:            Load pretrained ViT weights. Default True.
        vit_resize_to:             Resize slices to this square size before
                                   encoding.  ``None`` preserves native CT
                                   resolution (recommended). Only set when GPU
                                   memory is the binding constraint.
        llm_model_name:            HuggingFace model ID for the LLM backbone.
        llm_frozen:                Freeze LLM weights. Default True (only the
                                   projector trains by default).
        embed_dim:                 Hint for the token dimension.  Actual dims
                                   are derived from the ViT and LLM configs
                                   automatically; a ``visual_proj`` bridge is
                                   inserted when they differ.
        num_latents:               K — per-slice latents from Stage 1.
        num_tokens:                M — final tokens fed to the LLM.
        cond_dim:                  Hint for the instruction embedding dim.
                                   Actual value is derived from
                                   ``instruction_encoder_model``.
        use_film:                  FiLM conditioning in Stage 2. Default True.
        max_depth:                 Max CT depth for the depth encoder.
        instruction_encoder_model: HF model name for InstructionEncoder.
        device:                    Target device.
    """

    def __init__(
        self,
        vit_model_name: str = "vit_base_patch16_224",
        vit_frozen: bool = True,
        vit_pretrained: bool = True,
        vit_resize_to: Optional[int] = None,
        llm_model_name: str = "facebook/opt-125m",
        llm_frozen: bool = True,
        embed_dim: int = 768,
        num_latents: int = 32,
        num_tokens: int = 64,
        cond_dim: int = 768,
        use_film: bool = True,
        max_depth: int = 512,
        instruction_encoder_model: str = "facebook/opt-125m",
        projector: Optional[nn.Module] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()
        _ = embed_dim, cond_dim  # actual dims derived from models below
        _device = torch.device(device)

        # ── Slice encoder ────────────────────────────────────────────────────
        self.vit = SliceEncoder(
            model_name=vit_model_name,
            pretrained=vit_pretrained,
            frozen=vit_frozen,
            resize_to=vit_resize_to,
        )
        C_vit = self.vit.output_dim

        # ── Instruction encoder (always frozen) ──────────────────────────────
        self.instruction_encoder = InstructionEncoder(
            model_name=instruction_encoder_model,
            frozen=True,
        )
        C_cond = self.instruction_encoder.output_dim

        # ── Projector (pre-built or built here) ──────────────────────────────
        if projector is not None:
            self.projector = projector
            self._num_tokens = projector.num_tokens
        else:
            self.projector = AADPProjector(
                embed_dim=C_vit,
                num_latents=num_latents,
                num_tokens=num_tokens,
                cond_dim=C_cond,
                use_film=use_film,
                max_depth=max_depth,
                device=device,
            )
            self._num_tokens = num_tokens

        # ── LLM ──────────────────────────────────────────────────────────────
        # torch_dtype=float32 is explicit: transformers ≥5.x defaults to float16
        # which creates a mixed-precision state in OPT (float32 LayerNorm weights
        # but float16 linear layers) that is inconsistent when inputs_embeds are
        # float32 from our projector.  Loading in float32 keeps everything consistent.
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, dtype=torch.float32
        )
        if llm_frozen:
            self.llm.requires_grad_(False)
        C_llm = self.llm.config.hidden_size

        # LLM tokenizer — used to embed instruction tokens in forward()
        self._llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self._llm_tokenizer.pad_token is None:
            self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token
            self._llm_tokenizer.pad_token_id = self._llm_tokenizer.eos_token_id

        # ── Visual projection bridge ─────────────────────────────────────────
        # If ViT and LLM operate in different spaces, bridge them with a
        # trainable linear layer. Otherwise use identity (no extra params).
        if C_vit != C_llm:
            self.visual_proj: nn.Module = nn.Linear(C_vit, C_llm)
        else:
            self.visual_proj = nn.Identity()

        self.to(_device)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        volumes: torch.Tensor,
        instructions: List[str],
        report_tokens: Optional[torch.Tensor] = None,
        depth_spacing_mm: Optional[float] = None,
        max_new_tokens: int = 256,
    ) -> Any:
        """Run the full MedVLM pipeline.

        Args:
            volumes:          ``(B, D, H, W)`` preprocessed CT volumes in
                              ``[0, 1]``.  D can vary per call; the DataLoader
                              handles padding via ``variable_depth_collate_fn``.
            instructions:     One instruction string per batch item.
            report_tokens:    ``(B, L_rep)`` tokenized report for
                              teacher-forcing during training. ``None`` at
                              inference.
            depth_spacing_mm: Physical slice spacing in mm; passed to Stage 2's
                              depth positional encoding.
            max_new_tokens:   Token budget for inference generation.

        Returns:
            Training   (``report_tokens`` is not None):
                ``CausalLMOutputWithPast`` — has a ``loss`` attribute.
            Inference  (``report_tokens`` is None):
                ``torch.Tensor`` of shape ``(B, generated_length)`` — token ids.
        """
        device = volumes.device
        B, D, H, W = volumes.shape

        # ── Step 1: ViT encodes each slice ───────────────────────────────────
        slices = volumes.reshape(B * D, 1, H, W)        # (B*D, 1, H, W)
        patch_tokens = self.vit(slices).float()          # (B*D, N, C_vit) — force fp32
        N = patch_tokens.shape[1]
        C_vit = patch_tokens.shape[2]

        # When resize_to is set the ViT operates on a different spatial size than
        # the raw volume — use the resized dims for the patch grid, not H/W.
        if self.vit.resize_to is not None:
            H_patches = self.vit.resize_to // self.vit.patch_size
            W_patches = self.vit.resize_to // self.vit.patch_size
        else:
            H_patches = H // self.vit.patch_size
            W_patches = W // self.vit.patch_size

        patch_tokens = patch_tokens.reshape(B, D, N, C_vit)

        # ── Step 2: encode instructions ──────────────────────────────────────
        etext = self.instruction_encoder(instructions).float()  # (B, C_cond)

        # ── Step 3: A-ADP projector ──────────────────────────────────────────
        visual_tokens = self.projector(
            patch_tokens, etext, H_patches, W_patches, depth_spacing_mm
        )                                                # (B, M, C_vit)

        # ── Step 4: bridge to LLM space ──────────────────────────────────────
        visual_tokens = self.visual_proj(visual_tokens).float()  # (B, M, C_llm)

        # ── Step 5: tokenize instructions for LLM embedding look-up ─────────
        inst_enc = self._llm_tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inst_ids = inst_enc["input_ids"].to(device)      # (B, L_inst)
        embed_fn = self.llm.get_input_embeddings()
        text_embeds = embed_fn(inst_ids).float()         # (B, L_inst, C_llm)
        L_inst = text_embeds.shape[1]

        # ── Step 6: build input sequence [visual | instruction] ──────────────
        input_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        # (B, M + L_inst, C_llm)

        # ── Training path ─────────────────────────────────────────────────────
        if report_tokens is not None:
            report_tokens = report_tokens.to(device)     # (B, L_rep)
            L_rep = report_tokens.shape[1]
            report_embeds = embed_fn(report_tokens)      # (B, L_rep, C_llm)

            # Full sequence: visual | instruction | report
            input_embeds = torch.cat([input_embeds, report_embeds], dim=1)
            # (B, M + L_inst + L_rep, C_llm)

            # Labels: -100 masks visual + instruction positions from the loss
            labels_ignore = torch.full(
                (B, self._num_tokens + L_inst), -100,
                dtype=torch.long, device=device,
            )
            labels = torch.cat([labels_ignore, report_tokens], dim=1)
            # (B, M + L_inst + L_rep)

            attention_mask = torch.ones(
                B, self._num_tokens + L_inst + L_rep,
                dtype=torch.long, device=device,
            )

            return self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        # ── Inference path ────────────────────────────────────────────────────
        attention_mask = torch.ones(
            B, self._num_tokens + L_inst,
            dtype=torch.long, device=device,
        )

        return self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self._llm_tokenizer.eos_token_id,
        )


    def rebuild_projector_at_budget(self, M: int) -> None:
        """Rebuild Stage 2 of the projector with a new token budget M.

        Stage 1 weights are preserved.  Works with any projector that exposes
        a ``rebuild_at_budget(M)`` method (``AADPProjector``,
        ``TaskConditionedAADP``).

        Args:
            M: New number of output tokens for the LLM.

        Raises:
            NotImplementedError: If the projector does not support budget
                                 rebuilding (e.g. PerceiverProjector).
        """
        if not hasattr(self.projector, "rebuild_at_budget"):
            raise NotImplementedError(
                f"{type(self.projector).__name__} does not support "
                "rebuild_at_budget. Only AADPProjector and TaskConditionedAADP "
                "implement this method."
            )
        self.projector.rebuild_at_budget(M)
        self._num_tokens = M


# ── Variable-depth DataLoader collate function ────────────────────────────────


def variable_depth_collate_fn(batch: List[Dict]) -> Dict:
    """Collate a batch of CT samples that may differ in D, H, and W.

    CT-RATE volumes vary in depth (number of slices) and spatial resolution
    (e.g. 512×512 vs 1024×1024).  This function pads D, H, and W independently
    to the per-batch maximum using zeros so all volumes stack to (B, D, H, W).

    Expected keys per sample dict:
        "volumes"          — ``(D, H, W)`` float tensor
        "instructions"     — str
        "report_tokens"    — ``(L,)`` long tensor or None
        "depth_spacing_mm" — float or None
        "label_dict"       — dict (abnormality labels) or None
        "patient_id"       — str or None

    Returns:
        Dict with stacked/padded tensors and lists for non-tensor fields.
        Key ``"volumes"`` has shape ``(B, max_D, max_H, max_W)``.
    """
    max_D = max(item["volumes"].shape[0] for item in batch)
    max_H = max(item["volumes"].shape[1] for item in batch)
    max_W = max(item["volumes"].shape[2] for item in batch)

    # Pad volumes along D, H, W to batch maxima
    padded_vols: List[torch.Tensor] = []
    for item in batch:
        vol = item["volumes"]   # (D, H, W)
        D, H, W = vol.shape
        pad_D = max_D - D
        pad_H = max_H - H
        pad_W = max_W - W
        if pad_D > 0 or pad_H > 0 or pad_W > 0:
            # F.pad pads from last dim backwards: (W_left, W_right, H_top, H_bot, D_front, D_back)
            vol = F.pad(vol, (0, pad_W, 0, pad_H, 0, pad_D))
        padded_vols.append(vol)

    # Pad report_tokens to max report length (if any are present)
    report_tokens_list = [item.get("report_tokens") for item in batch]
    if all(rt is None for rt in report_tokens_list):
        report_tokens_stacked = None
    else:
        max_L = max(
            (rt.shape[0] for rt in report_tokens_list if rt is not None), default=0
        )
        padded_rts: List[torch.Tensor] = []
        for rt in report_tokens_list:
            if rt is None:
                padded_rts.append(torch.zeros(max_L, dtype=torch.long))
            else:
                if rt.shape[0] < max_L:
                    pad = torch.zeros(max_L - rt.shape[0], dtype=torch.long)
                    rt = torch.cat([rt, pad], dim=0)
                padded_rts.append(rt)
        report_tokens_stacked = torch.stack(padded_rts, dim=0)

    return {
        "volumes": torch.stack(padded_vols, dim=0),
        "instructions": [item.get("instructions", "") for item in batch],
        "report_tokens": report_tokens_stacked,
        "depth_spacing_mm": batch[0].get("depth_spacing_mm"),
        "label_dicts": [item.get("label_dict") for item in batch],
        "patient_ids": [item.get("patient_id") for item in batch],
    }
