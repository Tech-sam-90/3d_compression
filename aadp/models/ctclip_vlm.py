"""CTCLIPStage2VLM — VLM that consumes pre-extracted CT-CLIP features.

Replaces the full MedVLM (ViT + Stage 1 + Stage 2 + LLM) with a leaner model
that skips the ViT and Stage 1 entirely, feeding CT-CLIP's pre-extracted
(B, 24, 576, 512) feature tensors directly into Stage 2.

This is the main model for the ctclip-stage2-train experiment.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from aadp.data.instruction_encoder import InstructionEncoder
from aadp.models.projector.ctclip_stage2 import CTCLIPStage2Projector


class CTCLIPStage2VLM(nn.Module):
    """Full VLM that skips ViT + Stage 1 and feeds CT-CLIP features to Stage 2.

    Components (frozen unless noted):
        instruction_encoder:  Frozen InstructionEncoder for etext.
        projector:            Trainable CTCLIPStage2Projector.
        visual_proj:          Trainable bridge Linear(embed_dim → llm_hidden).
        llm:                  OPT-1.3B (frozen base + trainable LoRA adapters).

    Forward sequence (training):
        features (B,24,576,512)
          → projector          (B, M, embed_dim)
          → visual_proj        (B, M, llm_hidden)
          → cat [visual | inst | report]
          → LLM loss over report positions

    Args:
        ctclip_dim:               CT-CLIP channel dim (default 512).
        embed_dim:                Stage 2 working dim (default 512).
        num_tokens:               M — LLM token budget (default 64).
        num_heads:                Stage 2 attention heads (default 8).
        cond_dim:                 Instruction encoder output dim (default 2048).
        use_film:                 FiLM conditioning in Stage 2. Default True.
        max_depth:                Max depth for LearnableDepthEnc1D. Default 24.
        dropout:                  Stage 2 dropout. Default 0.0.
        llm_model_name:           HF model ID. Default "facebook/opt-1.3b".
        llm_frozen:               Freeze base LLM (ignored when LoRA enabled).
        llm_lora:                 LoRA config dict or None.
        instruction_encoder_model: HF model ID for instruction encoder.
        device:                   Target device.
    """

    def __init__(
        self,
        ctclip_dim: int = 512,
        embed_dim: int = 512,
        num_tokens: int = 64,
        num_heads: int = 8,
        cond_dim: int = 2048,
        use_film: bool = True,
        max_depth: int = 24,
        dropout: float = 0.0,
        llm_model_name: str = "facebook/opt-1.3b",
        llm_frozen: bool = False,
        llm_lora: Optional[Dict] = None,
        instruction_encoder_model: str = "facebook/opt-1.3b",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        _device = torch.device(device)

        # ── Instruction encoder (always frozen) ──────────────────────────────
        self.instruction_encoder = InstructionEncoder(
            model_name=instruction_encoder_model,
            pooling="mean",
            frozen=True,
            max_length=128,
        )
        actual_cond_dim = self.instruction_encoder.output_dim
        assert actual_cond_dim == cond_dim, (
            f"cond_dim mismatch: config says {cond_dim} but "
            f"{instruction_encoder_model} has hidden_size={actual_cond_dim}. "
            "Update cond_dim in the config to match."
        )

        # ── Stage 2 projector ─────────────────────────────────────────────────
        self.projector = CTCLIPStage2Projector(
            ctclip_dim=ctclip_dim,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            cond_dim=cond_dim,
            dropout=dropout,
            use_film=use_film,
            max_depth=max_depth,
            device=device,
        )

        # ── LLM backbone ─────────────────────────────────────────────────────
        # Explicit float32 to avoid mixed-precision OPT LayerNorm issues.
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=torch.float32
        )
        llm_hidden: int = self.llm.config.hidden_size

        if llm_lora is not None and llm_lora.get("enabled", False):
            from peft import LoraConfig, TaskType, get_peft_model

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=llm_lora.get("r", 16),
                lora_alpha=llm_lora.get("alpha", 32),
                target_modules=llm_lora.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=llm_lora.get("dropout", 0.05),
                bias="none",
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
        elif llm_frozen:
            for p in self.llm.parameters():
                p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── Visual projection bridge ─────────────────────────────────────────
        # Always a Linear: embed_dim (512) → llm_hidden (2048 for OPT-1.3B)
        if embed_dim != llm_hidden:
            self.visual_proj: nn.Module = nn.Linear(embed_dim, llm_hidden)
        else:
            self.visual_proj = nn.Identity()

        self._num_tokens = num_tokens
        self.to(_device)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        features: torch.Tensor,
        instructions: List[str],
        report_tokens: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Dict:
        """Run the CT-CLIP Stage 2 VLM pipeline.

        Args:
            features:      ``(B, 24, 576, 512)`` pre-extracted CT-CLIP features.
            instructions:  One instruction string per batch item.
            report_tokens: ``(B, L_rep)`` tokenized target text for training.
                           Pass ``None`` for inference.
            training:      If True and report_tokens is provided, compute LM loss.
                           If False (or report_tokens is None), generate text.

        Returns:
            Training: ``{"loss": Tensor scalar, "logits": Tensor (B, L, V)}``
            Inference: ``{"generated_ids": Tensor (B, gen_len)}``
        """
        device = features.device
        B = features.shape[0]

        # ── Step 1: instruction embedding ────────────────────────────────────
        etext = self.instruction_encoder(instructions).float()   # (B, cond_dim)

        # ── Step 2: Stage 2 projection ───────────────────────────────────────
        visual = self.projector(features, etext)                 # (B, M, embed_dim)
        visual = self.visual_proj(visual).float()                # (B, M, llm_hidden)

        # ── Step 3: tokenize instruction strings for LLM embedding ──────────
        inst_enc = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        embed_fn = self.llm.get_input_embeddings()
        inst_embeds = embed_fn(inst_enc.input_ids).float()       # (B, L_inst, llm_hidden)
        L_inst = inst_embeds.shape[1]
        M = visual.shape[1]

        # ── Training path ─────────────────────────────────────────────────────
        if training and report_tokens is not None:
            report_tokens = report_tokens.to(device)             # (B, L_rep)
            rep_embeds = embed_fn(report_tokens).float()         # (B, L_rep, llm_hidden)
            L_rep = rep_embeds.shape[1]

            inputs_embeds = torch.cat([visual, inst_embeds, rep_embeds], dim=1)
            # (B, M + L_inst + L_rep, llm_hidden)

            # Mask visual + instruction positions from the LM loss
            labels_ignore = torch.full(
                (B, M + L_inst), -100, dtype=torch.long, device=device
            )
            labels = torch.cat([labels_ignore, report_tokens], dim=1)
            # (B, M + L_inst + L_rep)

            attention_mask = torch.ones(
                B, M + L_inst + L_rep, dtype=torch.long, device=device
            )

            out = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            return {"loss": out.loss, "logits": out.logits}

        # ── Inference path ────────────────────────────────────────────────────
        inputs_embeds = torch.cat([visual, inst_embeds], dim=1)
        attention_mask = torch.ones(B, M + L_inst, dtype=torch.long, device=device)

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        return {"generated_ids": gen_ids}

    # ── Budget sweep ──────────────────────────────────────────────────────────

    def rebuild_at_budget(self, M: int) -> None:
        """Rebuild Stage 2 with a new token budget M for VTCB sweeps."""
        self.projector.rebuild_at_budget(M)
        self._num_tokens = M
