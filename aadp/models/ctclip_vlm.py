"""CTCLIPStage2VLM — VLM that consumes pre-extracted CT-CLIP features.

Replaces the full MedVLM (ViT + Stage 1 + Stage 2 + LLM) with a leaner model
that skips the ViT and Stage 1 entirely, feeding CT-CLIP's pre-extracted
(B, 24, 576, 512) feature tensors directly into Stage 2.

This is the main model for the ctclip-stage2-train experiment.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformer_maskgit import CTViT
from transformers import AutoModelForCausalLM, AutoTokenizer

from aadp.data.instruction_encoder import InstructionEncoder
from aadp.models.projector.ctclip_stage2 import CTCLIPStage2Projector

# CTViT hyperparameters — copied verbatim from the reference CT-CLIP repo's
# scripts/run_train.py (github.com/ibrahimethemhamamci/CT-CLIP, cloned to
# /scratch/sadeniji/ct_clip_src). These are checkpoint-mandated, not
# configurable: they must match models/CT-CLIP-Related/CT-CLIP_v2.pt's
# (ibrahimhamamci/CT-RATE HF dataset repo) weight shapes exactly, and they
# fix the encoder's input to (B, 1, 240, 480, 480) and its pre-pooling
# output to (240/10, (480/20)**2, 512) = (24, 576, 512) — the convention
# CTCLIPStage2Projector already expects. Verified end-to-end on GPU
# (scripts/check_ctclip_shape.py, job 66923233): checkpoint loads with 0
# missing/0 unexpected keys, output reshapes to exactly (B, 24, 576, 512).
_CTVIT_HPARAMS = dict(
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


def find_lora_target_modules(model: nn.Module) -> List[str]:
    """Auto-discover LoRA target module names (adapted from M3D's find_all_linear_names).

    Scans ``model`` for ``nn.Linear`` submodules and returns the set of
    distinct leaf names, excluding modules that shouldn't get LoRA adapters
    (projection/head/embedding layers). Intended to be called on the base
    LLM (``self.llm``) before it's wrapped by ``get_peft_model`` — after
    wrapping, module names/structure change and Linear layers become
    ``lora.Linear`` wrappers.
    """
    exclude = [
        "visual_proj", "projector", "cls_head",
        "instruction_encoder", "lm_head", "embed_tokens"
    ]
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if not any(kw in name for kw in exclude):
                names.add(name.split(".")[-1])
    return sorted(names)


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
        use_film:                 FiLM conditioning in Stage 2, only used when
                                   conditioning="film". Default True.
        max_depth:                Max depth for LearnableDepthEnc1D. Default 24.
        dropout:                  Stage 2 dropout. Default 0.0.
        top_k:                    Sparse cross-attention top-K (see
                                   InterSliceAggregator's docstring). Default 128.
        conditioning:              "film" (default) or "attention" — see
                                   CTCLIPStage2Projector's docstring.
        llm_model_name:           HF model ID. Default "facebook/opt-1.3b".
        llm_frozen:               Freeze base LLM (ignored when LoRA enabled).
        llm_lora:                 LoRA config dict or None.
        instruction_encoder_model: HF model ID for instruction encoder.
        device:                   Target device.
        training_stage:           "stage1" (ViT+LLM frozen, only projector
                                   trains) or "stage2" (everything unfrozen).
                                   Only affects the volume= (raw-CT,
                                   end-to-end) path — see set_training_stage().
        ctclip_checkpoint:        Optional path to CT-CLIP_v2.pt (or
                                   equivalent). If given, self.visual_encoder
                                   (CTViT) is loaded from its
                                   "visual_transformer.*" keys; raises if the
                                   architecture doesn't match exactly (any
                                   missing/unexpected key). If None,
                                   visual_encoder keeps its random init —
                                   only useful for architecture/shape tests,
                                   never for real training.
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
        top_k: int = 128,
        conditioning: str = "film",
        llm_model_name: str = "facebook/opt-1.3b",
        llm_frozen: bool = False,
        llm_lora: Optional[Dict] = None,
        instruction_encoder_model: str = "facebook/opt-1.3b",
        num_labels: Optional[int] = None,
        device: str = "cuda",
        training_stage: str = "stage2",
        ctclip_checkpoint: Optional[str] = None,
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
            top_k=top_k,
            conditioning=conditioning,
            device=device,
        )

        # ── LLM backbone ─────────────────────────────────────────────────────
        # Explicit float32 to avoid mixed-precision OPT LayerNorm issues.
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=torch.float32
        )
        # Trades ~30% speed for activation-memory savings needed to fit
        # 1024-token sequences on A100-40G. Safe here without
        # enable_input_require_grads(): inputs_embeds is a concat that
        # includes `visual` (from the trainable visual_proj), so it already
        # requires grad — the usual frozen-embedding gotcha doesn't apply.
        self.llm.gradient_checkpointing_enable()
        llm_hidden: int = self.llm.config.hidden_size
        # GPT-2-family models (BioMedLM included) have a FIXED positional
        # embedding table — n_positions/max_position_embeddings — and
        # indexing past it is a CUDA device-side assert (gather kernel
        # index out of bounds), not a graceful error. visual+instruction+
        # report tokens are concatenated before hitting the LLM, so a
        # report max_length sized for the LLM alone isn't safe once M and
        # L_inst are added on top (confirmed root cause of job 66397199's
        # crash at step ~125: BioMedLM's n_positions=1024, and
        # M+L_inst+L_rep reached ~1216). forward() clamps defensively
        # using this so it's safe regardless of num_tokens (M varies 16-512
        # across VTCB sweep configs) or instruction length.
        self._max_seq_len = (
            getattr(self.llm.config, "n_positions", None)
            or getattr(self.llm.config, "max_position_embeddings", None)
        )

        if llm_lora is not None and llm_lora.get("enabled", False):
            from peft import LoraConfig, TaskType, get_peft_model

            target_modules = find_lora_target_modules(self.llm)
            print(f"LoRA targets: {target_modules}")
            if not target_modules:
                target_modules = ["c_attn", "c_proj"]
                print(f"find_lora_target_modules found nothing; falling back to {target_modules}")

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=llm_lora.get("r", 16),
                lora_alpha=llm_lora.get("alpha", 32),
                target_modules=target_modules,
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

        # ── Auxiliary classification head (Argus Section 5.1 two-stage training) ──
        # Mean-pools visual_proj's (B, M, llm_hidden) output — AFTER visual_proj,
        # not before — so cls_loss's gradient reaches visual_proj too. Stage 1
        # trains aggregator + visual_proj + cls_head (per spec); pooling before
        # visual_proj would leave it receiving zero gradient and stuck at random
        # init, which is exactly what happened the first time this was tried
        # (confirmed via checkpoint inspection: saved visual_proj weights were
        # bit-for-bit consistent with fresh nn.Linear init, and Test 1 generation
        # was pure gibberish — a random affine map feeding the LLM). The LLM
        # itself is still skippable via compute_lm=False; only visual_proj (one
        # more Linear) is now unconditionally in the Stage 1 forward path.
        self.cls_head: Optional[nn.Module] = (
            nn.Linear(llm_hidden, num_labels) if num_labels else None
        )

        # ── Visual encoder (CTViT) — end-to-end raw-volume path ────────────────
        # Hyperparameters are checkpoint-mandated (_CTVIT_HPARAMS module
        # comment) — not derived from ctclip_dim/embed_dim/etc. above, which
        # govern the projector, not this encoder.
        self.visual_encoder = CTViT(**_CTVIT_HPARAMS)
        if ctclip_checkpoint is not None:
            full_sd = torch.load(ctclip_checkpoint, map_location="cpu", weights_only=False)
            visual_sd = {
                k[len("visual_transformer."):]: v
                for k, v in full_sd.items()
                if k.startswith("visual_transformer.")
            }
            # null_kv keys are 0-element (num_null_kv=0 in every Attention
            # instance CTViT actually constructs) — transformer_maskgit/
            # attention.py no longer registers them at all (a genuinely
            # zero-numel nn.Parameter breaks DeepSpeed ZeRO-3's parameter
            # gathering: AllGatherCoalescedHandle.wait()'s partitions list
            # ends up empty, torch.cat() raises). Dropping them here is a
            # no-op for correctness — concatenating/padding with a 0-length
            # null_kv was already a mathematical no-op — but they'd
            # otherwise show up as "unexpected" against the checkpoint.
            visual_sd = {k: v for k, v in visual_sd.items() if not k.endswith(".null_kv")}
            result = self.visual_encoder.load_state_dict(visual_sd, strict=False)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(
                    f"CT-CLIP visual encoder checkpoint mismatch against {ctclip_checkpoint}: "
                    f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
                )
            print(f"Loaded CT-CLIP visual encoder from {ctclip_checkpoint} "
                  f"({len(visual_sd)} keys, 0 missing, 0 unexpected).")

        self._num_tokens = num_tokens
        self.to(_device)

        self.training_stage = training_stage
        self.set_training_stage(training_stage)

    # ── Training stage ───────────────────────────────────────────────────────

    def set_training_stage(self, stage: str) -> None:
        """Switch which components train. Only touches visual_encoder/llm/
        projector/visual_proj — instruction_encoder is always frozen
        (fixed at construction, orthogonal to staging) and cls_head (if
        present) is left as-is (its own Stage-1-aggregator-only-training use
        case, unrelated to this end-to-end ViT/LLM staging).

        stage1: freeze visual_encoder + llm, train projector + visual_proj
                only — matches the existing feature-based Stage 1 semantics,
                extended to also keep the ViT frozen.
        stage2: unfreeze everything. Enables gradient checkpointing on
                visual_encoder if it supports it (CTViT, this reference
                implementation, does not — hasattr() guard makes this a
                no-op rather than an error; full backprop through an
                unfrozen CTViT on (1, 240, 480, 480) inputs has no
                checkpointing memory relief today). llm's gradient
                checkpointing is already enabled unconditionally above,
                regardless of stage.
        """
        assert stage in ("stage1", "stage2"), f"stage must be 'stage1' or 'stage2', got {stage!r}"
        self.training_stage = stage

        if stage == "stage1":
            self.visual_encoder.requires_grad_(False)
            self.llm.requires_grad_(False)
            self.projector.requires_grad_(True)
            self.visual_proj.requires_grad_(True)
        else:  # stage2
            self.visual_encoder.requires_grad_(True)
            self.llm.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.visual_proj.requires_grad_(True)
            if hasattr(self.visual_encoder, "gradient_checkpointing_enable"):
                self.visual_encoder.gradient_checkpointing_enable()

    def get_parameter_groups(
        self,
        stage1_lr: float = 1e-4,
        vit_lr: float = 1e-5,
        llm_lr: float = 2e-5,
    ) -> List[Dict]:
        """Per-component LR groups for the optimizer, matching self.training_stage.

        stage1: only projector + visual_proj have requires_grad=True (see
                set_training_stage), so only they get a param group.
        stage2: visual_encoder / projector+visual_proj / llm, each at its
                own LR.
        """
        if self.training_stage == "stage1":
            return [{
                "params": list(self.projector.parameters()) + list(self.visual_proj.parameters()),
                "lr": stage1_lr,
            }]
        return [
            {"params": list(self.visual_encoder.parameters()), "lr": vit_lr},
            {"params": list(self.projector.parameters()) + list(self.visual_proj.parameters()), "lr": stage1_lr},
            {"params": list(self.llm.parameters()), "lr": llm_lr},
        ]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        features: Optional[torch.Tensor] = None,
        volume: Optional[torch.Tensor] = None,
        instructions: List[str] = None,
        report_tokens: Optional[torch.Tensor] = None,
        training: bool = True,
        compute_lm: bool = True,
        max_new_tokens: int = 256,
        generate_kwargs: Optional[Dict] = None,
    ) -> Dict:
        """Run the CT-CLIP Stage 2 VLM pipeline.

        Args:
            features:      ``(B, 24, 576, 512)`` pre-extracted CT-CLIP features.
                           Provide exactly one of features= / volume=.
            volume:        ``(B, 1, D, H, W)`` raw CT volume, e.g.
                           ``(B, 1, 240, 480, 480)`` from CTRawDataset — run
                           through self.visual_encoder (CTViT) to produce
                           features on the fly. Provide exactly one of
                           features= / volume=.
            instructions:  One instruction string per batch item.
            report_tokens: ``(B, L_rep)`` tokenized target text for training.
                           Pass ``None`` for inference.
            training:      If True and report_tokens is provided, compute LM loss.
                           If False (or report_tokens is None), generate text.
            compute_lm:    If False, skip the tokenizer/LLM entirely (visual_proj
                           still runs — cls_head depends on it) and return only
                           cls_logits (requires num_labels to have been set).
                           Used by Argus-style Stage 1 aggregator-only training,
                           where the LLM forward pass would be wasted compute
                           (LoRA frozen ⇒ zero contribution to loss).
            max_new_tokens: Inference path only. Default 256 matches this
                           method's long-standing hardcoded value exactly —
                           existing callers that don't pass it see no change.
            generate_kwargs: Inference path only. Extra kwargs merged into
                           the self.llm.generate() call (e.g. do_sample=True),
                           overriding the method's own defaults
                           (do_sample=False, repetition_penalty=1.3,
                           no_repeat_ngram_size=3) where keys collide. None
                           (the default) reproduces prior behavior exactly.

        Returns:
            Training: ``{"loss": Tensor scalar, "logits": Tensor (B, L, V),
                         "cls_logits": Tensor (B, num_labels) if num_labels set}``
            Stage 1 (compute_lm=False): ``{"cls_logits": Tensor (B, num_labels)}``
            Inference: ``{"generated_ids": Tensor (B, gen_len)}``
        """
        assert (features is None) != (volume is None), "Provide exactly one of features= or volume="
        if volume is not None:
            # (B, t, h, w, d) pre-pooling tokens — see _CTVIT_HPARAMS comment
            # for why this reshapes to exactly (B, 24, 576, 512).
            enc = self.visual_encoder(volume, return_encoded_tokens=True)
            b, t, h, w, d = enc.shape
            features = enc.reshape(b, t, h * w, d)

        device = features.device
        B = features.shape[0]
        # These were hardcoded .float() below, written for the plain-fp32 /
        # torch.autocast feature-based path (a no-op there — every param is
        # already fp32, so .to(model_dtype) == .float() in that mode). Under
        # DeepSpeed's bf16 mode (aadp/models/ctclip_vlm.py's volume= path,
        # scripts/train_ctclip.py's _train_e2e()) every submodule's weights
        # are bf16, so a hardcoded fp32 cast mid-graph mismatches the next
        # bf16 Linear it feeds into (RuntimeError: mat1 and mat2 must have
        # the same dtype) — cast to the model's own actual dtype instead.
        model_dtype = next(self.parameters()).dtype

        # ── Step 1: instruction embedding ────────────────────────────────────
        etext = self.instruction_encoder(instructions).to(model_dtype)   # (B, cond_dim)

        # ── Step 2: Stage 2 projection ───────────────────────────────────────
        visual = self.projector(features, etext)                 # (B, M, embed_dim)
        visual = self.visual_proj(visual).to(model_dtype)         # (B, M, llm_hidden)

        result: Dict = {}
        if self.cls_head is not None:
            pooled = visual.mean(dim=1)                           # (B, llm_hidden)
            result["cls_logits"] = self.cls_head(pooled)

        if not compute_lm:
            return result

        # ── Step 3: tokenize instruction strings for LLM embedding ──────────
        inst_enc = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        embed_fn = self.llm.get_input_embeddings()
        inst_embeds = embed_fn(inst_enc.input_ids).to(model_dtype)       # (B, L_inst, llm_hidden)
        L_inst = inst_embeds.shape[1]
        M = visual.shape[1]

        # ── Training path ─────────────────────────────────────────────────────
        if training and report_tokens is not None:
            report_tokens = report_tokens.to(device)             # (B, L_rep)
            rep_embeds = embed_fn(report_tokens).to(model_dtype)         # (B, L_rep, llm_hidden)
            L_rep = rep_embeds.shape[1]

            # Defensive clamp: visual (M) + instruction (L_inst) + report
            # (L_rep) must not exceed the LLM's positional embedding table
            # size, or the position lookup indexes out of bounds (CUDA
            # device-side assert). Truncate the report's TAIL only — visual
            # and instruction tokens are never dropped.
            if self._max_seq_len is not None:
                total_len = M + L_inst + L_rep
                if total_len > self._max_seq_len:
                    keep_rep = max(self._max_seq_len - M - L_inst, 0)
                    rep_embeds = rep_embeds[:, :keep_rep, :]
                    report_tokens = report_tokens[:, :keep_rep]
                    L_rep = keep_rep

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
            result["loss"] = out.loss
            result["logits"] = out.logits
            return result

        # ── Inference path ────────────────────────────────────────────────────
        inputs_embeds = torch.cat([visual, inst_embeds], dim=1)
        attention_mask = torch.ones(B, M + L_inst, dtype=torch.long, device=device)

        gen_call_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        if generate_kwargs:
            gen_call_kwargs.update(generate_kwargs)

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_call_kwargs,
        )
        result["generated_ids"] = gen_ids
        return result

    # ── Generation convenience wrapper ──────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        instructions: List[str],
        features: Optional[torch.Tensor] = None,
        volume: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> List[str]:
        """Decoded-text convenience wrapper around forward()'s existing
        inference path (training=False, report_tokens=None) — no separate
        generation logic to duplicate/drift out of sync with forward()'s
        training path. Takes instructions: List[str] (not input_ids/
        attention_mask) because that's forward()'s real contract: it
        tokenizes instructions itself internally rather than accepting
        pre-tokenized ids from the caller.

        Args:
            instructions: One instruction string per batch item.
            features:     ``(B, 24, 576, 512)``. Exactly one of features=/volume=.
            volume:       ``(B, 1, D, H, W)``. Exactly one of features=/volume=.
            max_new_tokens: Forwarded to forward().
            **generate_kwargs: Forwarded to forward()'s generate_kwargs (e.g.
                do_sample=True), overriding forward()'s own generation defaults.

        Returns:
            List[str], one decoded (special-tokens-stripped) string per batch item.
        """
        was_training = self.training
        self.eval()
        try:
            result = self.forward(
                features=features,
                volume=volume,
                instructions=instructions,
                report_tokens=None,
                training=False,
                compute_lm=True,
                max_new_tokens=max_new_tokens,
                generate_kwargs=generate_kwargs or None,
            )
        finally:
            self.train(was_training)
        gen_ids = result["generated_ids"]
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in gen_ids
        ]

    # ── Budget sweep ──────────────────────────────────────────────────────────

    def rebuild_at_budget(self, M: int) -> None:
        """Rebuild Stage 2 with a new token budget M for VTCB sweeps."""
        self.projector.rebuild_at_budget(M)
        self._num_tokens = M
