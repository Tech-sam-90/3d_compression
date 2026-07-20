"""Projector factory — builds the right projector from a config dict."""

from typing import Any, Dict, Union

import torch
import torch.nn as nn

from aadp.models.projector.aadp import AADPProjector


def build_projector(
    config: Dict[str, Any],
    embed_dim: int,
    cond_dim: int,
    device: Union[torch.device, str] = "cuda",
) -> nn.Module:
    """Instantiate the projector specified by ``config["projector"]``.

    Args:
        config:    Training config dict (from YAML + CLI overrides).
        embed_dim: ViT output dimension C_vit.
        cond_dim:  InstructionEncoder output dimension C_cond.
        device:    Target device.

    Returns:
        An ``nn.Module`` with ``num_tokens`` property and
        ``forward(patch_tokens, etext, H_patches, W_patches)`` signature.

    Raises:
        ValueError: If ``config["projector"]`` is not a known key.
    """
    projector_type: str = config.get("projector", "aadp")
    num_tokens: int = config.get("num_tokens", 64)
    max_depth: int = config.get("max_depth", 512)

    if projector_type == "aadp":
        return AADPProjector(
            embed_dim=embed_dim,
            num_latents=config.get("num_latents", 32),
            num_tokens=num_tokens,
            cond_dim=cond_dim,
            use_film=config.get("use_film", True),
            max_depth=max_depth,
            device=device,
        )

    if projector_type == "attention_conditioned_aadp":
        from aadp.ablations.attention_conditioned_stage2 import AttentionConditionedAADP

        return AttentionConditionedAADP(
            embed_dim=embed_dim,
            num_latents=config.get("num_latents", 32),
            num_tokens=num_tokens,
            cond_dim=cond_dim,
            max_depth=max_depth,
            device=device,
        )

    if projector_type == "perceiver":
        from aadp.baselines.perceiver_projector import PerceiverProjector

        return PerceiverProjector(
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            device=device,
        )

    if projector_type == "aadp_task_cond_stage1":
        from aadp.ablations.task_conditioned_stage1 import TaskConditionedAADP

        return TaskConditionedAADP(
            embed_dim=embed_dim,
            num_latents=config.get("num_latents", 32),
            num_tokens=num_tokens,
            cond_dim=cond_dim,
            use_film=config.get("use_film", True),
            max_depth=max_depth,
            device=device,
        )

    if projector_type == "medpruner":
        from aadp.baselines.medpruner_projector import MedPrunerProjector

        return MedPrunerProjector(
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            similarity_threshold=config.get("similarity_threshold", 0.95),
            device=device,
        )

    if projector_type == "ctclip_stage2":
        from aadp.models.ctclip_vlm import CTCLIPStage2VLM

        return CTCLIPStage2VLM(
            ctclip_dim=config.get("ctclip_dim", 512),
            embed_dim=config.get("embed_dim", 512),
            num_tokens=config.get("num_tokens", 64),
            num_heads=config.get("num_heads", 8),
            cond_dim=config.get("cond_dim", 2048),
            use_film=config.get("use_film", True),
            max_depth=config.get("max_depth", 24),
            dropout=config.get("dropout", 0.0),
            llm_model_name=config.get("llm_model_name", "facebook/opt-1.3b"),
            llm_frozen=config.get("llm_frozen", False),
            llm_lora=config.get("llm_lora"),
            instruction_encoder_model=config.get(
                "instruction_encoder_model", "facebook/opt-1.3b"
            ),
            device=config.get("device", "cuda"),
        )

    raise ValueError(
        f"Unknown projector type: {projector_type!r}. "
        "Valid choices: 'aadp', 'aadp_task_cond_stage1', "
        "'attention_conditioned_aadp', 'perceiver', 'medpruner', 'ctclip_stage2'."
    )
