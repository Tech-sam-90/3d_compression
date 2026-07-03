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

    raise ValueError(
        f"Unknown projector type: {projector_type!r}. "
        "Valid choices: 'aadp', 'aadp_task_cond_stage1', "
        "'attention_conditioned_aadp', 'perceiver', 'medpruner'."
    )
