from typing import List, Literal, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class InstructionEncoder(nn.Module):
    """Encodes clinical instruction text into a fixed-size vector for FiLM conditioning.

    Wraps any HuggingFace causal or encoder LM. Reduces the last hidden state
    to a single (B, C) representation via mean / last-token / CLS pooling.

    Compatible with (and tested against):
        "meta-llama/Llama-3.2-1B"
        "Qwen/Qwen2.5-1.5B"
        "mistralai/Mistral-7B-v0.3"
        "facebook/opt-125m"  (used in tests — no auth required)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        pooling: Literal["mean", "last", "cls"] = "mean",
        frozen: bool = True,
        max_length: int = 128,
        token: Optional[str] = None,
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "last", "cls"):
            raise ValueError(f"pooling must be 'mean', 'last', or 'cls', got '{pooling}'")

        self.pooling = pooling
        self.frozen = frozen
        self.max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=token, trust_remote_code=True
        )
        # GPT-style models have no dedicated pad token; fall back to eos.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModel.from_pretrained(
            model_name, token=token, trust_remote_code=True
        )

        if frozen:
            self._model.requires_grad_(False)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        """Hidden size C of the underlying language model."""
        return self._model.config.hidden_size

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of instruction strings.

        Args:
            texts: List of B instruction strings.

        Returns:
            etext: Tensor of shape (B, C).
        """
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if self.frozen:
            with torch.no_grad():
                outputs = self._model(**inputs)
        else:
            outputs = self._model(**inputs)

        last_hidden = outputs.last_hidden_state        # (B, L, C)
        attention_mask = inputs["attention_mask"]      # (B, L)
        return self._pool(last_hidden, attention_mask)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _pool(
        self, last_hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling == "mean":
            # Average over real (non-padding) token positions only.
            mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
            return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        if self.pooling == "last":
            # Index of the last real token per sequence.
            seq_len = attention_mask.sum(dim=1) - 1              # (B,)
            batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            return last_hidden[batch_idx, seq_len]               # (B, C)

        # pooling == "cls"
        return last_hidden[:, 0, :]                              # (B, C)


# ── Convenience function ──────────────────────────────────────────────────────


def encode_instruction(text: str, encoder: InstructionEncoder) -> torch.Tensor:
    """Encode a single instruction string into a (C,) vector.

    Wraps the batch interface of InstructionEncoder for single-sample inference.
    """
    return encoder([text]).squeeze(0)
