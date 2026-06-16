"""Tests for InstructionEncoder using facebook/opt-125m (no auth required)."""

import pytest
import torch

from aadp.data.instruction_encoder import InstructionEncoder, encode_instruction

_MODEL = "facebook/opt-125m"
_TEXTS = ["There is a lung nodule in the right lower lobe.", "Normal cardiac silhouette."]


# ── Fixtures — module-scoped so the model is downloaded only once ─────────────


@pytest.fixture(scope="module")
def frozen_encoder() -> InstructionEncoder:
    return InstructionEncoder(model_name=_MODEL, pooling="mean", frozen=True)


@pytest.fixture(scope="module")
def unfrozen_encoder() -> InstructionEncoder:
    return InstructionEncoder(model_name=_MODEL, pooling="mean", frozen=False)


# ── Output shape ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pooling", ["mean", "last", "cls"])
def test_forward_shape_batch(pooling: str) -> None:
    enc = InstructionEncoder(model_name=_MODEL, pooling=pooling, frozen=True)
    out = enc(_TEXTS)
    assert out.shape == (2, enc.output_dim), f"pooling={pooling}: got {out.shape}"


def test_forward_shape_single(frozen_encoder: InstructionEncoder) -> None:
    out = frozen_encoder(["lung opacity in left upper lobe"])
    assert out.shape == (1, frozen_encoder.output_dim)


# ── output_dim property ───────────────────────────────────────────────────────


def test_output_dim_matches_tensor_last_dim(frozen_encoder: InstructionEncoder) -> None:
    out = frozen_encoder(_TEXTS)
    assert frozen_encoder.output_dim == out.shape[-1]


def test_output_dim_is_int(frozen_encoder: InstructionEncoder) -> None:
    assert isinstance(frozen_encoder.output_dim, int)


# ── encode_instruction (single-string convenience) ────────────────────────────


def test_encode_instruction_shape(frozen_encoder: InstructionEncoder) -> None:
    vec = encode_instruction("pleural effusion on left", frozen_encoder)
    assert vec.shape == (frozen_encoder.output_dim,)


def test_encode_instruction_is_1d(frozen_encoder: InstructionEncoder) -> None:
    vec = encode_instruction("cardiomegaly", frozen_encoder)
    assert vec.dim() == 1


# ── Frozen / unfrozen gradient behaviour ─────────────────────────────────────


def test_frozen_encoder_no_requires_grad(frozen_encoder: InstructionEncoder) -> None:
    assert not any(p.requires_grad for p in frozen_encoder.parameters())


def test_unfrozen_encoder_has_requires_grad(unfrozen_encoder: InstructionEncoder) -> None:
    assert any(p.requires_grad for p in unfrozen_encoder.parameters())


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_empty_string_does_not_raise(frozen_encoder: InstructionEncoder) -> None:
    out = frozen_encoder([""])
    assert out.shape == (1, frozen_encoder.output_dim)


def test_batch_with_mixed_lengths(frozen_encoder: InstructionEncoder) -> None:
    texts = ["a", "a much longer clinical instruction about a finding in the right lower lobe"]
    out = frozen_encoder(texts)
    assert out.shape == (2, frozen_encoder.output_dim)


# ── Device ────────────────────────────────────────────────────────────────────


def test_output_is_on_cpu_by_default(frozen_encoder: InstructionEncoder) -> None:
    out = frozen_encoder(_TEXTS)
    assert out.device.type == "cpu"


# ── Output dtype ─────────────────────────────────────────────────────────────


def test_output_dtype_is_float(frozen_encoder: InstructionEncoder) -> None:
    out = frozen_encoder(_TEXTS)
    assert out.dtype in (torch.float32, torch.float16, torch.bfloat16)


# ── Invalid pooling ───────────────────────────────────────────────────────────


def test_invalid_pooling_raises() -> None:
    with pytest.raises(ValueError, match="pooling"):
        InstructionEncoder(model_name=_MODEL, pooling="max", frozen=True)  # type: ignore[arg-type]
