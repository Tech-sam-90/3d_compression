"""GREEN metric for radiology report evaluation.

GREEN (Generative Radiology Report Evaluation and Error Notation) uses a
fine-tuned language model to assess clinical correctness of generated reports
across six error categories (false positive / false negative for findings,
locations, and severity).

Reference:
    Ostmeier et al., "GREEN: Generative Radiology Report Evaluation and Error
    Notation", MICCAI 2024.  Model: StanfordAIMI/GREEN-radllama2-7b.

Requires:
    pip install green-score
    GPU (returns None with a warning when no GPU is detected).
"""

import warnings
from typing import List, Optional


def compute_green(
    hypotheses: List[str],
    references: List[str],
) -> Optional[float]:
    """Compute mean GREEN score for a batch of report pairs.

    GREEN evaluates clinical correctness by scoring each (hypothesis, reference)
    pair on six error categories and returning a composite mean reward.

    Args:
        hypotheses: List of B generated report strings.
        references: List of B ground-truth report strings.

    Returns:
        Mean GREEN score (float in [0, 1]), or ``None`` if GPU is unavailable.

    Raises:
        ImportError: If ``green-score`` is not installed.
        ValueError:  If ``hypotheses`` and ``references`` differ in length.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"hypotheses and references must have the same length "
            f"({len(hypotheses)} != {len(references)})"
        )

    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required to run GREEN.") from e

    if not torch.cuda.is_available():
        warnings.warn(
            "GREEN requires a GPU but none is detected — returning None. "
            "Run on a CUDA-capable machine to obtain GREEN scores.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    try:
        from green_score import GREEN
    except ImportError as e:
        raise ImportError(
            "green-score not installed. Run: pip install green-score"
        ) from e

    model = GREEN(
        model_id_or_path="StanfordAIMI/GREEN-radllama2-7b",
        output_dir="green_output",
        batch_size=4,
        return_0_if_no_green_score=True,
        cuda=True,
    )
    mean_score, _, _, _, _ = model(references, hypotheses)
    return float(mean_score)
