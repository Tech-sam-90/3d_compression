"""GREEN metric for radiology report evaluation.

GREEN (Generative Radiology Report Evaluation and Error Notation) uses a
fine-tuned language model to assess clinical correctness of generated reports
across six error categories (false positive / false negative for findings,
locations, and severity).

Reference:
    Ostmeier et al., "GREEN: Generative Radiology Report Evaluation and Error
    Notation", MICCAI 2024.  Model: StanfordAIMI/GREEN-radllama2-7b.

KNOWN LIMITATION (in-process, this environment's transformers==5.14.1):
green-score hard-imports HF `datasets`, which needs pyarrow. This cluster's
custom Python builds report a bare `linux_x86_64` platform tag rather than a
standard `manylinux*` tag, so real PyPI wheels for pyarrow can't be
installed into any isolated venv here — confirmed via extensive testing.
GREEN therefore runs via aadp/evaluation/metrics/container_bridge.py — a
separate, exactly-pinned Apptainer container (torch==2.2.2,
transformers==4.40.0, matching upstream) built by
scripts/build_metrics_container.sh — when that container is available.
Falls back to the in-process attempt (which raises ImportError here)
otherwise, so this still degrades gracefully to NaN via compute_all.py for
anyone who hasn't built the container.

Requires:
    Container: run scripts/build_metrics_container.sh once (login node).
    In-process fallback: pip install green-score + a working `datasets`.
    GPU either way (returns None with a warning when no GPU is detected).
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

    Tries the pinned Apptainer container first (see module docstring);
    falls back to an in-process attempt if the container isn't built.

    Args:
        hypotheses: List of B generated report strings.
        references: List of B ground-truth report strings.

    Returns:
        Mean GREEN score (float in [0, 1]), or ``None`` if GPU is unavailable.

    Raises:
        ImportError: If neither the container nor an in-process
            ``green-score`` install is available.
        ValueError:  If ``hypotheses`` and ``references`` differ in length.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"hypotheses and references must have the same length "
            f"({len(hypotheses)} != {len(references)})"
        )

    from aadp.evaluation.metrics.container_bridge import container_available, run_in_container

    if container_available():
        result = run_in_container("green", hypotheses, references)
        if "error" in result:
            warnings.warn(result["error"], RuntimeWarning, stacklevel=2)
            return None
        return float(result["score"])

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

    model = GREEN(model_name="StanfordAIMI/GREEN-radllama2-7b", output_dir="green_output")
    mean_score, _, _, _, _ = model(references, hypotheses)
    return float(mean_score)
