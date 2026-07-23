"""CIDEr-D metric for radiology report evaluation.

CIDEr (Consensus-based Image Description Evaluation) rewards n-gram overlap
weighted by inverse document frequency over the reference corpus.  IDF is
computed over the reference set supplied at call time — no external corpus
file is required.

Requires:
    pip install pycocoevalcap
"""

from typing import Dict, List


def compute_cider(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute CIDEr-D between predicted and reference radiology reports.

    IDF weights are derived from ``references`` at call time, so the score
    adapts to the domain of the evaluation set (no external IDF file needed).

    Args:
        predictions: List of B generated report strings.
        references:  List of B ground-truth report strings.

    Returns:
        ``{"cider": float}`` — mean CIDEr-D score across the batch.

    Raises:
        ValueError:   If ``predictions`` and ``references`` differ in length.
        ImportError:  If ``pycocoevalcap`` is not installed.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have the same length "
            f"({len(predictions)} != {len(references)})"
        )

    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError as e:
        raise ImportError(
            "pycocoevalcap not installed. Run: pip install pycocoevalcap"
        ) from e

    # pycocoevalcap expects {image_id: [list_of_strings]}
    gts = {i: [ref]  for i, ref  in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}

    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return {"cider": float(score)}
