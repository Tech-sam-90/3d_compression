"""RadGraph F1 metric for clinical entity/relation extraction evaluation.

Follows the CT-RATE evaluation protocol: score each (prediction, reference)
pair independently, then macro-average across the batch.
"""

from typing import Dict, List


def compute_radgraph_f1(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute RadGraph F1 between predicted and reference radiology reports.

    Args:
        predictions: List of B generated report strings.
        references:  List of B ground-truth report strings.

    Returns:
        Dict with ``"precision"``, ``"recall"``, ``"f1"`` — macro-averaged.

    Raises:
        ImportError: If the ``radgraph`` package is not installed.
    """
    scorer = RadGraphF1()
    return scorer.compute(predictions, references)


class RadGraphF1:
    """Holds the RadGraph model in memory for repeated evaluation calls.

    The underlying AllenNLP model is expensive to load; instantiate once and
    reuse via :meth:`compute`.

    Args:
        model_type: Passed to ``F1RadGraph``.  ``None`` uses the default model.

    Raises:
        ImportError: If the ``radgraph`` package is not installed.
    """

    def __init__(self, model_type=None) -> None:
        try:
            from radgraph import F1RadGraph as _F1RadGraph
        except ImportError as e:
            raise ImportError(
                "radgraph not installed. Run: pip install radgraph"
            ) from e

        self._scorer = _F1RadGraph(reward_level="all", model_type=model_type)

    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Score a batch of prediction–reference pairs.

        Args:
            predictions: Generated report strings (length B).
            references:  Ground-truth report strings (length B).

        Returns:
            ``{"precision": float, "recall": float, "f1": float}``
            — macro-averaged across the batch.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions and references must have the same length "
                f"({len(predictions)} != {len(references)})"
            )

        # F1RadGraph.forward(refs, hyps) → (mean_reward, reward_list, hyp_annots, ref_annots)
        # With reward_level="all", mean_reward = (mean_precision, mean_recall, mean_f1)
        mean_reward, _, _, _ = self._scorer(refs=references, hyps=predictions)
        precision, recall, f1 = mean_reward

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
