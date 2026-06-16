"""RaTEScore metric — radiology-specific text similarity.

Primary: ``ratescore`` package.
Fallback: ``bert_score`` with ``StanfordAIMI/RadBERT``.

If neither is installed, ``RaTEScore()`` raises ``ImportError``.
"""

from typing import Dict, List


def compute_ratescore(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute RaTEScore for a batch of report pairs.

    Args:
        predictions: Generated report strings (length B).
        references:  Ground-truth report strings (length B).

    Returns:
        ``{"ratescore_mean": float, "ratescore_std": float}``

    Raises:
        ImportError: If neither ``ratescore`` nor ``bert_score`` is installed.
    """
    return RaTEScore().compute(predictions, references)


class RaTEScore:
    """Holds the scoring model in memory across evaluation calls.

    Attempts to load backends in order:

    1. Native ``ratescore`` package (pip install ratescore).
    2. ``bert_score`` with ``StanfordAIMI/RadBERT`` as the radiology-domain model.

    Raises:
        ImportError: If neither backend is available.
    """

    _RADBERT = "StanfordAIMI/RadBERT"

    def __init__(self) -> None:
        self._backend: str = self._detect_backend()

    # ── Backend detection ─────────────────────────────────────────────────────

    @staticmethod
    def _detect_backend() -> str:
        try:
            import ratescore  # noqa: F401
            return "ratescore"
        except ImportError:
            pass
        try:
            import bert_score  # noqa: F401
            return "bert_score"
        except ImportError:
            pass
        raise ImportError(
            "RaTEScore requires either 'ratescore' or 'bert_score'.\n"
            "Install one of:\n"
            "  pip install ratescore\n"
            "  pip install bert-score"
        )

    # ── Compute ───────────────────────────────────────────────────────────────

    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Score a batch of prediction–reference pairs.

        Returns:
            ``{"ratescore_mean": float, "ratescore_std": float}``
        """
        import numpy as np

        if len(predictions) != len(references):
            raise ValueError(
                f"predictions and references must be the same length "
                f"({len(predictions)} != {len(references)})"
            )

        if self._backend == "ratescore":
            return self._compute_ratescore(predictions, references)
        else:
            return self._compute_bertscore(predictions, references)

    def _compute_ratescore(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        import numpy as np
        import ratescore

        scores = ratescore.compute(predictions, references)
        arr = np.array(scores, dtype=float)
        return {
            "ratescore_mean": float(arr.mean()),
            "ratescore_std": float(arr.std()),
        }

    def _compute_bertscore(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        import numpy as np
        from bert_score import BERTScorer

        scorer = BERTScorer(model_type=self._RADBERT, num_layers=9)
        # RadBERT's tokenizer has model_max_length ~ 1e30 (the HF "no limit" sentinel).
        # The Rust fast-tokenizer backend stores max_length as i64, so it overflows.
        # Cap it here before any encode call is made.
        for _attr in ("_tokenizer", "tokenizer"):
            _tok = getattr(scorer, _attr, None)
            if _tok is not None:
                _tok.model_max_length = 512
                break

        _, _, F = scorer.score(predictions, references, verbose=False)
        arr = F.numpy()
        return {
            "ratescore_mean": float(arr.mean()),
            "ratescore_std": float(arr.std()),
        }
