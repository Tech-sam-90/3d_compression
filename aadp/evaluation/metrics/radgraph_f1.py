"""RadGraph F1 metric for clinical entity/relation extraction evaluation.

Follows the CT-RATE evaluation protocol: score each (prediction, reference)
pair independently, then macro-average across the batch.

Supports both the standard RadGraph model and the larger RadGraph-XL model
(Delbrouck et al., "RadGraph-XL: A Large-Scale Expert-Annotated Dataset for
Clinical Information Extraction from Radiology Reports", 2024).

Default: RadGraph-XL (``use_xl=True``) to match the Argus Table 2 protocol.

KNOWN LIMITATION (in-process, this environment's ``transformers==5.14.1``):
``radgraph==0.1.18`` vendors a subset of AllenNLP that calls two HF tokenizer
methods transformers has since removed from the public API —
``.encode_plus()`` and ``.build_inputs_with_special_tokens()``. The former
could be safely shimmed (a pure pass-through to ``__call__``), but the latter
would require reimplementing model-specific special-token wrapping by hand,
which risks silently-wrong RadGraph-XL scores if the reimplementation is
subtly off — worse than an honest NaN for a benchmark metric.

RadGraph-XL therefore runs via
``aadp/evaluation/metrics/container_bridge.py`` — a separate, exactly-pinned
Apptainer container (torch==2.2.2, transformers==4.40.0, where both removed
methods still exist natively — no shim needed at all) built by
``scripts/build_metrics_container.sh`` — when that container is available.
radgraph's own declared requirements (``torch>=2.1.0``,
``transformers>=4.39.0``, no upper bound) are satisfied by GREEN's exact
pins, so one container serves both metrics. Falls back to the in-process
``RadGraphF1`` class (which raises in this environment) otherwise, so this
still degrades gracefully to NaN via ``compute_all.py`` for anyone who
hasn't built the container.
"""

from typing import Dict, List


def compute_radgraph_f1(
    predictions: List[str],
    references: List[str],
    use_xl: bool = True,
) -> Dict[str, float]:
    """Compute RadGraph F1 between predicted and reference radiology reports.

    Tries the pinned Apptainer container first when ``use_xl`` (see module
    docstring); falls back to the in-process ``RadGraphF1`` class otherwise.

    Args:
        predictions: List of B generated report strings.
        references:  List of B ground-truth report strings.
        use_xl:      When ``True`` (default), loads the RadGraph-XL checkpoint
                     to match the Argus evaluation protocol.  Set ``False`` to
                     use the standard RadGraph model.

    Returns:
        Dict with ``"precision"``, ``"recall"``, ``"f1"`` — macro-averaged.

    Raises:
        ValueError:   If ``predictions`` and ``references`` differ in length.
        ImportError:  If the ``radgraph`` package is not installed and the
                      container is unavailable.
        RuntimeError: If the container is available but the worker process
                      itself fails (as opposed to the metric computation
                      failing inside it).
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have the same length "
            f"({len(predictions)} != {len(references)})"
        )

    if use_xl:
        from aadp.evaluation.metrics.container_bridge import container_available, run_in_container

        if container_available():
            result = run_in_container("radgraph_xl", predictions, references)
            if "error" in result:
                raise RuntimeError(result["error"])
            return {
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "f1": float(result["f1"]),
            }

    scorer = RadGraphF1(use_xl=use_xl)
    return scorer.compute(predictions, references)


class RadGraphF1:
    """Holds the RadGraph model in memory for repeated evaluation calls.

    The underlying model is expensive to load; instantiate once and
    reuse via :meth:`compute`.

    Args:
        model_type: Passed directly to ``F1RadGraph``.  When ``None`` the value
                    is derived from ``use_xl``.
        use_xl:     When ``True`` (default), selects the RadGraph-XL checkpoint
                    (``model_type="radgraph-xl"``).  When ``False``, uses the
                    standard RadGraph model (``model_type=None``).

    Raises:
        ImportError: If the ``radgraph`` package is not installed.
    """

    def __init__(self, model_type=None, use_xl: bool = True) -> None:
        try:
            from radgraph import F1RadGraph as _F1RadGraph
        except ImportError as e:
            raise ImportError(
                "radgraph not installed. Run: pip install radgraph"
            ) from e

        if model_type is None:
            model_type = "radgraph-xl" if use_xl else None

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
