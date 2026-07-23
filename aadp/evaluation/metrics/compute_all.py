"""compute_all_metrics — canonical entry point for the Argus Table 2 metrics.

Both ``aadp/evaluation/benchmarks/vtcb.py`` (used by ``scripts/evaluate.py``)
and ``scripts/vtcb_sweep_ctclip.py`` call this single function so the two
entry points can never drift into different key names or partially-computed
Avg. NLP composites.

Returns exactly these 8 keys:
    bleu4, rouge_l, meteor, cider, avg_nlp, green, ratescore, radgraph_xl_f1

Each metric is computed independently — one metric failing does not prevent
the others from being computed. A failure is logged at WARNING (never DEBUG,
so it cannot go silently unnoticed at the default logging level) and the
corresponding value is set to NaN rather than raising.
"""

import logging
import math
from typing import Dict, List

logger = logging.getLogger(__name__)

_METRIC_KEYS = (
    "bleu4", "rouge_l", "meteor", "cider",
    "avg_nlp", "green", "ratescore", "radgraph_xl_f1",
)


def _nan_result() -> Dict[str, float]:
    return {k: float("nan") for k in _METRIC_KEYS}


def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute all 8 Argus Table 2 metrics for a batch of report pairs.

    Args:
        predictions: Generated report strings (length B).
        references:  Ground-truth report strings (length B).

    Returns:
        Dict with keys ``bleu4, rouge_l, meteor, cider, avg_nlp, green,
        ratescore, radgraph_xl_f1`` — all present even when individual
        metrics fail (as NaN).

    Raises:
        ValueError: If ``predictions`` and ``references`` differ in length.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have the same length "
            f"({len(predictions)} != {len(references)})"
        )
    if not predictions:
        return _nan_result()

    scores: Dict[str, float] = {}

    # ── BLEU-4 + METEOR via nltk ────────────────────────────────────────────
    try:
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
        from nltk.translate.meteor_score import meteor_score

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("wordnet", quiet=True)

        smoothing = SmoothingFunction().method1
        refs_tok = [[r.split()] for r in references]
        hyps_tok = [p.split() for p in predictions]

        scores["bleu4"] = float(
            corpus_bleu(refs_tok, hyps_tok, smoothing_function=smoothing)
        )
        scores["meteor"] = float(
            sum(meteor_score([r.split()], p.split()) for r, p in zip(references, predictions))
            / len(predictions)
        )
    except Exception as exc:
        logger.warning("BLEU-4/METEOR scoring failed, returning NaN: %s", exc)
        scores.setdefault("bleu4", float("nan"))
        scores.setdefault("meteor", float("nan"))

    # ── ROUGE-L via rouge_score ──────────────────────────────────────────────
    try:
        from rouge_score import rouge_scorer as rs_module

        scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for ref, pred in zip(references, predictions)
        ]
        scores["rouge_l"] = float(sum(rouge_scores) / len(rouge_scores))
    except Exception as exc:
        logger.warning("ROUGE-L scoring failed, returning NaN: %s", exc)
        scores["rouge_l"] = float("nan")

    # ── CIDEr via pycocoevalcap ──────────────────────────────────────────────
    try:
        from aadp.evaluation.metrics.cider import compute_cider

        scores["cider"] = compute_cider(predictions, references)["cider"]
    except Exception as exc:
        logger.warning("CIDEr scoring failed, returning NaN: %s", exc)
        scores["cider"] = float("nan")

    # ── Avg. NLP = mean(BLEU-4, ROUGE-L, METEOR, CIDEr) — Argus Table 2 ─────
    nlp_keys = ("bleu4", "rouge_l", "meteor", "cider")
    nlp_vals = [scores[k] for k in nlp_keys]
    finite_vals = [v for v in nlp_vals if not math.isnan(v)]
    if len(finite_vals) == 4:
        scores["avg_nlp"] = float(sum(finite_vals) / 4)
    else:
        missing = [k for k, v in zip(nlp_keys, nlp_vals) if math.isnan(v)]
        logger.warning(
            "avg_nlp: %d/4 NLP sub-metric(s) failed (%s) — returning NaN "
            "rather than averaging a partial set.",
            4 - len(finite_vals), missing,
        )
        scores["avg_nlp"] = float("nan")

    # ── GREEN ─────────────────────────────────────────────────────────────
    try:
        from aadp.evaluation.metrics.green import compute_green

        green_val = compute_green(predictions, references)
        scores["green"] = float(green_val) if green_val is not None else float("nan")
    except Exception as exc:
        logger.warning("GREEN unavailable, returning NaN: %s", exc)
        scores["green"] = float("nan")

    # ── RaTEScore ─────────────────────────────────────────────────────────
    try:
        from aadp.evaluation.metrics.ratescore import RaTEScore

        rs_out = RaTEScore().compute(predictions, references)
        scores["ratescore"] = float(rs_out["ratescore_mean"])
    except Exception as exc:
        logger.warning("RaTEScore failed, returning NaN: %s", exc)
        scores["ratescore"] = float("nan")

    # ── RadGraph-XL F1 ────────────────────────────────────────────────────
    try:
        from aadp.evaluation.metrics.radgraph_f1 import compute_radgraph_f1

        rg_out = compute_radgraph_f1(predictions, references, use_xl=True)
        scores["radgraph_xl_f1"] = float(rg_out["f1"])
    except Exception as exc:
        logger.warning("RadGraph-XL F1 failed, returning NaN: %s", exc)
        scores["radgraph_xl_f1"] = float("nan")

    return scores
