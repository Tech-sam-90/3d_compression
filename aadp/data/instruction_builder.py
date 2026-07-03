"""instruction_builder.py

Converts a single CT-RATE sample into a list of ``(instruction, target_text)``
pairs covering the four training instruction types:

    T1 Generic report      — "Generate a radiology report for this CT scan."
    T2 Entity-conditioned   — "Describe the findings related to {entity}."
    T3 Classification       — "Is there evidence of {abnormality}? Answer yes or no."
    T4 Localisation         — "Describe the location and distribution of findings."

Entity extraction is a lightweight keyword match against an anatomy/finding
vocabulary — a dependency-free stand-in for RadGraph entity parsing that keeps
training self-contained.  The abnormality names for T3 come directly from the
CT-RATE ``label_dict`` keys, so they always match the dataset's 18 labels.
"""

import random
from typing import Dict, List, Optional, Tuple

# Anatomy / finding vocabulary for the T2 entity-conditioned instructions.
ANATOMY_KEYWORDS = [
    "lung", "lobe", "pleura", "effusion", "nodule", "mass", "opacity",
    "consolidation", "atelectasis", "pneumonia", "heart", "aorta",
    "trachea", "mediastinum", "lymph node", "liver", "spleen", "kidney",
    "adrenal", "bone", "rib", "vertebra", "chest wall",
]


def extract_entities_from_report(report: str) -> List[str]:
    """Return anatomy/finding terms that appear in ``report`` (keyword match)."""
    report_lower = report.lower()
    found = [kw for kw in ANATOMY_KEYWORDS if kw in report_lower]
    return found if found else ["lung"]  # fallback so T2 is always constructible


def sentences_containing(report: str, entity: str) -> str:
    """Return the sentences of ``report`` that mention ``entity``.

    Falls back to the full report when no sentence matches, so the target is
    never empty.
    """
    sentences = [
        s.strip() for s in report.replace("\n", " ").split(".") if s.strip()
    ]
    matching = [s for s in sentences if entity.lower() in s.lower()]
    return ". ".join(matching) + "." if matching else report


def _prettify_label(label: str) -> str:
    """Lower-case a CT-RATE label name for natural-language instructions."""
    return label.strip().lower()


def build_instructions(
    report: str,
    label_dict: Dict[str, int],
    radgenome_annotation: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Build ``(instruction, target)`` pairs for one CT-RATE sample.

    Always includes T1 (generic report).  T2 is added whenever a report is
    present.  T3 adds up to one positive and one negative classification pair
    drawn from ``label_dict``.  T4 is added only when a RadGenome grounding
    annotation is supplied.

    Args:
        report:               Full CT-RATE report text.
        label_dict:           18 CT-RATE binary abnormality labels
                              (``{name: 0|1}``).
        radgenome_annotation: Optional grounding/localisation text; enables T4.

    Returns:
        List of ``(instruction, target_text)`` tuples (length 2–5).
    """
    report = (report or "").strip()
    label_dict = label_dict or {}
    pairs: List[Tuple[str, str]] = []

    # ── T1 — Generic report generation ────────────────────────────────────────
    pairs.append(
        ("Generate a radiology report for this CT scan.", report)
    )

    # ── T2 — Entity-conditioned ───────────────────────────────────────────────
    if report:
        entity = random.choice(extract_entities_from_report(report))
        pairs.append((
            f"Describe the findings related to {entity} in this CT scan.",
            sentences_containing(report, entity),
        ))

    # ── T3 — Classification (one positive + one negative when available) ──────
    positives = [k for k, v in label_dict.items() if v == 1]
    negatives = [k for k, v in label_dict.items() if v == 0]
    if positives:
        abn = _prettify_label(random.choice(positives))
        pairs.append((
            f"Is there evidence of {abn} in this scan? Answer yes or no.",
            "Yes.",
        ))
    if negatives:
        abn = _prettify_label(random.choice(negatives))
        pairs.append((
            f"Is there evidence of {abn} in this scan? Answer yes or no.",
            "No.",
        ))

    # ── T4 — Localisation (RadGenome grounding text) ──────────────────────────
    if radgenome_annotation:
        pairs.append((
            "Describe the location and distribution of findings in this CT scan.",
            radgenome_annotation,
        ))

    return pairs
