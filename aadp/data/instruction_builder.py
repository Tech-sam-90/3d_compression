"""instruction_builder.py

Converts a single CT-RATE sample into a list of ``(instruction, target_text)``
pairs covering the four training instruction types:

    T1 Generic report      — randomly sampled from T1_TEMPLATES
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

# ── T1 prompt pool ────────────────────────────────────────────────────────────
# One template is sampled per training example to diversify the instruction
# surface seen by the model.  Pass ``seed`` to ``build_instructions`` for
# reproducible selection (e.g. during validation scoring).
T1_TEMPLATES = [
    "Generate a radiology report for this chest CT scan.",
    "Write a detailed radiology report based on this CT volume.",
    "Describe all findings in this 3D chest CT scan.",
    "Provide a comprehensive radiology report for this CT scan.",
    "What does this chest CT show? Write a full radiology report.",
    "Summarise the radiological findings from this chest CT volume.",
    "Interpret this chest CT scan and produce a complete radiology report.",
    "Review this thoracic CT volume and generate a diagnostic radiology report.",
    "Examine the chest CT images and document all relevant radiological findings.",
    "Analyze this CT volume and prepare a structured radiology report.",
    "Identify any abnormalities in this chest CT and write a comprehensive report.",
    "Assess this chest CT examination and generate a detailed diagnostic report.",
    "Produce a formal radiology report describing the findings in this chest CT.",
    "Carefully evaluate this chest CT volume and summarize all significant findings.",
    "Generate a clinically appropriate radiology report from this chest CT examination.",
]

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
    seed: Optional[int] = None,
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
        seed:                 Optional integer seed for reproducible T1 template
                              selection.  Pass a fixed value during evaluation so
                              validation metrics are stable across runs.  When
                              ``None`` (default), the global ``random`` state is
                              used (training-time diversity).

    Returns:
        List of ``(instruction, target_text)`` tuples (length 2–5).
    """
    report = (report or "").strip()
    label_dict = label_dict or {}
    pairs: List[Tuple[str, str]] = []

    # ── T1 — Generic report generation (random template) ─────────────────────
    # Use a local RNG when seed is provided so the global random state is not
    # affected (important for reproducible T2/T3 sampling in other callers).
    _t1_rng = random.Random(seed) if seed is not None else random
    pairs.append((_t1_rng.choice(T1_TEMPLATES), report))

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
