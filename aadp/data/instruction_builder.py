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
    # ── M3D Caption_templates (M3D/LaMed/src/dataset/prompt_templates.py) ──────
    # Added for phrasing diversity — M3D's pool is shorter/terser and mostly
    # interrogative, complementing the formal/declarative style above. One
    # exact internal duplicate ("Please caption this medical scan with
    # findings.") was present twice in M3D's source list; only kept once.
    # No overlap (exact or near) with the templates above.
    "Can you provide a caption consists of findings for this medical image?",
    "Describe the findings of the medical image you see.",
    "Please caption this medical scan with findings.",
    "What is the findings of this image?",
    "Describe this medical scan with findings.",
    "Please write a caption consists of findings for this image.",
    "Can you summarize with findings the images presented?",
    "Please caption this scan with findings.",
    "Please provide a caption consists of findings for this medical image.",
    "Can you provide a summary consists of findings of this radiograph?",
    "What are the findings presented in this medical scan?",
    "Please write a caption consists of findings for this scan.",
    "Can you provide a description consists of findings of this medical scan?",
    "Can you provide a caption consists of findings for this medical scan?",
    "Please generate a medical report based on this image.",
    "Can you generate a diagnose report from this image.",
    "Could you analyze and provide a caption for the findings in this medical image?",
    "Please describe the observations depicted in this medical scan.",
    "Can you summarize the findings of this image in a caption?",
    "What are the significant findings in this medical image?",
    "Please provide a detailed caption outlining the findings of this image.",
    "Could you interpret and describe the findings shown in this medical scan?",
    "What conclusions can you draw from the observations in this image?",
    "Please write a descriptive caption based on the findings in this scan.",
    "What key findings can you identify from examining this medical image?",
    "Could you generate a detailed report based on the observations in this image?",
    "Can you provide a diagnosis based on the findings in this image?",
    "Please generate a comprehensive report summarizing the findings in this image.",
    "Caption the findings in this medical image?",
    "Describe the findings you see.",
    "Caption this medical scan's findings.",
    "What are the findings here?",
    "Describe these findings.",
    "Summarize the findings in these images.",
    "Caption this scan's findings.",
    "Provide a caption for this medical image's findings.",
    "Summarize the findings of this radiograph.",
    "What findings are presented in this scan?",
    "Describe this scan's findings.",
    "Generate a medical report based on this image.",
    "Can you provide a diagnosis based on this image?",
]

# Anatomy / finding vocabulary for the T2 entity-conditioned instructions.
ANATOMY_KEYWORDS = [
    "lung", "lobe", "pleura", "effusion", "nodule", "mass", "opacity",
    "consolidation", "atelectasis", "pneumonia", "heart", "aorta",
    "trachea", "mediastinum", "lymph node", "liver", "spleen", "kidney",
    "adrenal", "bone", "rib", "vertebra", "chest wall",
]

# Terms too generic to produce a scan-varying T2 target — every CT-RATE
# report mentions "lung"/"trachea"/etc. somewhere, so picking one of these
# as the T2 entity makes sentences_containing() match nearly the same
# boilerplate sentences regardless of scan content. Confirmed the concrete
# failure mode this fixes: docs/QUALITATIVE_EVAL_V2.md's cross-scan T2
# check (entity="lung" on all 5 scans) found 1/5 distinct outputs, mean
# cosine 0.98 — "lung" alone doesn't disambiguate scans.
GENERIC_ENTITIES = {
    "lung", "lungs", "chest", "scan", "ct", "trachea",
    "bronchi", "bronchus", "parenchyma", "examination",
    "structure", "structures", "finding", "findings",
}


def extract_entities_from_report(report: str) -> List[str]:
    """Return anatomy/finding terms that appear in ``report`` (keyword
    match), excluding GENERIC_ENTITIES."""
    report_lower = report.lower()
    found = [kw for kw in ANATOMY_KEYWORDS if kw in report_lower]
    found = [kw for kw in found if kw not in GENERIC_ENTITIES]
    return found if found else ["lobe"]  # fallback so T2 is always constructible


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
