"""multitask_sampler.py

Wraps a streaming ``CTRATEDataset`` and yields one
``{volume, instruction, target, patient_id}`` dict per sample, sampling across
all four instruction types (T1–T4).  Because the instruction type is drawn
per sample, a batch of N samples contains a mix of types, so FiLM's γ/β
networks see varied instruction embeddings in every training step instead of a
single constant instruction.

``CTRATEDataset`` is a streaming ``IterableDataset`` (it has no ``__getitem__``
or ``__len__``), so this wrapper is an ``IterableDataset`` too rather than a
map-style ``Dataset``.  It consumes the base dataset's
``(volume, report, label_dict, patient_id)`` tuples and does not change that
contract, so the raw ``CTRATEDataset`` remains usable directly (e.g. by the
VTCB evaluation harness).
"""

import random
from typing import Dict, Iterator, List, Optional

from torch.utils.data import IterableDataset

from aadp.data.instruction_builder import build_instructions


class MultiTaskCTRATEDataset(IterableDataset):
    """Yield ``(volume, instruction, target)`` triples across instruction types.

    Args:
        base_dataset:      A ``CTRATEDataset`` (or any iterable yielding
                           ``(volume, report, label_dict, patient_id)`` tuples).
        radgenome_dataset: Optional ``RadGenomeDataset``.  When provided, its
                           grounding text for the sample's ``patient_id`` is
                           passed to the localisation (T4) instruction.
    """

    def __init__(
        self,
        base_dataset,
        radgenome_dataset=None,
    ) -> None:
        self.base = base_dataset
        self.radgenome = radgenome_dataset

    def _radgenome_findings(self, patient_id: str):
        """Return the RadGenome findings for a volume id, or an empty list."""
        if self.radgenome is None:
            return []
        try:
            return self.radgenome.get_by_patient(patient_id)
        except Exception:
            return []

    def __iter__(self) -> Iterator[Dict]:
        for volume, report, label_dict, patient_id in self.base:
            findings = self._radgenome_findings(patient_id)

            # Join finding texts for the T4 localisation target …
            radgenome_text = (
                ". ".join(f.finding_text for f in findings) if findings else None
            )
            # … and union their grounded slice indices for the A4 attention loss.
            gt_slice_indices: Optional[List[int]] = None
            if findings:
                idx = sorted({i for f in findings for i in f.gt_slice_indices})
                gt_slice_indices = idx if idx else None

            pairs = build_instructions(
                report=report,
                label_dict=label_dict,
                radgenome_annotation=radgenome_text,
            )
            instruction, target = random.choice(pairs)
            yield {
                "volume": volume,
                "instruction": instruction,
                "target": target,
                "patient_id": patient_id,
                "gt_slice_indices": gt_slice_indices,
            }
