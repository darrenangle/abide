"""
Terza Rima form template.

A poem with interlocking tercets in ABA BCB CDC pattern.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    EndRhymePairs,
    GroupedStanzas,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class TerzaRima(Constraint):
    """
    Terza Rima: interlocking tercets with ABA BCB CDC chain rhyme.

    Structure:
    - Multiple tercets (3-line stanzas)
    - Rhyme scheme: ABA BCB CDC DED...
    - Middle line of each tercet rhymes with outer lines of next
    - Typically ends with single line or couplet

    Famous example:
        "Ode to the West Wind" by Percy Bysshe Shelley

    Examples:
        >>> terza_rima = TerzaRima()
        >>> result = terza_rima.verify(poem)
    """

    name = "Terza Rima"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        min_tercets: int = 3,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize terza rima constraint.

        Args:
            weight: Relative weight for composition
            min_tercets: Minimum number of tercets required
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_tercets = min_tercets
        self.rhyme_threshold = rhyme_threshold
        self.strict = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        group_sizes = self._infer_group_sizes(structure)
        offsets = self._group_offsets(group_sizes)
        full_tercets = [index for index, size in enumerate(group_sizes) if size == 3]

        rhyme_pairs: list[tuple[int, int]] = []
        for group_index in full_tercets:
            base = offsets[group_index]
            rhyme_pairs.append((base, base + 2))

        for current_group, next_group in pairwise(full_tercets):
            base_current = offsets[current_group]
            base_next = offsets[next_group]
            rhyme_pairs.extend(
                [
                    (base_current + 1, base_next),
                    (base_current + 1, base_next + 2),
                ]
            )

        if group_sizes and group_sizes[-1] in {1, 2} and full_tercets:
            tail_offset = offsets[-1]
            previous_middle = offsets[full_tercets[-1]] + 1
            rhyme_pairs.append((previous_middle, tail_offset))
            if group_sizes[-1] == 2:
                rhyme_pairs.append((previous_middle, tail_offset + 1))

        constraints = [
            (
                GroupedStanzas(
                    3,
                    self.min_tercets,
                    allow_single_block_chunking=True,
                    allowed_tail_sizes=(1, 2),
                    weight=1.5,
                ),
                1.5,
            ),
            (
                EndRhymePairs(
                    rhyme_pairs,
                    threshold=self.rhyme_threshold,
                    weight=2.0,
                ),
                2.0,
            ),
        ]

        constraint: Constraint
        if self.strict:
            constraint = And([item for item, _ in constraints])
        else:
            constraint = WeightedSum(
                constraints,
                threshold=0.6,
                required_indices=list(range(len(constraints))),
            )

        result = constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                **result.details,
                "group_sizes": group_sizes,
                "tercet_count": len(full_tercets),
            },
        )

    def _infer_group_sizes(self, structure: PoemStructure) -> list[int]:
        if structure.stanza_count > 1:
            return list(structure.stanza_sizes)

        full_groups, remainder = divmod(structure.line_count, 3)
        sizes = [3] * full_groups
        if remainder:
            sizes.append(remainder)
        return sizes

    def _group_offsets(self, group_sizes: list[int]) -> list[int]:
        offsets: list[int] = []
        current = 0
        for size in group_sizes:
            offsets.append(current)
            current += size
        return offsets

    def describe(self) -> str:
        return f"Terza Rima: {self.min_tercets}+ tercets with ABA BCB chain rhyme"
