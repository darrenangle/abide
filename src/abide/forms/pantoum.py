"""
Pantoum form template.

A poem of interlocking quatrains with line repetition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    GroupedStanzas,
    LinePairSimilarity,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Pantoum(Constraint):
    """
    Pantoum: quatrains with interlocking line repetitions.

    Structure:
    - Multiple quatrains (4-line stanzas), typically 4+
    - Line 2 of stanza N becomes line 1 of stanza N+1
    - Line 4 of stanza N becomes line 3 of stanza N+1
    - Final stanza: lines 1 and 3 of first stanza return
    - Rhyme scheme: ABAB per stanza (optional)

    Examples:
        >>> pantoum = Pantoum()
        >>> result = pantoum.verify(poem)
    """

    name = "Pantoum"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        min_stanzas: int = 4,
        refrain_threshold: float = 0.85,
        check_circular: bool = True,
        strict: bool = False,
    ) -> None:
        """
        Initialize pantoum constraint.

        Args:
            weight: Relative weight for composition
            min_stanzas: Minimum number of quatrains required
            refrain_threshold: Minimum similarity for line repetitions
            check_circular: Whether to check that final stanza closes the loop
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_stanzas = min_stanzas
        self.refrain_threshold = refrain_threshold
        self.check_circular = check_circular
        self.strict = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        group_sizes = self._infer_group_sizes(structure)
        offsets = self._group_offsets(group_sizes)

        interlock_pairs: list[tuple[int, int]] = []
        for index in range(len(group_sizes) - 1):
            if group_sizes[index] >= 4 and group_sizes[index + 1] >= 4:
                interlock_pairs.extend(
                    [
                        (offsets[index] + 1, offsets[index + 1]),
                        (offsets[index] + 3, offsets[index + 1] + 2),
                    ]
                )

        circular_pairs: list[tuple[int, int]] = []
        if (
            self.check_circular
            and len(group_sizes) >= 2
            and group_sizes[0] >= 4
            and group_sizes[-1] >= 4
        ):
            circular_pairs = [
                (0, offsets[-1] + 1),
                (2, offsets[-1] + 3),
            ]

        constraints: list[tuple[Constraint, float]] = [
            (
                GroupedStanzas(
                    4,
                    self.min_stanzas,
                    allow_single_block_chunking=True,
                    weight=2.0,
                ),
                2.0,
            ),
            (
                LinePairSimilarity(
                    interlock_pairs,
                    threshold=self.refrain_threshold,
                    weight=2.0,
                ),
                2.0,
            ),
        ]
        if self.check_circular:
            constraints.append(
                (
                    LinePairSimilarity(
                        circular_pairs,
                        threshold=self.refrain_threshold,
                        weight=1.5,
                    ),
                    1.5,
                )
            )

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
            },
        )

    def _infer_group_sizes(self, structure: PoemStructure) -> list[int]:
        if structure.stanza_count > 1:
            return list(structure.stanza_sizes)

        full_groups, remainder = divmod(structure.line_count, 4)
        sizes = [4] * full_groups
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
        return f"Pantoum: {self.min_stanzas}+ quatrains with interlocking line repetition"
