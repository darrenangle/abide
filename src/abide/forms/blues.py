"""
Blues Poem form template.

A poem with AAB tercet structure and line repetition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    EndRhymePairs,
    GroupedStanzas,
    LinePairSimilarity,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class BluesPoem(Constraint):
    """
    Blues Poem: AAB tercets with line repetition.

    Structure:
    - Multiple tercets (3-line stanzas)
    - Line 1: statement/question
    - Line 2: repetition/variation of line 1
    - Line 3: response/resolution that rhymes with lines 1-2
    - Each stanza is self-contained

    Famous example:
        "The Weary Blues" by Langston Hughes

    Examples:
        >>> blues = BluesPoem()
        >>> result = blues.verify(poem)
    """

    name = "Blues Poem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        min_stanzas: int = 2,
        repetition_threshold: float = 0.6,
        rhyme_threshold: float = 0.5,
        strict: bool = False,
    ) -> None:
        """
        Initialize blues poem constraint.

        Args:
            weight: Relative weight for composition
            min_stanzas: Minimum number of AAB tercets
            repetition_threshold: Minimum similarity for L1-L2 repetition
            rhyme_threshold: Minimum score for A-B rhyme
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_stanzas = min_stanzas
        self.repetition_threshold = repetition_threshold
        self.rhyme_threshold = rhyme_threshold
        self.strict = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        group_sizes = self._infer_group_sizes(structure)
        offsets = self._group_offsets(group_sizes)

        repetition_pairs: list[tuple[int, int]] = []
        rhyme_pairs: list[tuple[int, int]] = []
        for index, size in enumerate(group_sizes):
            if size >= 3:
                base = offsets[index]
                repetition_pairs.append((base, base + 1))
                rhyme_pairs.extend(
                    [
                        (base, base + 2),
                        (base + 1, base + 2),
                    ]
                )

        constraints = [
            (
                GroupedStanzas(
                    3,
                    self.min_stanzas,
                    allow_single_block_chunking=False,
                    weight=1.5,
                ),
                1.5,
            ),
            (
                LinePairSimilarity(
                    repetition_pairs,
                    threshold=self.repetition_threshold,
                    weight=2.0,
                ),
                2.0,
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
                threshold=0.5,
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
        return f"Blues Poem: {self.min_stanzas}+ AAB tercets with line repetition"
