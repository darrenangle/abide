"""
Rondeau form template.

A 15-line poem with rentrement (refrain from opening).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    OpeningPhraseRefrain,
    Or,
    RhymeScheme,
    StanzaSizes,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Rondeau(Constraint):
    """
    Rondeau: 15 lines with AABBA AABR AABBAR rhyme and rentrement.

    Structure:
    - 15 lines in 3 stanzas (5-4-6 or presented as single block)
    - Rhyme scheme: AABBA AABR AABBAR
    - R = rentrement (first phrase of line 1, unrhymed)
    - Rentrement appears at lines 9 and 15
    - Only two rhyme sounds used (A and B)

    Famous example:
        "In Flanders Fields" by John McCrae (variant)

    Examples:
        >>> rondeau = Rondeau()
        >>> result = rondeau.verify(poem)
    """

    name = "Rondeau"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEME = "AABBAAABCAABBAD"
    LINE_COUNT = 15

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        refrain_threshold: float = 0.8,
        strict: bool = False,
    ) -> None:
        """
        Initialize rondeau constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum score for rhymes
            refrain_threshold: Minimum similarity for rentrement
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_threshold = rhyme_threshold
        self.refrain_threshold = refrain_threshold
        self.strict = strict
        self._line_count = LineCount(self.LINE_COUNT, weight=2.0)
        self._layout = Or(
            [
                StanzaSizes([5, 4, 6], weight=1.0),
                StanzaSizes([15], weight=1.0),
            ],
            weight=1.0,
        )
        self._rhyme = RhymeScheme(
            self.RHYME_SCHEME,
            weight=2.0,
            threshold=rhyme_threshold,
        )
        self._rentrement = OpeningPhraseRefrain(
            reference_line=0,
            repeat_at=[8, 14],
            weight=2.0,
            threshold=refrain_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._layout, 1.0),
            (self._rhyme, 2.0),
            (self._rentrement, 2.0),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([constraint for constraint, _ in constraints])
        else:
            self._constraint = WeightedSum(
                constraints,
                threshold=0.6,
                required_indices=list(range(len(constraints))),
            )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return "Rondeau: 15 lines with AABBA AABR AABBAR rhyme and rentrement"
