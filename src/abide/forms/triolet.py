"""
Triolet form template.

An 8-line poem with ABaAabAB rhyme and line repetitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    Refrain,
    RhymeScheme,
    StanzaCount,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Triolet(Constraint):
    """
    Triolet: 8 lines with ABaAabAB rhyme and refrains.

    Structure:
    - 8 lines total, single stanza
    - Rhyme scheme: ABaAabAB (capitals = refrains)
    - Line 1 repeats at lines 4 and 7
    - Line 2 repeats at line 8
    - Only 5 original lines needed

    Famous example:
        "How great my grief, my joys how few" by Thomas Hardy

    Examples:
        >>> triolet = Triolet()
        >>> result = triolet.verify(poem)
    """

    name = "Triolet"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEME = "ABAAABAB"  # a/A and b/B rhyme together

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        refrain_threshold: float = 0.9,
        strict: bool = False,
    ) -> None:
        """
        Initialize triolet constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum score for rhymes
            refrain_threshold: Minimum similarity for refrains
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_threshold = rhyme_threshold
        self.refrain_threshold = refrain_threshold
        self.strict = strict

        # Structural constraints
        self._line_count = LineCount(8, weight=2.0)
        self._stanza_count = StanzaCount(1, weight=0.5)

        # Refrain A: line 0 repeats at lines 3 and 6 (1-indexed: 1, 4, 7)
        self._refrain_a = Refrain(
            reference_line=0,
            repeat_at=[3, 6],
            weight=2.0,
            threshold=refrain_threshold,
        )

        # Refrain B: line 1 repeats at line 7 (1-indexed: 2, 8)
        self._refrain_b = Refrain(
            reference_line=1,
            repeat_at=[7],
            weight=2.0,
            threshold=refrain_threshold,
        )

        # Rhyme scheme (A rhymes with A, B rhymes with B)
        self._rhyme_scheme = RhymeScheme(
            self.RHYME_SCHEME,
            weight=1.5,
            threshold=rhyme_threshold,
        )

        # Compose constraints
        self._constraint: Constraint
        if strict:
            self._constraint = And(
                [
                    self._line_count,
                    self._stanza_count,
                    self._refrain_a,
                    self._refrain_b,
                    self._rhyme_scheme,
                ]
            )
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 2.0),
                    (self._stanza_count, 0.5),
                    (self._refrain_a, 2.0),
                    (self._refrain_b, 2.0),
                    (self._rhyme_scheme, 1.5),
                ],
                threshold=0.6,
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
        return "Triolet: 8 lines with ABaAabAB rhyme, lines 1/4/7 and 2/8 repeat"
