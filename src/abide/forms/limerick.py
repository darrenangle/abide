"""
Limerick form template.

A humorous five-line poem with AABBA rhyme scheme.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    RhymeScheme,
    StanzaCount,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Limerick(Constraint):
    """
    Limerick: 5 lines with AABBA rhyme scheme.

    Structure:
    - 5 lines in a single stanza
    - Rhyme scheme: AABBA
    - Lines 1, 2, 5 are longer (typically 7-10 syllables)
    - Lines 3, 4 are shorter (typically 5-7 syllables)

    Famous example:
        There once was a man from Nantucket
        Who kept all his cash in a bucket
        His daughter named Nan
        Ran away with a man
        And as for the bucket, Nantucket

    Examples:
        >>> limerick = Limerick()
        >>> result = limerick.verify(poem)
    """

    name = "Limerick"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEME = "AABBA"

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 2,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize limerick constraint.

        Args:
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhymes to count
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict = strict

        self._line_count = LineCount(5, weight=2.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._rhyme_scheme = RhymeScheme(
            self.RHYME_SCHEME,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        # Syllable pattern: lines 1,2,5 are longer (8 syllables)
        # Lines 3,4 are shorter (5 syllables)
        self._syllables = SyllablesPerLine(
            [8, 8, 5, 5, 8],
            weight=1.5,
            tolerance=syllable_tolerance,
        )

        self._constraint: Constraint
        if strict:
            self._constraint = And(
                [
                    self._line_count,
                    self._stanza_count,
                    self._rhyme_scheme,
                    self._syllables,
                ]
            )
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 2.0),
                    (self._stanza_count, 0.5),
                    (self._rhyme_scheme, 2.0),
                    (self._syllables, 1.5),
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
        return "Limerick: 5 lines with AABBA rhyme scheme (8-8-5-5-8 syllables)"
