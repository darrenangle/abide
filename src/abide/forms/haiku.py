"""
Haiku and Tanka form templates.

Japanese short-form poetry with strict syllable requirements.
"""

from __future__ import annotations

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    StanzaCount,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)
from abide.primitives import PoemStructure


class Haiku(Constraint):
    """
    Haiku: 3 lines with 5-7-5 syllable pattern.

    A traditional Japanese form capturing a moment in nature.

    Examples:
        >>> haiku = Haiku()
        >>> result = haiku.verify("An old silent pond / A frog jumps into the pond / Splash! Silence again")
    """

    name = "Haiku"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 0,
        strict: bool = True,
    ) -> None:
        """
        Initialize haiku constraint.

        Args:
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            strict: If True, all constraints must pass; if False, uses weighted scoring
        """
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.strict = strict

        # Build constraints
        self._line_count = LineCount(3, weight=1.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            [5, 7, 5],
            weight=2.0,
            tolerance=syllable_tolerance,
        )

        if strict:
            self._constraint = And(
                [self._line_count, self._stanza_count, self._syllables]
            )
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 1.0),
                    (self._stanza_count, 0.5),
                    (self._syllables, 2.0),
                ],
                threshold=0.7,
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
        return "Haiku: 3 lines with 5-7-5 syllable pattern"


class Tanka(Constraint):
    """
    Tanka: 5 lines with 5-7-5-7-7 syllable pattern.

    An extension of haiku with two additional 7-syllable lines.

    Examples:
        >>> tanka = Tanka()
        >>> result = tanka.verify(poem)
    """

    name = "Tanka"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 0,
        strict: bool = True,
    ) -> None:
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.strict = strict

        self._line_count = LineCount(5, weight=1.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            [5, 7, 5, 7, 7],
            weight=2.0,
            tolerance=syllable_tolerance,
        )

        if strict:
            self._constraint = And(
                [self._line_count, self._stanza_count, self._syllables]
            )
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 1.0),
                    (self._stanza_count, 0.5),
                    (self._syllables, 2.0),
                ],
                threshold=0.7,
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
        return "Tanka: 5 lines with 5-7-5-7-7 syllable pattern"
