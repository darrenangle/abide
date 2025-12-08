"""
Additional Japanese poetic forms.

Senryū, Sedoka, and Katauta - related to haiku and tanka.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Senryu(Constraint):
    """
    Senryū: 3 lines with 5-7-5 syllable pattern (same as haiku).

    Unlike haiku, senryū focuses on human nature and foibles
    rather than nature and seasons. Structurally identical to haiku.

    Examples:
        >>> senryu = Senryu()
        >>> result = senryu.verify(poem)
    """

    name = "Senryū"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 0,
        strict: bool = True,
    ) -> None:
        """
        Initialize senryū constraint.

        Args:
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.strict = strict

        self._line_count = LineCount(3, weight=1.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            [5, 7, 5],
            weight=2.0,
            tolerance=syllable_tolerance,
        )

        self._constraint: Constraint
        if strict:
            self._constraint = And([self._line_count, self._stanza_count, self._syllables])
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
        return "Senryū: 3 lines with 5-7-5 syllables (human-focused haiku)"


class Katauta(Constraint):
    """
    Katauta: 3 lines with 5-7-7 syllable pattern.

    An ancient Japanese form, often called a "half poem" as it
    represents one half of a sedoka or mondō (question-answer poem).

    Examples:
        >>> katauta = Katauta()
        >>> result = katauta.verify(poem)
    """

    name = "Katauta"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 0,
        strict: bool = True,
    ) -> None:
        """
        Initialize katauta constraint.

        Args:
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.strict = strict

        self._line_count = LineCount(3, weight=1.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            [5, 7, 7],
            weight=2.0,
            tolerance=syllable_tolerance,
        )

        self._constraint: Constraint
        if strict:
            self._constraint = And([self._line_count, self._stanza_count, self._syllables])
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
        return "Katauta: 3 lines with 5-7-7 syllables"


class Sedoka(Constraint):
    """
    Sedoka: 6 lines with 5-7-7-5-7-7 syllable pattern.

    Two katauta (5-7-7) joined together. Often used as a
    question-and-answer or call-and-response poem.

    Examples:
        >>> sedoka = Sedoka()
        >>> result = sedoka.verify(poem)
    """

    name = "Sedoka"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 0,
        strict: bool = True,
    ) -> None:
        """
        Initialize sedoka constraint.

        Args:
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.strict = strict

        self._line_count = LineCount(6, weight=1.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            [5, 7, 7, 5, 7, 7],
            weight=2.0,
            tolerance=syllable_tolerance,
        )

        self._constraint: Constraint
        if strict:
            self._constraint = And([self._line_count, self._stanza_count, self._syllables])
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
        return "Sedoka: 6 lines with 5-7-7-5-7-7 syllables (two katauta)"
