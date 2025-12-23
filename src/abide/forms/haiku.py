"""
Haiku and Tanka form templates.

Japanese short-form poetry with strict syllable requirements.
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
        structure = self._ensure_structure(poem)

        # Get individual constraint results
        line_result = self._line_count.verify(poem)
        stanza_result = self._stanza_count.verify(poem)
        syllables_result = self._syllables.verify(poem)

        # Count violations in syllable pattern
        from abide.primitives import count_line_syllables

        expected = [5, 7, 5]
        violations = 0
        for i, exp_syl in enumerate(expected):
            if i < len(structure.lines):
                actual_syl = count_line_syllables(structure.lines[i])
                if abs(actual_syl - exp_syl) > self.syllable_tolerance:
                    violations += 1
            else:
                violations += 1  # Missing line is a violation

        # Apply steep penalty based on violations
        if violations == 0:
            syllable_score = 1.0
        elif violations == 1:
            syllable_score = 0.5
        elif violations == 2:
            syllable_score = 0.25
        else:
            syllable_score = 0.05

        # Combine scores
        scores = [
            (line_result.score, 1.0),
            (stanza_result.score, 0.5),
            (syllable_score, 2.0),
        ]
        total_weight = sum(w for _, w in scores)
        score = sum(s * w for s, w in scores) / total_weight

        passed = (
            score >= 0.7
            if not self.strict
            else (violations == 0 and line_result.passed and stanza_result.passed)
        )

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=line_result.rubric + stanza_result.rubric + syllables_result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"violations": violations},
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
        structure = self._ensure_structure(poem)

        # Get individual constraint results
        line_result = self._line_count.verify(poem)
        stanza_result = self._stanza_count.verify(poem)
        syllables_result = self._syllables.verify(poem)

        # Count violations in syllable pattern
        from abide.primitives import count_line_syllables

        expected = [5, 7, 5, 7, 7]
        violations = 0
        for i, exp_syl in enumerate(expected):
            if i < len(structure.lines):
                actual_syl = count_line_syllables(structure.lines[i])
                if abs(actual_syl - exp_syl) > self.syllable_tolerance:
                    violations += 1
            else:
                violations += 1  # Missing line is a violation

        # Apply steep penalty based on violations
        if violations == 0:
            syllable_score = 1.0
        elif violations == 1:
            syllable_score = 0.5
        elif violations == 2:
            syllable_score = 0.25
        else:
            syllable_score = 0.05

        # Combine scores
        scores = [
            (line_result.score, 1.0),
            (stanza_result.score, 0.5),
            (syllable_score, 2.0),
        ]
        total_weight = sum(w for _, w in scores)
        score = sum(s * w for s, w in scores) / total_weight

        passed = (
            score >= 0.7
            if not self.strict
            else (violations == 0 and line_result.passed and stanza_result.passed)
        )

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=line_result.rubric + stanza_result.rubric + syllables_result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"violations": violations},
        )

    def describe(self) -> str:
        return "Tanka: 5 lines with 5-7-5-7-7 syllable pattern"
