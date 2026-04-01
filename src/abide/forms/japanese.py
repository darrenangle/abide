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


def _count_syllable_pattern_violations(
    lines: tuple[str, ...] | list[str],
    expected: list[int],
    tolerance: int,
) -> int:
    """Count line-level syllable mismatches against an expected pattern."""
    from abide.primitives import count_line_syllables

    violations = 0
    for i, expected_syllables in enumerate(expected):
        if i >= len(lines):
            violations += 1
            continue

        actual_syllables = count_line_syllables(lines[i])
        if abs(actual_syllables - expected_syllables) > tolerance:
            violations += 1

    return violations


def _steep_violation_score(violations: int) -> float:
    """Convert a syllable violation count into the repo's steep partial-credit curve."""
    if violations == 0:
        return 1.0
    if violations == 1:
        return 0.5
    if violations == 2:
        return 0.25
    return 0.05


class Senryu(Constraint):
    """
    Senryū: 3 lines with 5-7-5 syllable pattern (same as haiku).

    Traditionally senryū focuses on human nature and foibles rather than
    seasonal nature imagery, but this verifier checks only the 5-7-5 structure.

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
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        stanza_result = self._stanza_count.verify(poem)
        syllables_result = self._syllables.verify(poem)
        expected = [5, 7, 5]
        violations = _count_syllable_pattern_violations(
            structure.lines,
            expected,
            self.syllable_tolerance,
        )
        syllable_score = _steep_violation_score(violations)
        scores = [
            (line_result.score, 1.0),
            (stanza_result.score, 0.5),
            (syllable_score, 2.0),
        ]
        total_weight = sum(w for _, w in scores)
        score = sum(s * w for s, w in scores) / total_weight

        passed = violations == 0 and line_result.passed and stanza_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=line_result.rubric + stanza_result.rubric + syllables_result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"violations": violations},
        )

    def describe(self) -> str:
        return "Senryū: 3 lines with 5-7-5 syllables"


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
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        stanza_result = self._stanza_count.verify(poem)
        syllables_result = self._syllables.verify(poem)
        expected = [5, 7, 7]
        violations = _count_syllable_pattern_violations(
            structure.lines,
            expected,
            self.syllable_tolerance,
        )
        syllable_score = _steep_violation_score(violations)
        scores = [
            (line_result.score, 1.0),
            (stanza_result.score, 0.5),
            (syllable_score, 2.0),
        ]
        total_weight = sum(w for _, w in scores)
        score = sum(s * w for s, w in scores) / total_weight

        passed = violations == 0 and line_result.passed and stanza_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=line_result.rubric + stanza_result.rubric + syllables_result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"violations": violations},
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
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        stanza_result = self._stanza_count.verify(poem)
        syllables_result = self._syllables.verify(poem)
        expected = [5, 7, 7, 5, 7, 7]
        violations = _count_syllable_pattern_violations(
            structure.lines,
            expected,
            self.syllable_tolerance,
        )
        syllable_score = _steep_violation_score(violations)
        scores = [
            (line_result.score, 1.0),
            (stanza_result.score, 0.5),
            (syllable_score, 2.0),
        ]
        total_weight = sum(w for _, w in scores)
        score = sum(s * w for s, w in scores) / total_weight

        passed = violations == 0 and line_result.passed and stanza_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=line_result.rubric + stanza_result.rubric + syllables_result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"violations": violations},
        )

    def describe(self) -> str:
        return "Sedoka: 6 lines with 5-7-7-5-7-7 syllables (two katauta)"
