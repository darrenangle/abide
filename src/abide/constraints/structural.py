"""
Structural constraints for poem verification.

Line count, stanza count, stanza sizes, syllables per line, etc.
"""

from __future__ import annotations

import math
from typing import Sequence

from abide.constraints.base import Constraint, NumericConstraint
from abide.constraints.types import (
    BoundType,
    ConstraintType,
    NumericBound,
    RubricItem,
    VerificationResult,
)
from abide.primitives import PoemStructure, count_line_syllables


class LineCount(NumericConstraint):
    """
    Constraint on the number of lines in a poem.

    Examples:
        >>> constraint = LineCount(14)  # Exactly 14 lines (sonnet)
        >>> constraint = LineCount(NumericBound.at_least(10))  # At least 10
    """

    name = "Line Count"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        bound: int | NumericBound,
        weight: float = 1.0,
        sigma: float = 2.0,
    ) -> None:
        if isinstance(bound, int):
            bound = NumericBound.exact(bound)
        super().__init__(bound, weight, sigma)

    def _get_actual_value(self, structure: PoemStructure) -> int:
        return structure.line_count

    def _get_criterion_name(self) -> str:
        return "lines"


class StanzaCount(NumericConstraint):
    """
    Constraint on the number of stanzas in a poem.

    Examples:
        >>> constraint = StanzaCount(6)  # Exactly 6 stanzas (villanelle)
        >>> constraint = StanzaCount(NumericBound.at_least(2))
    """

    name = "Stanza Count"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        bound: int | NumericBound,
        weight: float = 1.0,
        sigma: float = 1.0,
    ) -> None:
        if isinstance(bound, int):
            bound = NumericBound.exact(bound)
        super().__init__(bound, weight, sigma)

    def _get_actual_value(self, structure: PoemStructure) -> int:
        return structure.stanza_count

    def _get_criterion_name(self) -> str:
        return "stanzas"


class StanzaSizes(Constraint):
    """
    Constraint on the size (line count) of each stanza.

    Examples:
        >>> constraint = StanzaSizes([3, 3, 3, 3, 3, 4])  # Villanelle
        >>> constraint = StanzaSizes([6, 6, 6, 6, 6, 6, 3])  # Sestina
    """

    name = "Stanza Sizes"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        expected_sizes: Sequence[int],
        weight: float = 1.0,
        sigma: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.expected_sizes = tuple(expected_sizes)
        self.sigma = sigma

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        actual_sizes = structure.stanza_sizes

        rubric: list[RubricItem] = []
        scores: list[float] = []

        # First check stanza count
        if len(actual_sizes) != len(self.expected_sizes):
            stanza_passed = False
            stanza_score = 0.0
            # Partial credit based on how many stanzas match
            min_len = min(len(actual_sizes), len(self.expected_sizes))
            if min_len > 0:
                matching = sum(
                    1
                    for i in range(min_len)
                    if actual_sizes[i] == self.expected_sizes[i]
                )
                stanza_score = matching / len(self.expected_sizes)
        else:
            stanza_passed = True
            stanza_score = 1.0

        rubric.append(
            RubricItem(
                criterion="Stanza count",
                expected=str(len(self.expected_sizes)),
                actual=str(len(actual_sizes)),
                score=stanza_score,
                passed=len(actual_sizes) == len(self.expected_sizes),
            )
        )
        scores.append(stanza_score)

        # Check each stanza size
        for i, expected in enumerate(self.expected_sizes):
            if i < len(actual_sizes):
                actual = actual_sizes[i]
                passed = actual == expected
                if passed:
                    score = 1.0
                else:
                    diff = abs(actual - expected)
                    score = math.exp(-0.5 * (diff / self.sigma) ** 2)
            else:
                actual = 0
                passed = False
                score = 0.0

            rubric.append(
                RubricItem(
                    criterion=f"Stanza {i + 1} size",
                    expected=str(expected),
                    actual=str(actual),
                    score=score,
                    passed=passed,
                )
            )
            scores.append(score)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "expected_sizes": list(self.expected_sizes),
                "actual_sizes": list(actual_sizes),
            },
        )

    def describe(self) -> str:
        sizes_str = ", ".join(str(s) for s in self.expected_sizes)
        return f"Has stanza sizes [{sizes_str}]"


class SyllablesPerLine(Constraint):
    """
    Constraint on syllable count for each line.

    Examples:
        >>> constraint = SyllablesPerLine([5, 7, 5])  # Haiku
        >>> constraint = SyllablesPerLine([10] * 14)  # Sonnet (iambic pentameter)
    """

    name = "Syllables Per Line"
    constraint_type = ConstraintType.PROSODIC

    def __init__(
        self,
        expected_syllables: Sequence[int],
        weight: float = 1.0,
        sigma: float = 1.0,
        tolerance: int = 0,
    ) -> None:
        """
        Initialize syllable constraint.

        Args:
            expected_syllables: Expected syllable count for each line
            weight: Relative weight for composition
            sigma: Standard deviation for Gaussian decay
            tolerance: Allow +/- this many syllables and still pass
        """
        super().__init__(weight)
        self.expected_syllables = tuple(expected_syllables)
        self.sigma = sigma
        self.tolerance = tolerance

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        for i, expected in enumerate(self.expected_syllables):
            if i < len(structure.lines):
                line = structure.lines[i]
                actual = count_line_syllables(line)

                diff = abs(actual - expected)
                passed = diff <= self.tolerance

                if passed:
                    score = 1.0
                else:
                    # Gaussian decay from tolerance boundary
                    adjusted_diff = diff - self.tolerance
                    score = math.exp(-0.5 * (adjusted_diff / self.sigma) ** 2)
            else:
                actual = 0
                passed = False
                score = 0.0

            rubric.append(
                RubricItem(
                    criterion=f"Line {i + 1} syllables",
                    expected=str(expected) + (f" ± {self.tolerance}" if self.tolerance else ""),
                    actual=str(actual),
                    score=score,
                    passed=passed,
                    explanation=structure.lines[i][:50] + "..." if i < len(structure.lines) and len(structure.lines[i]) > 50 else (structure.lines[i] if i < len(structure.lines) else ""),
                )
            )
            scores.append(score)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
        )

    def describe(self) -> str:
        if len(set(self.expected_syllables)) == 1:
            return f"Has {self.expected_syllables[0]} syllables per line"
        syllables_str = ", ".join(str(s) for s in self.expected_syllables[:5])
        if len(self.expected_syllables) > 5:
            syllables_str += ", ..."
        return f"Has syllable pattern [{syllables_str}]"


class TotalSyllables(NumericConstraint):
    """
    Constraint on total syllable count in poem.
    """

    name = "Total Syllables"
    constraint_type = ConstraintType.PROSODIC

    def __init__(
        self,
        bound: int | NumericBound,
        weight: float = 1.0,
        sigma: float = 5.0,
    ) -> None:
        if isinstance(bound, int):
            bound = NumericBound.exact(bound)
        super().__init__(bound, weight, sigma)

    def _get_actual_value(self, structure: PoemStructure) -> int:
        return sum(count_line_syllables(line) for line in structure.lines)

    def _get_criterion_name(self) -> str:
        return "total syllables"
