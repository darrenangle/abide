"""
Fibonacci poem (Fib) form template.

A modern form where syllables follow the Fibonacci sequence.
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


def fibonacci_sequence(n: int) -> list[int]:
    """Generate first n Fibonacci numbers starting from 1, 1."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]

    seq = [1, 1]
    for _ in range(n - 2):
        seq.append(seq[-1] + seq[-2])
    return seq


class FibonacciPoem(Constraint):
    """
    Fibonacci Poem (Fib): Syllables follow the Fibonacci sequence.

    Standard form is 6 lines with pattern: 1-1-2-3-5-8 syllables.
    Can be extended to more lines (1-1-2-3-5-8-13-21...).

    Examples:
        >>> fib = FibonacciPoem()  # Standard 6-line
        >>> result = fib.verify(poem)

        >>> fib = FibonacciPoem(lines=8)  # Extended
        >>> result = fib.verify(poem)
    """

    name = "Fibonacci Poem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        lines: int = 6,
        weight: float = 1.0,
        syllable_tolerance: int = 0,
        strict: bool = False,
    ) -> None:
        """
        Initialize Fibonacci poem constraint.

        Args:
            lines: Number of lines (determines Fibonacci length)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.num_lines = lines
        self.syllable_tolerance = syllable_tolerance
        self.strict_mode = strict

        self.syllable_pattern = fibonacci_sequence(lines)

        self._line_count = LineCount(lines, weight=2.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            self.syllable_pattern,
            weight=2.0,
            tolerance=syllable_tolerance,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 0.5),
            (self._syllables, 2.0),
        ]

        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.7)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "syllable_pattern": self.syllable_pattern,
                **result.details,
            },
        )

    def describe(self) -> str:
        pattern = "-".join(str(x) for x in self.syllable_pattern)
        return f"Fibonacci Poem: {self.num_lines} lines ({pattern} syllables)"


class ReverseFibonacci(Constraint):
    """
    Reverse Fibonacci Poem: Syllables follow reversed Fibonacci sequence.

    Standard 6-line pattern: 8-5-3-2-1-1 syllables.
    Creates a tapering effect opposite to the standard Fib.

    Examples:
        >>> rfib = ReverseFibonacci()
        >>> result = rfib.verify(poem)
    """

    name = "Reverse Fibonacci"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        lines: int = 6,
        weight: float = 1.0,
        syllable_tolerance: int = 0,
        strict: bool = False,
    ) -> None:
        """
        Initialize reverse Fibonacci constraint.

        Args:
            lines: Number of lines
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.num_lines = lines
        self.syllable_tolerance = syllable_tolerance
        self.strict_mode = strict

        self.syllable_pattern = list(reversed(fibonacci_sequence(lines)))

        self._line_count = LineCount(lines, weight=2.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            self.syllable_pattern,
            weight=2.0,
            tolerance=syllable_tolerance,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 0.5),
            (self._syllables, 2.0),
        ]

        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.7)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "syllable_pattern": self.syllable_pattern,
                **result.details,
            },
        )

    def describe(self) -> str:
        pattern = "-".join(str(x) for x in self.syllable_pattern)
        return f"Reverse Fibonacci: {self.num_lines} lines ({pattern} syllables)"


class DoubleFibonacci(Constraint):
    """
    Double Fibonacci: Standard Fib followed by Reverse Fib.

    Creates a diamond-like syllable structure:
    1-1-2-3-5-8-8-5-3-2-1-1 (12 lines)

    Examples:
        >>> dfib = DoubleFibonacci()
        >>> result = dfib.verify(poem)
    """

    name = "Double Fibonacci"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        base_lines: int = 6,
        weight: float = 1.0,
        syllable_tolerance: int = 0,
        strict: bool = False,
    ) -> None:
        """
        Initialize double Fibonacci constraint.

        Args:
            base_lines: Lines in each half (total = 2 * base_lines)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.base_lines = base_lines
        self.num_lines = base_lines * 2
        self.syllable_tolerance = syllable_tolerance
        self.strict_mode = strict

        fib = fibonacci_sequence(base_lines)
        self.syllable_pattern = fib + list(reversed(fib))

        self._line_count = LineCount(self.num_lines, weight=2.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            self.syllable_pattern,
            weight=2.0,
            tolerance=syllable_tolerance,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 0.5),
            (self._syllables, 2.0),
        ]

        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.7)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "syllable_pattern": self.syllable_pattern,
                **result.details,
            },
        )

    def describe(self) -> str:
        pattern = "-".join(str(x) for x in self.syllable_pattern)
        return f"Double Fibonacci: {self.num_lines} lines ({pattern} syllables)"
