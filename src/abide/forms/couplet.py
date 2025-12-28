"""
Couplet form templates.

Couplets are two-line units that rhyme (AA).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    RhymeScheme,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Couplet(Constraint):
    """
    Generic couplet: 2 lines that rhyme.

    Examples:
        >>> couplet = Couplet()
        >>> result = couplet.verify("True wit is nature to advantage dressed,\\nWhat oft was thought, but ne'er so well expressed.")
    """

    name = "Couplet"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        syllables_per_line: int = 10,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize couplet constraint.

        Args:
            syllables_per_line: Syllables per line
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhyme
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.syllables = syllables_per_line
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict = strict

        self._line_count = LineCount(2, weight=2.0)
        self._syllables = SyllablesPerLine(
            [syllables_per_line] * 2,
            weight=1.0,
            tolerance=syllable_tolerance,
        )
        self._rhyme_scheme = RhymeScheme(
            "AA",
            weight=1.5,
            threshold=rhyme_threshold,
        )

        self._constraint: Constraint
        if strict:
            self._constraint = And([self._line_count, self._syllables, self._rhyme_scheme])
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 2.0),
                    (self._syllables, 1.0),
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
        return f"Couplet: 2 rhyming lines ({self.syllables} syllables each)"


class HeroicCouplet(Couplet):
    """
    Heroic couplet: 2 lines of iambic pentameter that rhyme (AA).

    The form of Pope, Dryden, and much 18th-century English poetry.
    Often end-stopped with a complete thought.

    Examples:
        >>> couplet = HeroicCouplet()
        >>> result = couplet.verify(poem)
    """

    name = "Heroic Couplet"

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        super().__init__(
            syllables_per_line=10,
            weight=weight,
            syllable_tolerance=syllable_tolerance,
            rhyme_threshold=rhyme_threshold,
            strict=strict,
        )

    def describe(self) -> str:
        return "Heroic Couplet: 2 lines of iambic pentameter, rhyming AA"


class ShortCouplet(Couplet):
    """
    Short couplet: 2 lines of iambic tetrameter that rhyme (AA).

    Also known as octosyllabic couplet. Common in light verse.

    Examples:
        >>> couplet = ShortCouplet()
        >>> result = couplet.verify(poem)
    """

    name = "Short Couplet"

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        super().__init__(
            syllables_per_line=8,
            weight=weight,
            syllable_tolerance=syllable_tolerance,
            rhyme_threshold=rhyme_threshold,
            strict=strict,
        )

    def describe(self) -> str:
        return "Short Couplet: 2 lines of iambic tetrameter, rhyming AA"


class Elegiac(Constraint):
    """
    Elegiac couplet: Hexameter + pentameter lines (classical form).

    Traditional form for Greek/Latin elegies and epigrams.
    Adapted here as 12 + 10 syllables.

    Examples:
        >>> couplet = Elegiac()
        >>> result = couplet.verify(poem)
    """

    name = "Elegiac Couplet"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict = strict

        self._line_count = LineCount(2, weight=2.0)
        # Hexameter + pentameter
        self._syllables = SyllablesPerLine(
            [12, 10],
            weight=1.0,
            tolerance=syllable_tolerance,
        )
        self._rhyme_scheme = RhymeScheme(
            "AA",
            weight=1.5,
            threshold=rhyme_threshold,
        )

        self._constraint: Constraint
        if strict:
            self._constraint = And([self._line_count, self._syllables, self._rhyme_scheme])
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 2.0),
                    (self._syllables, 1.0),
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
        return "Elegiac Couplet: hexameter + pentameter (12-10 syllables), rhyming AA"
