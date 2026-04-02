"""
Epigram and other short forms.

Structural proxies for epigrams and related short forms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    LineLengthRange,
    MeasureMode,
    RhymeScheme,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Epigram(Constraint):
    """
    Epigram structural proxy, typically 2-4 lines.

    Historical epigrams are often witty or pointed, but this verifier checks
    only line-count bounds plus rhyme on the 2-line and 4-line variants.

    Examples:
        >>> epigram = Epigram()
        >>> result = epigram.verify(poem)
    """

    name = "Epigram"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        min_lines: int = 2,
        max_lines: int = 4,
        min_words_per_line: int = 2,
        max_words_per_line: int = 12,
        weight: float = 1.0,
        syllable_tolerance: int = 2,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize epigram constraint.

        Args:
            min_lines: Minimum number of lines (default 2)
            max_lines: Maximum number of lines (default 4)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.min_words_per_line = min_words_per_line
        self.max_words_per_line = max_words_per_line
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict
        self._line_length = LineLengthRange(
            min_length=min_words_per_line,
            max_length=max_words_per_line,
            mode=MeasureMode.WORDS,
            weight=1.0,
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check line count - steep penalties for strict GRPO training
        # Perfect = 1.0, 1 violation = 0.5, 2 violations = 0.25, 3+ = 0.05
        if structure.line_count < self.min_lines:
            violations = self.min_lines - structure.line_count
            if violations == 1:
                score = 0.5
            elif violations == 2:
                score = 0.25
            else:
                score = 0.05
            return VerificationResult(
                score=score,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={
                    "error": f"Too few lines (minimum {self.min_lines}, got {structure.line_count})"
                },
            )
        if structure.line_count > self.max_lines:
            violations = structure.line_count - self.max_lines
            if violations == 1:
                score = 0.5
            elif violations == 2:
                score = 0.25
            else:
                score = 0.05
            return VerificationResult(
                score=score,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={
                    "error": f"Too many lines (maximum {self.max_lines}, got {structure.line_count})"
                },
            )

        length_result = self._line_length.verify(poem)

        # For 2-line epigrams, check couplet rhyme
        if structure.line_count == 2:
            rhyme = RhymeScheme("AA", threshold=self.rhyme_threshold)
            rhyme_result = rhyme.verify(poem)
            return VerificationResult(
                score=(length_result.score + (2 * rhyme_result.score)) / 3,
                passed=length_result.passed and rhyme_result.passed,
                rubric=length_result.rubric + rhyme_result.rubric,
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={
                    "line_count": structure.line_count,
                    "min_words_per_line": self.min_words_per_line,
                    "max_words_per_line": self.max_words_per_line,
                    **rhyme_result.details,
                },
            )

        # For 4-line epigrams, check AABB or ABAB
        if structure.line_count == 4:
            rhyme_aabb = RhymeScheme("AABB", threshold=self.rhyme_threshold)
            rhyme_abab = RhymeScheme("ABAB", threshold=self.rhyme_threshold)
            result_aabb = rhyme_aabb.verify(poem)
            result_abab = rhyme_abab.verify(poem)
            best = result_aabb if result_aabb.score > result_abab.score else result_abab
            return VerificationResult(
                score=(length_result.score + (2 * best.score)) / 3,
                passed=length_result.passed and best.passed,
                rubric=length_result.rubric + best.rubric,
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={
                    "line_count": structure.line_count,
                    "min_words_per_line": self.min_words_per_line,
                    "max_words_per_line": self.max_words_per_line,
                    **best.details,
                },
            )

        # For 3 lines, enforce the short-line proxy without a rhyme requirement.
        return VerificationResult(
            score=length_result.score,
            passed=length_result.passed,
            rubric=length_result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "line_count": structure.line_count,
                "min_words_per_line": self.min_words_per_line,
                "max_words_per_line": self.max_words_per_line,
            },
        )

    def describe(self) -> str:
        return (
            f"Epigram: {self.min_lines}-{self.max_lines} lines; "
            f"{self.min_words_per_line}-{self.max_words_per_line} words per line; "
            "rhyme proxy on 2-line and 4-line variants"
        )


class Monostich(Constraint):
    """
    Monostich: Single-line poem.

    The simplest poetic form - one complete line of poetry.

    Examples:
        >>> mono = Monostich()
        >>> result = mono.verify("All that glitters is not gold")
    """

    name = "Monostich"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        min_syllables: int = 5,
        max_syllables: int = 20,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables

        self._line_count = LineCount(1, weight=2.0)
        self._length = LineLengthRange(
            min_length=min_syllables,
            max_length=max_syllables,
            mode=MeasureMode.SYLLABLES,
            weight=1.0,
        )

        self._constraint = And([self._line_count, self._length])

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
        return f"Monostich: 1 line ({self.min_syllables}-{self.max_syllables} syllables)"


class Distich(Constraint):
    """
    Distich: Two-line poem or couplet unit.

    Classical distiches often used elegiac verse.
    This verifier checks line count, a per-line syllable proxy,
    and optional rhyme.

    Examples:
        >>> distich = Distich()
        >>> result = distich.verify(poem)
    """

    name = "Distich"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        syllables_per_line: int = 10,
        require_rhyme: bool = True,
        weight: float = 1.0,
        syllable_tolerance: int = 2,
        rhyme_threshold: float = 0.6,
    ) -> None:
        super().__init__(weight)
        self.syllables = syllables_per_line
        self.require_rhyme = require_rhyme
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold

        self._line_count = LineCount(2, weight=2.0)
        self._syllables = SyllablesPerLine(
            [syllables_per_line] * 2,
            weight=1.0,
            tolerance=syllable_tolerance,
        )

        constraints: list[tuple[Constraint, float]] = [
            (self._line_count, 2.0),
            (self._syllables, 1.0),
        ]

        if require_rhyme:
            self._rhyme = RhymeScheme("AA", threshold=rhyme_threshold, weight=1.5)
            constraints.append((self._rhyme, 1.5))

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
        rhyme = " rhyming" if self.require_rhyme else ""
        return f"Distich: 2{rhyme} lines ({self.syllables} syllables each)"


class Tercet(Constraint):
    """
    Tercet: Three-line stanza or poem.

    Various rhyme schemes: AAA (triplet), ABA (enclosed), ABC (unrhymed).

    Examples:
        >>> tercet = Tercet(rhyme_scheme="ABA")
        >>> result = tercet.verify(poem)
    """

    name = "Tercet"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        rhyme_scheme: str = "ABA",
        syllables_per_line: int = 10,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        super().__init__(weight)
        self.rhyme_scheme_str = rhyme_scheme
        self.syllables = syllables_per_line
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        self._line_count = LineCount(3, weight=2.0)
        self._syllables = SyllablesPerLine(
            [syllables_per_line] * 3,
            weight=1.0,
            tolerance=syllable_tolerance,
        )
        self._rhyme_scheme = RhymeScheme(
            rhyme_scheme,
            weight=1.5,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._syllables, 1.0),
            (self._rhyme_scheme, 1.5),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
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
        return f"Tercet: 3 lines with {self.rhyme_scheme_str} rhyme"


class Triplet(Tercet):
    """
    Triplet: Three-line stanza with AAA rhyme (monorhyme).

    Examples:
        >>> triplet = Triplet()
        >>> result = triplet.verify(poem)
    """

    name = "Triplet"

    def __init__(
        self,
        syllables_per_line: int = 10,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        super().__init__(
            rhyme_scheme="AAA",
            syllables_per_line=syllables_per_line,
            weight=weight,
            syllable_tolerance=syllable_tolerance,
            rhyme_threshold=rhyme_threshold,
            strict=strict,
        )

    def describe(self) -> str:
        return "Triplet: 3 lines with AAA monorhyme"
