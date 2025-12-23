"""
World poetic forms from various cultures.

Tanaga (Filipino), Naani (Telugu/Indian), Seguidilla (Spanish),
Lai/Lay (Medieval French), Rispetto (Italian).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

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


class Tanaga(Constraint):
    """
    Tanaga: Filipino quatrain with 7 syllables per line.

    Traditional Filipino form with:
    - 4 lines of 7 syllables each (28 total)
    - AABB or ABAB rhyme scheme
    - Often deals with love, nature, or social commentary

    Examples:
        >>> tanaga = Tanaga()
        >>> result = tanaga.verify(poem)
    """

    name = "Tanaga"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        rhyme_scheme: str = "AABB",
        weight: float = 1.0,
        syllable_tolerance: int = 0,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize tanaga constraint.

        Args:
            rhyme_scheme: Rhyme pattern (AABB or ABAB)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_scheme_str = rhyme_scheme
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        self._line_count = LineCount(4, weight=2.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            [7, 7, 7, 7],
            weight=2.0,
            tolerance=syllable_tolerance,
        )
        self._rhyme = RhymeScheme(
            rhyme_scheme,
            weight=1.5,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 0.5),
            (self._syllables, 2.0),
            (self._rhyme, 1.5),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.7)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Get individual constraint results
        line_result = self._line_count.verify(poem)
        stanza_result = self._stanza_count.verify(poem)
        syllables_result = self._syllables.verify(poem)
        rhyme_result = self._rhyme.verify(poem)

        # Count violations in syllable pattern (4 lines of 7 syllables each)
        from abide.primitives import count_line_syllables

        expected = [7, 7, 7, 7]
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
            (line_result.score, 2.0),
            (stanza_result.score, 0.5),
            (syllable_score, 2.0),
            (rhyme_result.score, 1.5),
        ]
        total_weight = sum(w for _, w in scores)
        score = sum(s * w for s, w in scores) / total_weight

        passed = (
            score >= 0.7
            if not self.strict_mode
            else (
                violations == 0
                and line_result.passed
                and stanza_result.passed
                and rhyme_result.passed
            )
        )

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=line_result.rubric
            + stanza_result.rubric
            + syllables_result.rubric
            + rhyme_result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"violations": violations},
        )

    def describe(self) -> str:
        return f"Tanaga: 4 lines of 7 syllables, {self.rhyme_scheme_str} rhyme"


class Naani(Constraint):
    """
    Naani: Telugu/Indian 4-line poem with 20-25 total syllables.

    Features:
    - 4 lines total
    - 20-25 syllables across all lines
    - No fixed syllable pattern per line
    - No required rhyme scheme

    Examples:
        >>> naani = Naani()
        >>> result = naani.verify(poem)
    """

    name = "Naani"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        min_syllables: int = 20,
        max_syllables: int = 25,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        """
        Initialize naani constraint.

        Args:
            min_syllables: Minimum total syllables
            max_syllables: Maximum total syllables
            weight: Relative weight for composition
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables
        self.strict_mode = strict

        self._line_count = LineCount(4, weight=2.0)
        self._stanza_count = StanzaCount(1, weight=0.5)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check line count
        line_result = self._line_count.verify(poem)
        stanza_result = self._stanza_count.verify(poem)

        # Count total syllables
        from abide.primitives.phonetics import count_line_syllables

        total_syllables = sum(count_line_syllables(line) for line in structure.lines)

        # Score syllable count - steep penalty for strict GRPO training
        # Treat syllable range violation as a single violation
        if self.min_syllables <= total_syllables <= self.max_syllables:
            syllable_score = 1.0
        else:
            # Out of range is a violation - apply steep penalty
            # Calculate "distance" from range as violation count
            if total_syllables < self.min_syllables:
                diff = self.min_syllables - total_syllables
            else:
                diff = total_syllables - self.max_syllables

            # Convert difference to violation count (every 2-3 syllables off = 1 violation)
            violations = max(1, (diff + 1) // 2)

            if violations == 1:
                syllable_score = 0.5
            elif violations == 2:
                syllable_score = 0.25
            else:
                syllable_score = 0.05

        # Combine scores
        scores = [
            (line_result.score, 2.0),
            (stanza_result.score, 0.5),
            (syllable_score, 2.0),
        ]
        total_weight = sum(w for _, w in scores)
        score = sum(s * w for s, w in scores) / total_weight

        passed = score >= 0.7

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=line_result.rubric + stanza_result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "total_syllables": total_syllables,
                "min_syllables": self.min_syllables,
                "max_syllables": self.max_syllables,
            },
        )

    def describe(self) -> str:
        return f"Naani: 4 lines with {self.min_syllables}-{self.max_syllables} total syllables"


class Seguidilla(Constraint):
    """
    Seguidilla: Spanish 7-line stanza form.

    Structure:
    - 7 lines with syllable pattern: 7-5-7-5-5-7-5
    - Rhyme scheme: xAxAxBx (where x is unrhymed)
    - Often used for dance songs

    Examples:
        >>> seguidilla = Seguidilla()
        >>> result = seguidilla.verify(poem)
    """

    name = "Seguidilla"
    constraint_type = ConstraintType.COMPOSITE

    SYLLABLE_PATTERN: ClassVar[list[int]] = [7, 5, 7, 5, 5, 7, 5]

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize seguidilla constraint.

        Args:
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        self._line_count = LineCount(7, weight=2.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            self.SYLLABLE_PATTERN,
            weight=2.0,
            tolerance=syllable_tolerance,
        )
        # xAxAxBx - only lines 2,4 rhyme (A) and line 6 with 7 (B)
        # Approximate with XAXAXBX
        self._rhyme = RhymeScheme(
            "XAXAXBX",
            weight=1.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 0.5),
            (self._syllables, 2.0),
            (self._rhyme, 1.0),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.6)

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
        return "Seguidilla: 7 lines (7-5-7-5-5-7-5 syllables)"


class Lai(Constraint):
    """
    Lai (or Lay): Medieval French 9-line form.

    Structure:
    - 9 lines
    - Rhyme scheme: AABAABAAB
    - Lines 1,2,4,5,7,8 have 5 syllables (A lines)
    - Lines 3,6,9 have 2 syllables (B lines)

    Examples:
        >>> lai = Lai()
        >>> result = lai.verify(poem)
    """

    name = "Lai"
    constraint_type = ConstraintType.COMPOSITE

    # A=5 syllables, B=2 syllables
    SYLLABLE_PATTERN: ClassVar[list[int]] = [5, 5, 2, 5, 5, 2, 5, 5, 2]

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 0,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize lai constraint.

        Args:
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        self._line_count = LineCount(9, weight=2.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            self.SYLLABLE_PATTERN,
            weight=2.0,
            tolerance=syllable_tolerance,
        )
        self._rhyme = RhymeScheme(
            "AABAABAAB",
            weight=1.5,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 0.5),
            (self._syllables, 2.0),
            (self._rhyme, 1.5),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.6)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)

        # Count violations (rubric items that failed)
        violations = sum(1 for r in result.rubric if not r.passed)

        # Steep penalty scoring: 0 violations = 1.0, 1 = 0.5, 2 = 0.25, 3+ = 0.05
        if violations == 0:
            overall_score = 1.0
        elif violations == 1:
            overall_score = 0.5
        elif violations == 2:
            overall_score = 0.25
        else:
            overall_score = 0.05

        return VerificationResult(
            score=overall_score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return "Lai: 9 lines (5-5-2 pattern), AABAABAAB rhyme"


class Rispetto(Constraint):
    """
    Rispetto: Italian 8-line form.

    Structure:
    - 8 lines of hendecasyllables (11 syllables, or 10 in English adaptation)
    - Rhyme scheme: ABABABCC (Tuscan) or ABABCCDD (Sicilian)
    - Often about love or respect (rispetto = respect)

    Examples:
        >>> rispetto = Rispetto(variant="tuscan")
        >>> result = rispetto.verify(poem)
    """

    name = "Rispetto"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEMES: ClassVar[dict[str, str]] = {
        "tuscan": "ABABABCC",
        "sicilian": "ABABCCDD",
    }

    def __init__(
        self,
        variant: str = "tuscan",
        syllables_per_line: int = 10,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize rispetto constraint.

        Args:
            variant: "tuscan" (ABABABCC) or "sicilian" (ABABCCDD)
            syllables_per_line: Expected syllables per line (10 or 11)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.variant = variant
        self.syllables_per_line = syllables_per_line
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        rhyme_scheme = self.RHYME_SCHEMES.get(variant, "ABABABCC")

        self._line_count = LineCount(8, weight=2.0)
        self._stanza_count = StanzaCount(1, weight=0.5)
        self._syllables = SyllablesPerLine(
            [syllables_per_line] * 8,
            weight=1.5,
            tolerance=syllable_tolerance,
        )
        self._rhyme = RhymeScheme(
            rhyme_scheme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 0.5),
            (self._syllables, 1.5),
            (self._rhyme, 2.0),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.6)

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
        scheme = self.RHYME_SCHEMES.get(self.variant, "ABABABCC")
        return f"Rispetto ({self.variant}): 8 lines of {self.syllables_per_line} syllables, {scheme} rhyme"
