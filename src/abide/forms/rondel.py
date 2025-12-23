"""
Rondel family of French fixed forms.

Rondel, Rondelet, Roundel, Rondine, and Rondeau Redoublé.
All feature refrains and interlocking rhyme schemes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    Refrain,
    RhymeScheme,
    StanzaCount,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Rondel(Constraint):
    """
    Rondel: 13-14 line French form with two refrains.

    Structure (13-line version):
    - Lines: ABba abAB abbaA (capitals = refrain)
    - Line 1 repeats at lines 7 and 13
    - Line 2 repeats at line 8
    - Two quatrains and a quintet (4-4-5)

    14-line version adds line 2 at the end: ABba abAB abbaAB

    Examples:
        >>> rondel = Rondel()
        >>> result = rondel.verify(poem)
    """

    name = "Rondel"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        lines: int = 13,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize rondel constraint.

        Args:
            lines: 13 or 14 lines
            weight: Relative weight for composition
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.num_lines = lines
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        self._line_count = LineCount(lines, weight=2.0)

        # Rhyme scheme depends on line count
        if lines == 14:
            rhyme_pattern = "ABBAABABABBAB"  # without the repeated lines considered
        else:
            rhyme_pattern = "ABBAABABABBAA"

        self._rhyme = RhymeScheme(
            rhyme_pattern,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        # Refrains: line 1 repeats at 7 and 13 (0-indexed: 0, 6, 12)
        # Line 2 repeats at 8 (0-indexed: 1, 7), and at 14 in 14-line version
        self._refrain1 = Refrain(
            reference_line=0,
            repeat_at=[6, 12],
            weight=1.5,
        )
        self._refrain2 = Refrain(
            reference_line=1,
            repeat_at=[7] if lines == 13 else [7, 13],
            weight=1.5,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._rhyme, 2.0),
            (self._refrain1, 1.5),
            (self._refrain2, 1.5),
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
        return f"Rondel: {self.num_lines} lines with two refrains"


class Rondelet(Constraint):
    """
    Rondelet: 7-line French form with a refrain.

    Structure:
    - 7 lines with rhyme scheme: AbAabbA
    - Lines 1, 3, 7 are the refrain (4 syllables)
    - Other lines have 8 syllables

    Examples:
        >>> rondelet = Rondelet()
        >>> result = rondelet.verify(poem)
    """

    name = "Rondelet"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize rondelet constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        self._line_count = LineCount(7, weight=2.0)
        self._rhyme = RhymeScheme(
            "ABAABBA",  # Simplified - A lines are refrains
            weight=1.5,
            threshold=rhyme_threshold,
        )

        # Refrain: line 1 (0) repeats at lines 3 (2) and 7 (6)
        self._refrain = Refrain(
            reference_line=0,
            repeat_at=[2, 6],
            weight=2.0,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._rhyme, 1.5),
            (self._refrain, 2.0),
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
        return "Rondelet: 7 lines (AbAabbA) with refrain"


class Roundel(Constraint):
    """
    Roundel: Swinburne's 11-line variant of the rondeau.

    Structure:
    - 11 lines with rhyme scheme: ABaR BAb ABaR
    - R is a refrain (the rentrement) from line 1's opening
    - Three stanzas: 4-3-4 lines

    Examples:
        >>> roundel = Roundel()
        >>> result = roundel.verify(poem)
    """

    name = "Roundel"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize roundel constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        self._line_count = LineCount(11, weight=2.0)
        self._stanza_count = StanzaCount(3, weight=1.0)

        # Rhyme scheme (R is a short refrain line)
        # ABAR BAB ABAR -> using R as different letter
        self._rhyme = RhymeScheme(
            "ABAB BABA ABB",  # Simplified
            weight=1.5,
            threshold=rhyme_threshold,
        )

        # Refrain at positions 4 and 11 (0-indexed: 3 and 10)
        # Note: In a true Roundel, this is a partial refrain (rentrement)
        # but we check for approximate match with lower threshold
        self._refrain = Refrain(
            reference_line=0,
            repeat_at=[3, 10],
            weight=1.5,
            threshold=0.5,  # Lower threshold for partial match
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.0),
            (self._rhyme, 1.5),
            (self._refrain, 1.5),  # Rentrement verification
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
        return "Roundel: 11 lines in 3 stanzas with partial refrain (rentrement)"


class Rondine(Constraint):
    """
    Rondine: 12-line French form with two refrains.

    Similar to rondel but one line shorter. Structure:
    - 12 lines: ABba abAB abaB
    - Line 1 repeats at line 7
    - Line 2 repeats at lines 8 and 12

    Examples:
        >>> rondine = Rondine()
        >>> result = rondine.verify(poem)
    """

    name = "Rondine"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize rondine constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        self._line_count = LineCount(12, weight=2.0)
        self._rhyme = RhymeScheme(
            "ABBAABABABAB",
            weight=2.0,
            threshold=rhyme_threshold,
        )

        # Refrains
        self._refrain1 = Refrain(
            reference_line=0,
            repeat_at=[6],  # 0-indexed
            weight=1.5,
        )
        self._refrain2 = Refrain(
            reference_line=1,
            repeat_at=[7, 11],  # 0-indexed
            weight=1.5,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._rhyme, 2.0),
            (self._refrain1, 1.5),
            (self._refrain2, 1.5),
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
        return "Rondine: 12 lines with two refrains"


class RondeauRedouble(Constraint):
    """
    Rondeau Redoublé: Extended 25-line rondeau form.

    Structure:
    - 25 lines in 6 stanzas (5 quatrains + 1 quintet)
    - Opening quatrain's lines become final lines of stanzas 2-5
    - Final quintet ends with rentrement (opening phrase of line 1)
    - Rhyme scheme: ABAB BABA ABAB BABA ABAB BABAX (X = rentrement)

    Examples:
        >>> rr = RondeauRedouble()
        >>> result = rr.verify(poem)
    """

    name = "Rondeau Redoublé"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize rondeau redoublé constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        # 24 lines + 1 rentrement = 25 lines typically
        # But the rentrement is often counted as part of line 25
        self._line_count = LineCount(24, weight=2.0)
        self._stanza_count = StanzaCount(6, weight=1.0)

        self._rhyme = RhymeScheme(
            "ABAB BABA ABAB BABA ABAB BABA",
            weight=2.0,
            threshold=rhyme_threshold,
        )

        # Lines 1-4 repeat as final lines of stanzas 2-5
        # Line 1 (0) -> line 8 (7)
        # Line 2 (1) -> line 12 (11)
        # Line 3 (2) -> line 16 (15)
        # Line 4 (3) -> line 20 (19)
        self._refrain1 = Refrain(reference_line=0, repeat_at=[7], weight=1.0)
        self._refrain2 = Refrain(reference_line=1, repeat_at=[11], weight=1.0)
        self._refrain3 = Refrain(reference_line=2, repeat_at=[15], weight=1.0)
        self._refrain4 = Refrain(reference_line=3, repeat_at=[19], weight=1.0)

        constraints = [
            (self._line_count, 2.0),
            (self._rhyme, 2.0),
            (self._refrain1, 1.0),
            (self._refrain2, 1.0),
            (self._refrain3, 1.0),
            (self._refrain4, 1.0),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.5)

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
        return "Rondeau Redoublé: 24+ lines with 4 repeated lines"
