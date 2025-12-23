"""
Medieval French fixed forms.

Chant Royal, Double Ballade, Virelai, Canzone.
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
    StanzaSizes,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class ChantRoyal(Constraint):
    """
    Chant Royal: Complex medieval French form.

    Structure:
    - 5 stanzas of 11 lines each = 55 lines
    - Plus 5-line envoi = 60 lines total
    - Rhyme scheme per stanza: ABABCCDDEDE
    - Envoi: DDEDE
    - Each stanza ends with the same refrain line
    - No rhyme word repeated except in the refrain

    Examples:
        >>> chant = ChantRoyal()
        >>> result = chant.verify(poem)
    """

    name = "Chant Royal"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.5,
        strict: bool = False,
    ) -> None:
        """
        Initialize chant royal constraint.

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

        # 5 stanzas x 11 lines + 5-line envoi = 60 lines
        total_lines = 60

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(6, weight=1.0)  # 5 + envoi
        self._stanza_sizes = StanzaSizes([11, 11, 11, 11, 11, 5], weight=1.0)

        # Full rhyme scheme
        # ABABCCDDEDE x 5 + DDEDE
        stanza_rhyme = "ABABCCDDEDE"
        envoi_rhyme = "DDEDE"
        full_rhyme = stanza_rhyme * 5 + envoi_rhyme

        self._rhyme = RhymeScheme(
            full_rhyme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        # 10 syllables per line (decasyllabic)
        syllable_pattern = [10] * total_lines
        self._syllables = SyllablesPerLine(
            syllable_pattern,
            weight=1.0,
            tolerance=syllable_tolerance,
        )

        # Refrain: last line of each stanza should be identical
        # 5 stanzas of 11 lines each: lines 11, 22, 33, 44, 55 (0-indexed: 10, 21, 32, 43, 54)
        self._refrain = Refrain(
            reference_line=10,
            repeat_at=[21, 32, 43, 54],
            weight=2.0,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.0),
            (self._stanza_sizes, 1.0),
            (self._rhyme, 2.0),
            (self._syllables, 1.0),
            (self._refrain, 2.0),
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
        return "Chant Royal: 60 lines (5x11 stanzas + 5-line envoi)"


class DoubleBallade(Constraint):
    """
    Double Ballade: Extended ballade form.

    Structure:
    - 6 stanzas of 8 lines (instead of 3)
    - Plus 4-line envoi = 52 lines total
    - Rhyme scheme per stanza: ABABBCBC
    - Envoi: BCBC
    - Each stanza ends with refrain

    Examples:
        >>> ballade = DoubleBallade()
        >>> result = ballade.verify(poem)
    """

    name = "Double Ballade"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.5,
        strict: bool = False,
    ) -> None:
        """
        Initialize double ballade constraint.

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

        # 6 stanzas x 8 lines + 4-line envoi = 52 lines
        total_lines = 52

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(7, weight=1.0)  # 6 + envoi
        self._stanza_sizes = StanzaSizes([8, 8, 8, 8, 8, 8, 4], weight=1.0)

        # ABABBCBC x 6 + BCBC
        stanza_rhyme = "ABABBCBC"
        envoi_rhyme = "BCBC"
        full_rhyme = stanza_rhyme * 6 + envoi_rhyme

        self._rhyme = RhymeScheme(
            full_rhyme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        # 8 or 10 syllables per line
        syllable_pattern = [10] * total_lines
        self._syllables = SyllablesPerLine(
            syllable_pattern,
            weight=1.0,
            tolerance=syllable_tolerance,
        )

        # Refrain: last line of each stanza should be identical
        # 6 stanzas of 8 lines each: lines 8, 16, 24, 32, 40, 48 (0-indexed: 7, 15, 23, 31, 39, 47)
        self._refrain = Refrain(
            reference_line=7,
            repeat_at=[15, 23, 31, 39, 47],
            weight=2.0,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.0),
            (self._stanza_sizes, 1.0),
            (self._rhyme, 2.0),
            (self._syllables, 1.0),
            (self._refrain, 2.0),
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
        return "Double Ballade: 52 lines (6x8 stanzas + 4-line envoi)"


class Virelai(Constraint):
    """
    Virelai: Medieval French interlocking stanza form.

    Structure varies, but classic virelai ancien:
    - Stanzas with alternating long and short lines
    - Rhyme from end of each stanza becomes main rhyme of next
    - Creates interlocking chain structure
    - Typical: 3 stanzas, each 9-12 lines

    For simplicity, we implement a common 3-stanza version.

    Examples:
        >>> virelai = Virelai()
        >>> result = virelai.verify(poem)
    """

    name = "Virelai"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        stanza_count: int = 3,
        lines_per_stanza: int = 9,
        weight: float = 1.0,
        rhyme_threshold: float = 0.5,
        strict: bool = False,
    ) -> None:
        """
        Initialize virelai constraint.

        Args:
            stanza_count: Number of stanzas
            lines_per_stanza: Lines per stanza
            weight: Relative weight for composition
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.lines_per_stanza = lines_per_stanza
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = stanza_count * lines_per_stanza

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.5)
        self._stanza_sizes = StanzaSizes([lines_per_stanza] * stanza_count, weight=1.0)

        # Simplified rhyme scheme: AABaabAAB for each stanza
        # (uppercase = long line, lowercase = short)
        stanza_rhyme = "AABAABAAB"[:lines_per_stanza]
        full_rhyme = stanza_rhyme * stanza_count

        self._rhyme = RhymeScheme(
            full_rhyme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.5),
            (self._rhyme, 2.0),
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
        total = self.stanza_count_val * self.lines_per_stanza
        return f"Virelai: {total} lines in {self.stanza_count_val} interlocking stanzas"


class Canzone(Constraint):
    """
    Canzone: Provençal/Italian complex stanza form.

    Structure:
    - 5-7 stanzas of 7-20 lines each (typically 13)
    - Plus shorter tornada (envoi) of 3-7 lines
    - Complex rhyme schemes specific to each poem
    - All stanzas follow identical structure

    This implements a common Petrarchan canzone structure.

    Examples:
        >>> canzone = Canzone()
        >>> result = canzone.verify(poem)
    """

    name = "Canzone"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        stanza_count: int = 5,
        lines_per_stanza: int = 13,
        tornada_lines: int = 5,
        weight: float = 1.0,
        rhyme_threshold: float = 0.5,
        strict: bool = False,
    ) -> None:
        """
        Initialize canzone constraint.

        Args:
            stanza_count: Number of main stanzas
            lines_per_stanza: Lines per main stanza
            tornada_lines: Lines in the tornada/envoi
            weight: Relative weight for composition
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.lines_per_stanza = lines_per_stanza
        self.tornada_lines = tornada_lines
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = stanza_count * lines_per_stanza + tornada_lines
        total_stanzas = stanza_count + 1

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(total_stanzas, weight=1.5)

        stanza_sizes = [lines_per_stanza] * stanza_count + [tornada_lines]
        self._stanza_sizes = StanzaSizes(stanza_sizes, weight=1.0)

        # Common canzone rhyme scheme (simplified)
        # ABCBACcDdEE per 13-line stanza
        stanza_rhyme = "ABCBACCDDEEFF"[:lines_per_stanza]
        tornada_rhyme = "CDCDC"[:tornada_lines]
        full_rhyme = stanza_rhyme * stanza_count + tornada_rhyme

        self._rhyme = RhymeScheme(
            full_rhyme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.5),
            (self._stanza_sizes, 1.0),
            (self._rhyme, 2.0),
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
        total = self.stanza_count_val * self.lines_per_stanza + self.tornada_lines
        return f"Canzone: {total} lines ({self.stanza_count_val}x{self.lines_per_stanza} + tornada)"
