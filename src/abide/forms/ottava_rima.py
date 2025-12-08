"""
Ottava rima and Rhyme royal form templates.

Ottava rima: 8 lines, ABABABCC (Byron's Don Juan)
Rhyme royal: 7 lines, ABABBCC (Chaucer's Troilus and Criseyde)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


class OttavaRima(Constraint):
    """
    Ottava rima: 8-line stanza with ABABABCC rhyme.

    Italian form used by Ariosto, Tasso. In English, famously
    used by Byron in Don Juan and Beppo.

    Structure:
    - 8 lines of iambic pentameter
    - Rhyme scheme: ABABABCC
    - The closing couplet often delivers a witty turn

    Examples:
        >>> stanza = OttavaRima()
        >>> result = stanza.verify(poem)
    """

    name = "Ottava Rima"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEME = "ABABABCC"

    def __init__(
        self,
        stanza_count: int = 1,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize ottava rima constraint.

        Args:
            stanza_count: Number of stanzas (1 for single stanza)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = stanza_count * 8

        self._line_count = LineCount(total_lines, weight=2.0)
        self._syllables = SyllablesPerLine(
            [10] * total_lines,
            weight=1.5,
            tolerance=syllable_tolerance,
        )

        full_scheme = self.RHYME_SCHEME * stanza_count
        self._rhyme_scheme = RhymeScheme(
            full_scheme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._syllables, 1.5),
            (self._rhyme_scheme, 2.0),
        ]

        if stanza_count > 1:
            self._stanza_count = StanzaCount(stanza_count, weight=1.0)
            constraints.append((self._stanza_count, 1.0))

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
        if self.stanza_count_val == 1:
            return "Ottava Rima: 8 lines of iambic pentameter, ABABABCC rhyme"
        return f"Ottava Rima: {self.stanza_count_val} stanzas, ABABABCC rhyme"


class RhymeRoyal(Constraint):
    """
    Rhyme royal: 7-line stanza with ABABBCC rhyme.

    English form used by Chaucer (Troilus and Criseyde) and Shakespeare
    (The Rape of Lucrece). Also called the Chaucerian stanza.

    Structure:
    - 7 lines of iambic pentameter
    - Rhyme scheme: ABABBCC
    - Often used for narrative and philosophical verse

    Examples:
        >>> stanza = RhymeRoyal()
        >>> result = stanza.verify(poem)
    """

    name = "Rhyme Royal"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEME = "ABABBCC"

    def __init__(
        self,
        stanza_count: int = 1,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize rhyme royal constraint.

        Args:
            stanza_count: Number of stanzas
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = stanza_count * 7

        self._line_count = LineCount(total_lines, weight=2.0)
        self._syllables = SyllablesPerLine(
            [10] * total_lines,
            weight=1.5,
            tolerance=syllable_tolerance,
        )

        full_scheme = self.RHYME_SCHEME * stanza_count
        self._rhyme_scheme = RhymeScheme(
            full_scheme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._syllables, 1.5),
            (self._rhyme_scheme, 2.0),
        ]

        if stanza_count > 1:
            self._stanza_count = StanzaCount(stanza_count, weight=1.0)
            constraints.append((self._stanza_count, 1.0))

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
        if self.stanza_count_val == 1:
            return "Rhyme Royal: 7 lines of iambic pentameter, ABABBCC rhyme"
        return f"Rhyme Royal: {self.stanza_count_val} stanzas, ABABBCC rhyme"


class SpenserianStanza(Constraint):
    """
    Spenserian stanza: 9-line stanza with ABABBCBCC rhyme.

    Invented by Edmund Spenser for The Faerie Queene.

    Structure:
    - 8 lines of iambic pentameter
    - 1 line of alexandrine (iambic hexameter, 12 syllables)
    - Rhyme scheme: ABABBCBCC

    Examples:
        >>> stanza = SpenserianStanza()
        >>> result = stanza.verify(poem)
    """

    name = "Spenserian Stanza"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEME = "ABABBCBCC"

    def __init__(
        self,
        stanza_count: int = 1,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = stanza_count * 9

        self._line_count = LineCount(total_lines, weight=2.0)

        # 8 lines of pentameter + 1 alexandrine per stanza
        syllable_pattern = []
        for _ in range(stanza_count):
            syllable_pattern.extend([10] * 8 + [12])

        self._syllables = SyllablesPerLine(
            syllable_pattern,
            weight=1.5,
            tolerance=syllable_tolerance,
        )

        full_scheme = self.RHYME_SCHEME * stanza_count
        self._rhyme_scheme = RhymeScheme(
            full_scheme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._syllables, 1.5),
            (self._rhyme_scheme, 2.0),
        ]

        if stanza_count > 1:
            self._stanza_count = StanzaCount(stanza_count, weight=1.0)
            constraints.append((self._stanza_count, 1.0))

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
        return "Spenserian Stanza: 9 lines (8 pentameter + 1 alexandrine), ABABBCBCC rhyme"
