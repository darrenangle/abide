"""
Rubaiyat (Persian quatrain) form template.

Persian quatrain with AABA rhyme scheme, famously used by Omar Khayyam.
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


class Rubai(Constraint):
    """
    Rubai (singular of Rubaiyat): Persian quatrain with AABA rhyme.

    Structure:
    - 4 lines
    - AABA rhyme scheme (3rd line doesn't rhyme)
    - Often 10-11 syllables per line

    Made famous by the Rubaiyat of Omar Khayyam (Edward FitzGerald's translation).

    Examples:
        >>> rubai = Rubai()
        >>> result = rubai.verify(poem)
    """

    name = "Rubai"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEME = "AABA"

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 2,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize rubai constraint.

        Args:
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        self._line_count = LineCount(4, weight=2.0)
        self._syllables = SyllablesPerLine(
            [10] * 4,
            weight=1.0,
            tolerance=syllable_tolerance,
        )
        self._rhyme_scheme = RhymeScheme(
            self.RHYME_SCHEME,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._syllables, 1.0),
            (self._rhyme_scheme, 2.0),
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
        return "Rubai: 4 lines with AABA rhyme (Persian quatrain)"


class Rubaiyat(Constraint):
    """
    Rubaiyat: Collection of rubai (Persian quatrains).

    A sequence of AABA quatrains, where successive stanzas may be linked
    by having the unrhymed line (B) become the rhyme of the next stanza.

    Structure:
    - Multiple 4-line stanzas
    - Each stanza has AABA rhyme
    - Stanzas may be linked or independent

    Examples:
        >>> rubaiyat = Rubaiyat(stanza_count=4)
        >>> result = rubaiyat.verify(poem)
    """

    name = "Rubaiyat"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        stanza_count: int = 4,
        linked: bool = False,
        weight: float = 1.0,
        syllable_tolerance: int = 2,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize rubaiyat constraint.

        Args:
            stanza_count: Number of rubai stanzas
            linked: If True, B line of each stanza rhymes with A of next
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.linked = linked
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = stanza_count * 4

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.5)
        self._syllables = SyllablesPerLine(
            [10] * total_lines,
            weight=1.0,
            tolerance=syllable_tolerance,
        )

        # Build rhyme scheme
        if linked:
            # Linked: AABA BBCB CCDC etc.
            scheme = ""
            letter = ord("A")
            for i in range(stanza_count):
                a = chr(letter + i)
                b = chr(letter + i + 1)
                scheme += a + a + b + a
        else:
            # Independent: AABA CCDC EEFE etc.
            scheme = ""
            for i in range(stanza_count):
                a = chr(ord("A") + i * 2)
                b = chr(ord("A") + i * 2 + 1)
                scheme += a + a + b + a

        self._rhyme_scheme = RhymeScheme(
            scheme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.5),
            (self._syllables, 1.0),
            (self._rhyme_scheme, 2.0),
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
        link = "linked" if self.linked else "independent"
        return f"Rubaiyat: {self.stanza_count_val} {link} AABA quatrains"
