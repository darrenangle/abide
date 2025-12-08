"""
Burns Stanza (Standard Habbie) form template.

Named after Robert Burns who popularized it, though it was invented
by Robert Sempill for his poem "The Life and Death of Habbie Simson".
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
    StanzaSizes,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class BurnsStanza(Constraint):
    """
    Burns Stanza (Standard Habbie): 6-line stanza with AAABAB rhyme.

    Structure per stanza:
    - Lines 1, 2, 3: 8 syllables (tetrameter), rhyme A
    - Line 4: 4 syllables (dimeter), rhyme B
    - Line 5: 8 syllables (tetrameter), rhyme A
    - Line 6: 4 syllables (dimeter), rhyme B

    Used extensively by Robert Burns in poems like "To a Mouse"
    and "To a Louse".

    Examples:
        >>> burns = BurnsStanza(stanza_count=2)
        >>> result = burns.verify(poem)
    """

    name = "Burns Stanza"
    constraint_type = ConstraintType.COMPOSITE

    # 8-8-8-4-8-4 syllable pattern
    SYLLABLE_PATTERN = [8, 8, 8, 4, 8, 4]

    def __init__(
        self,
        stanza_count: int = 1,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize Burns stanza constraint.

        Args:
            stanza_count: Number of stanzas
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = stanza_count * 6

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.0)
        self._stanza_sizes = StanzaSizes([6] * stanza_count, weight=1.0)

        # Repeat syllable pattern for each stanza
        syllable_pattern = self.SYLLABLE_PATTERN * stanza_count
        self._syllables = SyllablesPerLine(
            syllable_pattern,
            weight=1.5,
            tolerance=syllable_tolerance,
        )

        # AAABAB rhyme per stanza
        rhyme_scheme = "AAABAB" * stanza_count
        self._rhyme = RhymeScheme(
            rhyme_scheme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.0),
            (self._syllables, 1.5),
            (self._rhyme, 2.0),
        ]

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
        return f"Burns Stanza: {self.stanza_count_val} stanza(s) of 6 lines (8-8-8-4-8-4), AAABAB rhyme"
