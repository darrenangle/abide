"""
Kyrielle form template.

French form with quatrain stanzas and a refrain.
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
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Kyrielle(Constraint):
    """
    Kyrielle: French form with quatrain stanzas and repeating refrain.

    Structure:
    - Quatrain stanzas (4 lines each)
    - 8 syllables per line
    - Rhyme scheme: aabB, ccbB, ddbB (B = refrain line)
    - The last line of each stanza is a refrain

    Examples:
        >>> kyrielle = Kyrielle(stanza_count=3)
        >>> result = kyrielle.verify(poem)
    """

    name = "Kyrielle"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        stanza_count: int = 3,
        rhyme_scheme: str = "aabB",
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize kyrielle constraint.

        Args:
            stanza_count: Number of quatrain stanzas (minimum 2)
            rhyme_scheme: Rhyme pattern per stanza (aabB, ababB, etc.)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = max(2, stanza_count)
        self.rhyme_scheme_str = rhyme_scheme
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = self.stanza_count_val * 4

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(self.stanza_count_val, weight=1.5)
        self._syllables = SyllablesPerLine(
            [8] * total_lines,
            weight=1.0,
            tolerance=syllable_tolerance,
        )

        # Build rhyme scheme with refrain
        # Each stanza has aabB pattern where B lines all rhyme with each other
        full_scheme = self._build_rhyme_scheme()
        self._rhyme_scheme = RhymeScheme(
            full_scheme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        # Refrain: last line of each stanza should be identical
        # For stanza_count stanzas: lines 4, 8, 12, ... (0-indexed: 3, 7, 11, ...)
        refrain_positions = [4 * (i + 1) - 1 for i in range(1, self.stanza_count_val)]
        self._refrain = Refrain(
            reference_line=3,  # First refrain at line 4 (0-indexed: 3)
            repeat_at=refrain_positions,
            weight=2.0,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.5),
            (self._syllables, 1.0),
            (self._rhyme_scheme, 2.0),
            (self._refrain, 2.0),  # Refrain lines must be identical
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.6)

    def _build_rhyme_scheme(self) -> str:
        """Build full rhyme scheme for all stanzas."""
        # Refrain (B) rhymes across all stanzas
        # Other rhymes are per-stanza
        scheme = ""
        letter = ord("A")
        refrain = "Z"  # Use Z for refrain

        for i in range(self.stanza_count_val):
            # aabB pattern per stanza
            stanza_letter = chr(letter + i * 2)
            stanza_letter2 = (
                chr(letter + i * 2 + 1) if self.rhyme_scheme_str == "ababB" else stanza_letter
            )
            scheme += stanza_letter + stanza_letter + stanza_letter2 + refrain

        return scheme

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
        return (
            f"Kyrielle: {self.stanza_count_val} quatrains, 8 syllables, with repeating refrain line"
        )


class KyrielleSonnet(Constraint):
    """
    Kyrielle sonnet: 14-line variant combining kyrielle with sonnet.

    Structure:
    - 14 lines (like a sonnet)
    - 8 syllables per line
    - 3 quatrains + couplet
    - Rhyme scheme with refrain: AabB ccbB ddbB AB

    Examples:
        >>> sonnet = KyrielleSonnet()
        >>> result = sonnet.verify(poem)
    """

    name = "Kyrielle Sonnet"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEME = "AABBCCBBDDBBAB"  # Simplified: refrain on B lines

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
        self.strict_mode = strict

        self._line_count = LineCount(14, weight=2.0)
        self._syllables = SyllablesPerLine(
            [8] * 14,
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
        return "Kyrielle Sonnet: 14 lines, 8 syllables, with refrain pattern"
