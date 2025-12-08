"""
Ballad form templates.

Ballads are narrative poems, often in quatrain stanzas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    MeterPattern,
    NumericBound,
    RhymeScheme,
    StanzaCount,
    StanzaSizes,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)
from abide.primitives import MeterType

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Ballad(Constraint):
    """
    Traditional ballad: quatrains with ABCB or ABAB rhyme.

    Features:
    - Quatrain stanzas (4 lines each)
    - Common Meter: 8-6-8-6 syllables (iambic 4-3-4-3)
    - ABCB or ABAB rhyme scheme
    - Narrative content

    Examples:
        >>> ballad = Ballad(stanza_count=4, rhyme_scheme="ABCB")
        >>> result = ballad.verify(poem)
    """

    name = "Ballad"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        stanza_count: int = 4,
        rhyme_scheme: str = "ABCB",
        use_common_meter: bool = True,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize ballad constraint.

        Args:
            stanza_count: Number of quatrain stanzas
            rhyme_scheme: Rhyme pattern per stanza (ABCB or ABAB)
            use_common_meter: If True, enforce 8-6-8-6 syllable pattern
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.rhyme_scheme_str = rhyme_scheme
        self.use_common_meter = use_common_meter
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = stanza_count * 4

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.5)
        self._stanza_sizes = StanzaSizes([4] * stanza_count, weight=1.0)

        # Syllable pattern
        if use_common_meter:
            syllable_pattern = [8, 6, 8, 6] * stanza_count
        else:
            syllable_pattern = [8] * total_lines

        self._syllables = SyllablesPerLine(
            syllable_pattern,
            weight=1.5,
            tolerance=syllable_tolerance,
        )

        # Repeat rhyme scheme for each stanza
        full_scheme = rhyme_scheme * stanza_count
        self._rhyme_scheme = RhymeScheme(
            full_scheme,
            weight=1.5,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.5),
            (self._syllables, 1.5),
            (self._rhyme_scheme, 1.5),
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
        meter = "Common Meter (8-6-8-6)" if self.use_common_meter else "8 syllables"
        return (
            f"Ballad: {self.stanza_count_val} quatrains, {self.rhyme_scheme_str} rhyme, {meter}"
        )


class LiteraryBallad(Constraint):
    """
    Literary ballad: More flexible form for art ballads.

    Less strict than traditional ballad, allows:
    - Variable stanza sizes
    - More rhyme scheme options
    - Greater metrical flexibility

    Examples:
        >>> ballad = LiteraryBallad(min_stanzas=3)
        >>> result = ballad.verify(poem)
    """

    name = "Literary Ballad"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        min_stanzas: int = 3,
        min_lines: int = 12,
        weight: float = 1.0,
        syllable_tolerance: int = 2,
        strict: bool = False,
    ) -> None:
        """
        Initialize literary ballad constraint.

        Args:
            min_stanzas: Minimum number of stanzas
            min_lines: Minimum total lines
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_stanzas = min_stanzas
        self.min_lines = min_lines
        self.syllable_tolerance = syllable_tolerance
        self.strict_mode = strict

        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=2.0)
        self._stanza_count = StanzaCount(NumericBound.at_least(min_stanzas), weight=1.5)

        if strict:
            self._constraint = And([self._line_count, self._stanza_count])
        else:
            self._constraint = WeightedSum(
                [(self._line_count, 2.0), (self._stanza_count, 1.5)],
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
        return f"Literary Ballad: at least {self.min_lines} lines in {self.min_stanzas}+ stanzas"


class BroadBallad(Constraint):
    """
    Broadside ballad: Popular street ballad form.

    Features:
    - Simpler structure than traditional ballad
    - Often AABB couplet rhyme within quatrains
    - Regular syllable count (8 per line)
    - Designed for easy singing/recitation

    Examples:
        >>> ballad = BroadBallad(stanza_count=6)
        >>> result = ballad.verify(poem)
    """

    name = "Broadside Ballad"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        stanza_count: int = 4,
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

        total_lines = stanza_count * 4

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.5)
        self._syllables = SyllablesPerLine(
            [8] * total_lines,
            weight=1.0,
            tolerance=syllable_tolerance,
        )

        # AABB couplet rhyme within each quatrain
        full_scheme = "AABB" * stanza_count
        self._rhyme_scheme = RhymeScheme(
            full_scheme,
            weight=1.5,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.5),
            (self._syllables, 1.0),
            (self._rhyme_scheme, 1.5),
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
        return f"Broadside Ballad: {self.stanza_count_val} quatrains, AABB rhyme, 8 syllables"
