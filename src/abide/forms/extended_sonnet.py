"""
Extended and variant sonnet forms.

Curtal Sonnet, Crown of Sonnets, Onegin Stanza, Caudate Sonnet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    Meter,
    NumericBound,
    RhymeScheme,
    StanzaCount,
    StanzaSizes,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)
from abide.primitives import FootLength, MeterType

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class CurtalSonnet(Constraint):
    """
    Curtal Sonnet: Gerard Manley Hopkins' 10.5-line variant.

    Structure:
    - 10 full lines + 1 half-line (traditionally "and a half")
    - Based on 3/4 scale of Petrarchan sonnet
    - Rhyme scheme: ABCABC DBCDC + half-line
    - Written in sprung rhythm (Hopkins' invention)

    For verification, we check 11 lines with the last being shorter.

    Examples:
        >>> curtal = CurtalSonnet()
        >>> result = curtal.verify(poem)
    """

    name = "Curtal Sonnet"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 2,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize curtal sonnet constraint.

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

        # 10 lines + 1 "half-line" = 11 lines
        self._line_count = LineCount(11, weight=2.0)

        # Curtal sonnet rhyme: ABCABC DBCDC (D)
        self._rhyme = RhymeScheme(
            "ABCABCDBCDC",
            weight=2.0,
            threshold=rhyme_threshold,
        )

        # Syllables: 10 syllables for first 10, ~5 for the half-line
        syllable_pattern = [10] * 10 + [5]
        self._syllables = SyllablesPerLine(
            syllable_pattern,
            weight=1.5,
            tolerance=syllable_tolerance,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._rhyme, 2.0),
            (self._syllables, 1.5),
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
        return "Curtal Sonnet: 10.5 lines (Hopkins variant)"


class CrownOfSonnets(Constraint):
    """
    Crown of Sonnets (Corona): 7 linked sonnets.

    Structure:
    - 7 sonnets (usually Petrarchan or Shakespearean)
    - Last line of each sonnet = first line of next
    - Last line of 7th sonnet = first line of 1st
    - Creates a circular "crown" structure
    - 98 lines total (7 × 14)

    Examples:
        >>> crown = CrownOfSonnets()
        >>> result = crown.verify(poem)
    """

    name = "Crown of Sonnets"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        sonnet_count: int = 7,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize crown of sonnets constraint.

        Args:
            sonnet_count: Number of sonnets (traditionally 7)
            weight: Relative weight for composition
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.sonnet_count = sonnet_count
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = sonnet_count * 14

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(sonnet_count, weight=1.5)
        self._stanza_sizes = StanzaSizes([14] * sonnet_count, weight=1.0)

        # Each sonnet should follow Petrarchan rhyme
        # ABBAABBA CDECDE repeated 7 times
        rhyme_pattern = "ABBAABBACDECDE" * sonnet_count
        self._rhyme = RhymeScheme(
            rhyme_pattern,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.5),
            (self._rhyme, 2.0),
        ]

        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.5)

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
        return f"Crown of Sonnets: {self.sonnet_count} linked sonnets ({self.sonnet_count * 14} lines)"


class OneginStanza(Constraint):
    """
    Onegin Stanza: 14-line stanza from Pushkin's Eugene Onegin.

    Structure:
    - 14 lines of iambic tetrameter (8 syllables)
    - Rhyme scheme: AbAbCCddEffEgg
    - Where uppercase = feminine rhyme, lowercase = masculine
    - (For simplicity, we just check the pattern ignoring masc/fem)

    Examples:
        >>> onegin = OneginStanza()
        >>> result = onegin.verify(poem)
    """

    name = "Onegin Stanza"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        stanza_count: int = 1,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize Onegin stanza constraint.

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

        total_lines = stanza_count * 14

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.0)

        # 8 syllables per line (iambic tetrameter)
        syllable_pattern = [8] * total_lines
        self._syllables = SyllablesPerLine(
            syllable_pattern,
            weight=1.5,
            tolerance=syllable_tolerance,
        )

        # AbAbCCddEffEgg -> simplified to ABABCCDDEEFFGG
        rhyme_pattern = "ABABCCDDEEFFGG" * stanza_count
        self._rhyme = RhymeScheme(
            rhyme_pattern,
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
        return f"Onegin Stanza: {self.stanza_count_val} stanza(s) of 14 lines, iambic tetrameter"


class CaudateSonnet(Constraint):
    """
    Caudate Sonnet: Sonnet with additional "tail" lines (coda).

    Structure:
    - Standard 14-line sonnet base
    - Plus additional coda/tail of 2-6 lines
    - Coda often continues the rhyme or adds new pattern
    - Milton and others used this form

    Examples:
        >>> caudate = CaudateSonnet(coda_lines=2)
        >>> result = caudate.verify(poem)
    """

    name = "Caudate Sonnet"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        coda_lines: int = 2,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize caudate sonnet constraint.

        Args:
            coda_lines: Number of lines in the coda/tail (typically 2-6)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.coda_lines = coda_lines
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = 14 + coda_lines

        self._line_count = LineCount(total_lines, weight=2.0)

        # Shakespearean base + coda (coda often alternates or couplets)
        base_rhyme = "ABABCDCDEFEFGG"
        if coda_lines == 2:
            coda_rhyme = "HH"
        elif coda_lines == 4:
            coda_rhyme = "HIHI"
        else:
            coda_rhyme = "H" * coda_lines

        self._rhyme = RhymeScheme(
            base_rhyme + coda_rhyme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        # 10 syllables per line (iambic pentameter)
        syllable_pattern = [10] * total_lines
        self._syllables = SyllablesPerLine(
            syllable_pattern,
            weight=1.5,
            tolerance=syllable_tolerance,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._rhyme, 2.0),
            (self._syllables, 1.5),
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
        return f"Caudate Sonnet: 14 lines + {self.coda_lines}-line coda"
