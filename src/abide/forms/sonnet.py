"""
Sonnet form templates.

Various sonnet forms: Shakespearean, Petrarchan, Spenserian.
"""

from __future__ import annotations

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    NumericBound,
    RhymeScheme,
    StanzaCount,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)
from abide.primitives import PoemStructure


class Sonnet(Constraint):
    """
    Generic sonnet: 14 lines of iambic pentameter.

    This is a base sonnet without a specific rhyme scheme.
    Use ShakespeareanSonnet, PetrarchanSonnet, or SpenserianSonnet
    for specific variants.

    Examples:
        >>> sonnet = Sonnet()
        >>> result = sonnet.verify(poem)
    """

    name = "Sonnet"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        strict: bool = False,
    ) -> None:
        """
        Initialize sonnet constraint.

        Args:
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line (default 1)
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.strict = strict

        self._line_count = LineCount(14, weight=2.0)
        self._syllables = SyllablesPerLine(
            [10] * 14,  # Iambic pentameter
            weight=1.5,
            tolerance=syllable_tolerance,
        )

        if strict:
            self._constraint = And([self._line_count, self._syllables])
        else:
            self._constraint = WeightedSum(
                [(self._line_count, 2.0), (self._syllables, 1.5)],
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
        return "Sonnet: 14 lines of iambic pentameter"


class ShakespeareanSonnet(Constraint):
    """
    Shakespearean (English) sonnet: ABAB CDCD EFEF GG.

    Structure:
    - 14 lines of iambic pentameter
    - 3 quatrains + couplet
    - Rhyme scheme: ABAB CDCD EFEF GG

    Famous examples:
    - "Shall I compare thee to a summer's day?" (Sonnet 18)
    - "My mistress' eyes are nothing like the sun" (Sonnet 130)

    Examples:
        >>> sonnet = ShakespeareanSonnet()
        >>> result = sonnet.verify(poem)
    """

    name = "Shakespearean Sonnet"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEME = "ABABCDCDEFEFGG"

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
        self.strict = strict

        self._line_count = LineCount(14, weight=2.0)
        self._syllables = SyllablesPerLine(
            [10] * 14,
            weight=1.5,
            tolerance=syllable_tolerance,
        )
        self._rhyme_scheme = RhymeScheme(
            self.RHYME_SCHEME,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        if strict:
            self._constraint = And([
                self._line_count,
                self._syllables,
                self._rhyme_scheme,
            ])
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 2.0),
                    (self._syllables, 1.5),
                    (self._rhyme_scheme, 2.0),
                ],
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
        return "Shakespearean Sonnet: 14 lines, ABAB CDCD EFEF GG rhyme scheme"


class PetrarchanSonnet(Constraint):
    """
    Petrarchan (Italian) sonnet: ABBAABBA + varied sestet.

    Structure:
    - 14 lines of iambic pentameter
    - Octave (8 lines): ABBAABBA
    - Sestet (6 lines): CDECDE or CDCDCD

    Examples:
        >>> sonnet = PetrarchanSonnet()
        >>> result = sonnet.verify(poem)
    """

    name = "Petrarchan Sonnet"
    constraint_type = ConstraintType.COMPOSITE

    # Most common variant
    RHYME_SCHEME = "ABBAABBACDECDE"

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        sestet_scheme: str = "CDECDE",
        strict: bool = False,
    ) -> None:
        """
        Initialize Petrarchan sonnet constraint.

        Args:
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhymes
            sestet_scheme: Rhyme scheme for sestet (default CDECDE)
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.sestet_scheme = sestet_scheme
        self.strict = strict

        full_scheme = "ABBAABBA" + sestet_scheme

        self._line_count = LineCount(14, weight=2.0)
        self._syllables = SyllablesPerLine(
            [10] * 14,
            weight=1.5,
            tolerance=syllable_tolerance,
        )
        self._rhyme_scheme = RhymeScheme(
            full_scheme,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        if strict:
            self._constraint = And([
                self._line_count,
                self._syllables,
                self._rhyme_scheme,
            ])
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 2.0),
                    (self._syllables, 1.5),
                    (self._rhyme_scheme, 2.0),
                ],
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
        return f"Petrarchan Sonnet: 14 lines, ABBAABBA {self.sestet_scheme} rhyme scheme"


class SpenserianSonnet(Constraint):
    """
    Spenserian sonnet: ABAB BCBC CDCD EE.

    Structure:
    - 14 lines of iambic pentameter
    - 3 interlocking quatrains + couplet
    - Rhyme scheme links quatrains: ABAB BCBC CDCD EE

    Named after Edmund Spenser (The Faerie Queene).

    Examples:
        >>> sonnet = SpenserianSonnet()
        >>> result = sonnet.verify(poem)
    """

    name = "Spenserian Sonnet"
    constraint_type = ConstraintType.COMPOSITE

    RHYME_SCHEME = "ABABBCBCCDCDEE"

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
        self.strict = strict

        self._line_count = LineCount(14, weight=2.0)
        self._syllables = SyllablesPerLine(
            [10] * 14,
            weight=1.5,
            tolerance=syllable_tolerance,
        )
        self._rhyme_scheme = RhymeScheme(
            self.RHYME_SCHEME,
            weight=2.0,
            threshold=rhyme_threshold,
        )

        if strict:
            self._constraint = And([
                self._line_count,
                self._syllables,
                self._rhyme_scheme,
            ])
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 2.0),
                    (self._syllables, 1.5),
                    (self._rhyme_scheme, 2.0),
                ],
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
        return "Spenserian Sonnet: 14 lines, ABAB BCBC CDCD EE rhyme scheme"
