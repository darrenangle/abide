"""
Quatrain form templates.

Quatrains are four-line stanzas with various rhyme schemes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    RhymeScheme,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Quatrain(Constraint):
    """
    Generic quatrain: 4 lines with configurable rhyme scheme.

    Common schemes:
    - ABAB (alternating/cross rhyme)
    - AABB (couplet rhyme)
    - ABBA (envelope/enclosed rhyme)
    - ABCB (ballad stanza)
    - AAAA (monorhyme)

    Examples:
        >>> quatrain = Quatrain(rhyme_scheme="ABAB")
        >>> result = quatrain.verify(poem)
    """

    name = "Quatrain"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        rhyme_scheme: str = "ABAB",
        syllables_per_line: int | list[int] = 10,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize quatrain constraint.

        Args:
            rhyme_scheme: Rhyme pattern (ABAB, AABB, ABBA, ABCB, etc.)
            syllables_per_line: Syllables per line (int for uniform, list for varied)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_scheme_str = rhyme_scheme
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict = strict

        # Convert syllables to list if int
        if isinstance(syllables_per_line, int):
            syllable_list = [syllables_per_line] * 4
        else:
            syllable_list = syllables_per_line

        self._line_count = LineCount(4, weight=2.0)
        self._syllables = SyllablesPerLine(
            syllable_list,
            weight=1.0,
            tolerance=syllable_tolerance,
        )
        self._rhyme_scheme = RhymeScheme(
            rhyme_scheme,
            weight=1.5,
            threshold=rhyme_threshold,
        )

        self._constraint: Constraint
        if strict:
            self._constraint = And(
                [self._line_count, self._syllables, self._rhyme_scheme]
            )
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 2.0),
                    (self._syllables, 1.0),
                    (self._rhyme_scheme, 1.5),
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
        return f"Quatrain: 4 lines with {self.rhyme_scheme_str} rhyme scheme"


class BalladStanza(Quatrain):
    """
    Ballad stanza (Common Meter): 4 lines, 8-6-8-6 syllables, ABCB rhyme.

    The classic form for narrative ballads and hymns.
    Uses iambic tetrameter/trimeter alternation.

    Examples:
        >>> stanza = BalladStanza()
        >>> result = stanza.verify(poem)
    """

    name = "Ballad Stanza"

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        super().__init__(
            rhyme_scheme="ABCB",
            syllables_per_line=[8, 6, 8, 6],
            weight=weight,
            syllable_tolerance=syllable_tolerance,
            rhyme_threshold=rhyme_threshold,
            strict=strict,
        )

    def describe(self) -> str:
        return "Ballad Stanza: 4 lines, 8-6-8-6 syllables, ABCB rhyme"


class HeroicQuatrain(Quatrain):
    """
    Heroic quatrain: 4 lines of iambic pentameter, ABAB rhyme.

    Used in elegies and longer heroic poems.

    Examples:
        >>> quatrain = HeroicQuatrain()
        >>> result = quatrain.verify(poem)
    """

    name = "Heroic Quatrain"

    def __init__(
        self,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        super().__init__(
            rhyme_scheme="ABAB",
            syllables_per_line=10,
            weight=weight,
            syllable_tolerance=syllable_tolerance,
            rhyme_threshold=rhyme_threshold,
            strict=strict,
        )

    def describe(self) -> str:
        return "Heroic Quatrain: 4 lines of iambic pentameter, ABAB rhyme"


class EnvelopeQuatrain(Quatrain):
    """
    Envelope quatrain: 4 lines with ABBA rhyme (enclosed/envelope rhyme).

    The outer lines rhyme, enclosing the inner couplet.

    Examples:
        >>> quatrain = EnvelopeQuatrain()
        >>> result = quatrain.verify(poem)
    """

    name = "Envelope Quatrain"

    def __init__(
        self,
        syllables_per_line: int = 10,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        super().__init__(
            rhyme_scheme="ABBA",
            syllables_per_line=syllables_per_line,
            weight=weight,
            syllable_tolerance=syllable_tolerance,
            rhyme_threshold=rhyme_threshold,
            strict=strict,
        )

    def describe(self) -> str:
        return "Envelope Quatrain: 4 lines with ABBA rhyme"
