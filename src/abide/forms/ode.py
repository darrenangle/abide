"""
Ode form templates.

Odes are lyric poems, typically addressed to a subject,
with various structural patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    NumericBound,
    RhymeScheme,
    StanzaCount,
    StanzaSizes,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Ode(Constraint):
    """
    Generic ode: lyric poem with varied structure.

    This is a base ode class. Use specific variants:
    - HoratianOde: Regular stanzas
    - PindaricOde: Triadic structure (strophe, antistrophe, epode)
    - IrregularOde: Free form

    Examples:
        >>> ode = Ode(min_lines=12)
        >>> result = ode.verify(poem)
    """

    name = "Ode"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        min_lines: int = 10,
        max_lines: int = 100,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        """
        Initialize generic ode constraint.

        Args:
            min_lines: Minimum number of lines
            max_lines: Maximum number of lines
            weight: Relative weight for composition
            strict: If True, line constraints must pass
        """
        super().__init__(weight)
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.strict_mode = strict

        self._min_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check line count range
        if structure.line_count < self.min_lines:
            score = structure.line_count / self.min_lines
            passed = False
        elif structure.line_count > self.max_lines:
            score = self.max_lines / structure.line_count
            passed = False
        else:
            score = 1.0
            passed = True

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "line_count": structure.line_count,
                "min_lines": self.min_lines,
                "max_lines": self.max_lines,
            },
        )

    def describe(self) -> str:
        return f"Ode: {self.min_lines}-{self.max_lines} lines"


class HoratianOde(Constraint):
    """
    Horatian ode: Regular stanzas with consistent structure.

    Named after the Roman poet Horace. Features:
    - Regular stanza length
    - Consistent rhyme scheme per stanza
    - Often uses quatrains

    Examples:
        >>> ode = HoratianOde(stanza_size=4, rhyme_scheme="ABAB")
        >>> result = ode.verify(poem)
    """

    name = "Horatian Ode"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        stanza_count: int = 4,
        stanza_size: int = 4,
        rhyme_scheme: str = "ABAB",
        syllables_per_line: int = 10,
        weight: float = 1.0,
        syllable_tolerance: int = 2,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize Horatian ode constraint.

        Args:
            stanza_count: Number of stanzas
            stanza_size: Lines per stanza
            rhyme_scheme: Rhyme pattern per stanza
            syllables_per_line: Syllables per line
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.stanza_size = stanza_size
        self.rhyme_scheme_str = rhyme_scheme
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = stanza_count * stanza_size

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.5)
        self._stanza_sizes = StanzaSizes([stanza_size] * stanza_count, weight=1.0)
        self._syllables = SyllablesPerLine(
            [syllables_per_line] * total_lines,
            weight=1.0,
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
            (self._syllables, 1.0),
            (self._rhyme_scheme, 1.5),
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
        return (
            f"Horatian Ode: {self.stanza_count_val} stanzas of {self.stanza_size} lines, "
            f"{self.rhyme_scheme_str} rhyme"
        )


class PindaricOde(Constraint):
    """
    Pindaric ode: Triadic structure with strophe, antistrophe, and epode.

    Named after the Greek poet Pindar. Features:
    - Strophe and antistrophe have identical structure
    - Epode has different structure
    - Often used for ceremonial/praise poetry

    Examples:
        >>> ode = PindaricOde()
        >>> result = ode.verify(poem)
    """

    name = "Pindaric Ode"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        triads: int = 1,
        strophe_lines: int = 8,
        epode_lines: int = 6,
        weight: float = 1.0,
        syllable_tolerance: int = 2,
        strict: bool = False,
    ) -> None:
        """
        Initialize Pindaric ode constraint.

        Args:
            triads: Number of strophe-antistrophe-epode triads
            strophe_lines: Lines in strophe and antistrophe
            epode_lines: Lines in epode
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.triads = triads
        self.strophe_lines = strophe_lines
        self.epode_lines = epode_lines
        self.syllable_tolerance = syllable_tolerance
        self.strict_mode = strict

        # Total structure: [strophe, antistrophe, epode] * triads
        stanza_sizes = []
        for _ in range(triads):
            stanza_sizes.extend([strophe_lines, strophe_lines, epode_lines])

        total_lines = sum(stanza_sizes)
        total_stanzas = 3 * triads

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(total_stanzas, weight=1.5)
        self._stanza_sizes = StanzaSizes(stanza_sizes, weight=1.5)

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.5),
            (self._stanza_sizes, 1.5),
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
        return (
            f"Pindaric Ode: {self.triads} triad(s), "
            f"strophe/antistrophe {self.strophe_lines} lines, epode {self.epode_lines} lines"
        )


class IrregularOde(Constraint):
    """
    Irregular ode: Free form with no fixed structure.

    Also called Cowleyan ode (after Abraham Cowley).
    Features:
    - Variable line lengths
    - Variable stanza sizes
    - No fixed rhyme scheme
    - Minimum line count only

    Examples:
        >>> ode = IrregularOde(min_lines=20)
        >>> result = ode.verify(poem)
    """

    name = "Irregular Ode"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        min_lines: int = 15,
        min_stanzas: int = 3,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize irregular ode constraint.

        Args:
            min_lines: Minimum number of lines
            min_stanzas: Minimum number of stanzas
            weight: Relative weight for composition
        """
        super().__init__(weight)
        self.min_lines = min_lines
        self.min_stanzas = min_stanzas

        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=2.0)
        self._stanza_count = StanzaCount(NumericBound.at_least(min_stanzas), weight=1.5)

        self._constraint = And([self._line_count, self._stanza_count])

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
        return f"Irregular Ode: at least {self.min_lines} lines in {self.min_stanzas}+ stanzas"
