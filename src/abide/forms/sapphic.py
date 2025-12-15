"""
Sapphic stanza/ode form template.

Named after the ancient Greek poet Sappho from Lesbos.
Classical quantitative meter adapted to syllabic/accentual verse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    StanzaCount,
    StanzaSizes,
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class SapphicStanza(Constraint):
    """
    Sapphic Stanza: Classical 4-line stanza form.

    Structure:
    - Lines 1-3: 11 syllables each (Sapphic hendecasyllables)
    - Line 4: 5 syllables (Adonic line)

    The classical meter is: --u-u--u-u- for hendecasyllables
    and -uu-- for the Adonic.

    In English, often adapted as 11-11-11-5 syllables.

    Examples:
        >>> sapphic = SapphicStanza(stanza_count=2)
        >>> result = sapphic.verify(poem)
    """

    name = "Sapphic Stanza"
    constraint_type = ConstraintType.COMPOSITE

    # 11-11-11-5 syllable pattern
    SYLLABLE_PATTERN: ClassVar[list[int]] = [11, 11, 11, 5]

    def __init__(
        self,
        stanza_count: int = 1,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        strict: bool = False,
    ) -> None:
        """
        Initialize Sapphic stanza constraint.

        Args:
            stanza_count: Number of stanzas
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.syllable_tolerance = syllable_tolerance
        self.strict_mode = strict

        total_lines = stanza_count * 4

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.0)
        self._stanza_sizes = StanzaSizes([4] * stanza_count, weight=1.0)

        # Repeat syllable pattern for each stanza
        syllable_pattern = self.SYLLABLE_PATTERN * stanza_count
        self._syllables = SyllablesPerLine(
            syllable_pattern,
            weight=2.0,
            tolerance=syllable_tolerance,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.0),
            (self._syllables, 2.0),
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
            f"Sapphic Stanza: {self.stanza_count_val} stanza(s) of 4 lines (11-11-11-5 syllables)"
        )


class SapphicOde(Constraint):
    """
    Sapphic Ode: Multiple Sapphic stanzas forming an ode.

    A poem of at least 3 Sapphic stanzas (12+ lines).

    Examples:
        >>> ode = SapphicOde(min_stanzas=4)
        >>> result = ode.verify(poem)
    """

    name = "Sapphic Ode"
    constraint_type = ConstraintType.COMPOSITE

    SYLLABLE_PATTERN: ClassVar[list[int]] = [11, 11, 11, 5]

    def __init__(
        self,
        min_stanzas: int = 3,
        max_stanzas: int | None = None,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        strict: bool = False,
    ) -> None:
        """
        Initialize Sapphic ode constraint.

        Args:
            min_stanzas: Minimum number of stanzas
            max_stanzas: Maximum stanzas (None for unlimited)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_stanzas = min_stanzas
        self.max_stanzas = max_stanzas
        self.syllable_tolerance = syllable_tolerance
        self.strict_mode = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check stanza count - quadratic penalty for stricter GRPO training
        stanza_count = structure.stanza_count

        if stanza_count < self.min_stanzas:
            linear_stanza = stanza_count / self.min_stanzas
            stanza_score = linear_stanza**2
        elif self.max_stanzas and stanza_count > self.max_stanzas:
            linear_stanza = self.max_stanzas / stanza_count
            stanza_score = linear_stanza**2
        else:
            stanza_score = 1.0

        # Check stanza sizes (should all be 4)
        sizes_correct = sum(1 for s in structure.stanzas if len(s) == 4)
        # Quadratic penalty for stricter GRPO training
        linear_size = sizes_correct / max(1, stanza_count)
        size_score = linear_size**2

        # Check syllable pattern
        from abide.primitives.phonetics import count_line_syllables

        expected_pattern = self.SYLLABLE_PATTERN * stanza_count
        actual_counts = [count_line_syllables(line) for line in structure.lines]

        if len(actual_counts) < len(expected_pattern):
            # Pad expected to match actual
            expected_pattern = expected_pattern[: len(actual_counts)]
        elif len(actual_counts) > len(expected_pattern):
            # Extend expected pattern
            while len(expected_pattern) < len(actual_counts):
                expected_pattern.extend(self.SYLLABLE_PATTERN)
            expected_pattern = expected_pattern[: len(actual_counts)]

        syllable_matches = 0
        for actual, expected in zip(actual_counts, expected_pattern):
            if abs(actual - expected) <= self.syllable_tolerance:
                syllable_matches += 1
        # Quadratic penalty for stricter GRPO training
        linear_syllable = syllable_matches / max(1, len(actual_counts))
        syllable_score = linear_syllable**2

        # Combine scores - syllable pattern (11-11-11-5) is THE defining characteristic
        # It gets 80% weight. Stanza count/size are secondary
        score = stanza_score * 0.1 + size_score * 0.1 + syllable_score * 0.8
        passed = score >= 0.6 and stanza_count >= self.min_stanzas

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "stanza_count": stanza_count,
                "min_stanzas": self.min_stanzas,
                "stanza_score": stanza_score,
                "size_score": size_score,
                "syllable_score": syllable_score,
            },
        )

    def describe(self) -> str:
        max_str = f"-{self.max_stanzas}" if self.max_stanzas else "+"
        return f"Sapphic Ode: {self.min_stanzas}{max_str} stanzas (11-11-11-5 pattern)"
