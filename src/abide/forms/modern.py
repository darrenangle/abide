"""
Modern and experimental poetic forms.

Bop, Aubade, Skeltonic verse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    NumericBound,
    Refrain,
    RhymeScheme,
    StanzaCount,
    StanzaSizes,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Bop(Constraint):
    """
    Bop: Contemporary form with repeated refrain line.

    Structure:
    - 3 stanzas
    - Stanza 1: 6 lines + 1-line refrain = 7 lines
    - Stanza 2: 8 lines + 1-line refrain = 9 lines
    - Stanza 3: 6 lines + 1-line refrain = 7 lines
    - Total: 23 lines
    - The same refrain line ends each stanza
    - Often based on a line from music (hence "bop")

    Examples:
        >>> bop = Bop()
        >>> result = bop.verify(poem)
    """

    name = "Bop"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        """
        Initialize bop constraint.

        Args:
            weight: Relative weight for composition
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.strict_mode = strict

        # 3 stanzas: 6+1, 8+1, 6+1 = 23 lines total
        self._line_count = LineCount(23, weight=2.0)
        self._stanza_count = StanzaCount(3, weight=1.5)
        self._stanza_sizes = StanzaSizes([7, 9, 7], weight=1.5)

        # Refrain: last line of each stanza should match
        # Lines 7, 16, 23 (0-indexed: 6, 15, 22)
        self._refrain = Refrain(
            reference_line=6,  # First refrain (end of stanza 1)
            repeat_at=[15, 22],  # End of stanzas 2 and 3
            weight=2.0,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.5),
            (self._stanza_sizes, 1.5),
            (self._refrain, 2.0),
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
        return "Bop: 23 lines in 3 stanzas (6+8+6) with refrain"


class Aubade(Constraint):
    """
    Aubade: Dawn poem or morning song.

    Structure is flexible, but typically:
    - Multiple stanzas
    - Often addresses a lover or the dawn itself
    - May include refrain
    - Usually moderate length (16-40 lines)

    This implements a common 4-stanza version.

    Examples:
        >>> aubade = Aubade()
        >>> result = aubade.verify(poem)
    """

    name = "Aubade"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        stanza_count: int = 4,
        lines_per_stanza: int = 6,
        weight: float = 1.0,
        rhyme_threshold: float = 0.5,
        strict: bool = False,
    ) -> None:
        """
        Initialize aubade constraint.

        Args:
            stanza_count: Number of stanzas
            lines_per_stanza: Lines per stanza
            weight: Relative weight for composition
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.lines_per_stanza = lines_per_stanza
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        total_lines = stanza_count * lines_per_stanza

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.5)
        self._stanza_sizes = StanzaSizes(
            [lines_per_stanza] * stanza_count, weight=1.0
        )

        # Common aubade rhyme: ABABCC per stanza
        stanza_rhyme = "ABABCC"[:lines_per_stanza]
        full_rhyme = stanza_rhyme * stanza_count

        self._rhyme = RhymeScheme(
            full_rhyme,
            weight=1.5,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.5),
            (self._stanza_sizes, 1.0),
            (self._rhyme, 1.5),
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
        total = self.stanza_count_val * self.lines_per_stanza
        return f"Aubade: {total} lines in {self.stanza_count_val} stanzas (dawn poem)"


class Skeltonic(Constraint):
    """
    Skeltonic verse (tumbling verse): Short lines with heavy rhyme.

    Named after John Skelton. Features:
    - Very short lines (2-5 stresses)
    - Rapid rhyming (same rhyme repeated multiple times)
    - Irregular rhythm
    - Often satirical or comic

    This is a flexible form - we check for short lines and heavy rhyming.

    Examples:
        >>> skeltonic = Skeltonic()
        >>> result = skeltonic.verify(poem)
    """

    name = "Skeltonic Verse"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        min_lines: int = 10,
        max_syllables_per_line: int = 8,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        """
        Initialize skeltonic verse constraint.

        Args:
            min_lines: Minimum number of lines
            max_syllables_per_line: Maximum syllables (keeps lines short)
            weight: Relative weight for composition
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_lines = min_lines
        self.max_syllables = max_syllables_per_line
        self.strict_mode = strict

        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check minimum lines
        line_result = self._line_count.verify(poem)

        # Check for short lines
        from abide.primitives.phonetics import count_line_syllables

        line_syllables = [count_line_syllables(line) for line in structure.lines]
        short_line_count = sum(1 for s in line_syllables if s <= self.max_syllables)
        short_line_score = short_line_count / max(1, len(structure.lines))

        # Check for heavy rhyming (consecutive rhymes)
        # Skeltonic verse often has 3+ lines rhyming together
        from abide.primitives.phonetics import get_end_sound

        rhyme_runs = 0
        current_run = 1
        prev_sound = None

        for line in structure.lines:
            sound = get_end_sound(line)
            if sound and sound == prev_sound:
                current_run += 1
            else:
                if current_run >= 3:
                    rhyme_runs += 1
                current_run = 1
            prev_sound = sound

        if current_run >= 3:
            rhyme_runs += 1

        # Score based on number of rhyme runs (more = more Skeltonic)
        rhyme_score = min(1.0, rhyme_runs / 3)  # Expect ~3 runs of 3+ rhymes

        # Combine scores
        score = (
            line_result.score * 0.3 + short_line_score * 0.4 + rhyme_score * 0.3
        )
        passed = score >= 0.6

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=line_result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "line_count": len(structure.lines),
                "short_lines": short_line_count,
                "short_line_ratio": short_line_score,
                "rhyme_runs": rhyme_runs,
            },
        )

    def describe(self) -> str:
        return f"Skeltonic Verse: {self.min_lines}+ short lines with heavy rhyming"
