"""
Blank verse form template.

Blank verse is unrhymed iambic pentameter.
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
    SyllablesPerLine,
    VerificationResult,
    WeightedSum,
)
from abide.primitives import FootLength, MeterType

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class BlankVerse(Constraint):
    """
    Blank verse: unrhymed iambic pentameter.

    The meter of Shakespeare's plays, Milton's Paradise Lost,
    and much English dramatic and epic poetry.

    Structure:
    - Iambic pentameter (10 syllables, 5 iambs per line)
    - No rhyme scheme required
    - Variable line count

    Examples:
        >>> blank = BlankVerse(min_lines=10)
        >>> result = blank.verify(poem)

        >>> # With strict meter enforcement
        >>> blank = BlankVerse(strict_meter=True)
    """

    name = "Blank Verse"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        min_lines: int = 3,
        max_lines: int | None = None,
        weight: float = 1.0,
        syllable_tolerance: int = 1,
        strict_meter: bool = False,
        meter_threshold: float = 0.7,
        strict: bool = False,
    ) -> None:
        """
        Initialize blank verse constraint.

        Args:
            min_lines: Minimum number of lines
            max_lines: Maximum number of lines (None for unlimited)
            weight: Relative weight for composition
            syllable_tolerance: Allow +/- this many syllables per line
            strict_meter: If True, enforce actual iambic meter (not just syllable count)
            meter_threshold: Minimum meter score per line (for strict_meter)
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.syllable_tolerance = syllable_tolerance
        self.strict_meter = strict_meter
        self.meter_threshold = meter_threshold
        self.strict_mode = strict

        # Build constraints list
        constraints: list[tuple[Constraint, float]] = []

        # Line count (at least min_lines)
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.5)
        constraints.append((self._line_count, 1.5))

        if strict_meter:
            # Use actual meter detection
            self._meter = Meter(
                MeterType.IAMB,
                FootLength.PENTAMETER,
                min_score=meter_threshold,
                tolerance=1,
                weight=2.0,
            )
            constraints.append((self._meter, 2.0))
        else:
            # Just check syllable count (simpler, faster)
            # Use a placeholder for line count - will be checked dynamically
            self._syllables: SyllablesPerLine | None = None

        self._base_constraints = constraints
        self._strict = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        constraints = list(self._base_constraints)

        # Add syllable constraint based on actual line count
        if not self.strict_meter:
            syllables = SyllablesPerLine(
                [10] * structure.line_count,
                weight=2.0,
                tolerance=self.syllable_tolerance,
            )
            constraints.append((syllables, 2.0))

        # Check max lines if specified
        if self.max_lines is not None and structure.line_count > self.max_lines:
            return VerificationResult(
                score=0.5,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": f"Exceeds max lines ({self.max_lines})"},
            )

        constraint: Constraint
        if self._strict:
            constraint = And([c for c, _ in constraints])
        else:
            constraint = WeightedSum(constraints, threshold=0.6)

        result = constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        desc = f"Blank Verse: unrhymed iambic pentameter (min {self.min_lines} lines)"
        if self.strict_meter:
            desc += " [strict meter]"
        return desc


class DramaticVerse(BlankVerse):
    """
    Dramatic verse: blank verse with more flexibility for speech patterns.

    Allows greater variation in meter for natural dialogue.

    Examples:
        >>> verse = DramaticVerse()
        >>> result = verse.verify(speech)
    """

    name = "Dramatic Verse"

    def __init__(
        self,
        min_lines: int = 1,
        weight: float = 1.0,
        syllable_tolerance: int = 2,  # More flexible
        strict: bool = False,
    ) -> None:
        super().__init__(
            min_lines=min_lines,
            weight=weight,
            syllable_tolerance=syllable_tolerance,
            strict_meter=False,
            strict=strict,
        )

    def describe(self) -> str:
        return "Dramatic Verse: flexible blank verse for speech"
