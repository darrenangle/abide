"""
Meter and scansion constraints for poetry.

Provides constraints for enforcing metrical patterns like
iambic pentameter, trochaic tetrameter, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints.base import Constraint
from abide.constraints.types import (
    ConstraintType,
    RubricItem,
    VerificationResult,
)
from abide.primitives import (
    FootLength,
    MeterType,
    get_expected_syllables,
    meter_score,
    scan_line,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Meter(Constraint):
    """
    Constraint on metrical pattern for a poem.

    Verifies that lines follow a specific meter (iamb, trochee, etc.)
    with a specified number of feet (pentameter, tetrameter, etc.).

    Examples:
        >>> # Iambic pentameter (10 syllables, 5 iambs)
        >>> meter = Meter(MeterType.IAMB, FootLength.PENTAMETER)
        >>> result = meter.verify("Shall I compare thee to a summer's day")

        >>> # Trochaic tetrameter
        >>> meter = Meter(MeterType.TROCHEE, FootLength.TETRAMETER)

        >>> # With tolerance for substitutions
        >>> meter = Meter(MeterType.IAMB, FootLength.PENTAMETER, tolerance=1)
    """

    name = "Meter"
    constraint_type = ConstraintType.PROSODIC

    def __init__(
        self,
        meter_type: MeterType,
        foot_length: FootLength | int,
        tolerance: int = 0,
        min_score: float = 0.7,
        per_line: list[tuple[MeterType, int]] | None = None,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize meter constraint.

        Args:
            meter_type: The type of metrical foot (IAMB, TROCHEE, etc.)
            foot_length: Number of feet per line (PENTAMETER=5, etc.)
            tolerance: Allowed deviation in foot count
            min_score: Minimum meter match score per line (0-1)
            per_line: Override with specific (MeterType, feet) per line
            weight: Relative weight for composition
        """
        super().__init__(weight)
        self.meter_type = meter_type
        self.foot_count = (
            foot_length.value if isinstance(foot_length, FootLength) else foot_length
        )
        self.tolerance = tolerance
        self.min_score = min_score
        self.per_line = per_line

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        rubric: list[RubricItem] = []
        scores: list[float] = []

        for i, line in enumerate(structure.lines):
            # Get expected meter for this line
            if self.per_line and i < len(self.per_line):
                expected_meter, expected_feet = self.per_line[i]
            else:
                expected_meter = self.meter_type
                expected_feet = self.foot_count

            # Score the line
            line_score = meter_score(
                line,
                expected_meter,
                expected_feet,
                self.tolerance,
            )

            # Get scan result for details
            scan_result = scan_line(line)
            passed = line_score >= self.min_score

            # Build description
            expected_syllables = get_expected_syllables(expected_meter, expected_feet)
            meter_name = f"{expected_meter.value} {self._foot_length_name(expected_feet)}"

            rubric.append(
                RubricItem(
                    criterion=f"Line {i + 1} meter",
                    expected=f"{meter_name} ({expected_syllables} syllables)",
                    actual=f"{scan_result.dominant_meter.value if scan_result.dominant_meter else 'none'} "
                    f"({scan_result.syllable_count} syllables, {scan_result.foot_count} feet, "
                    f"regularity {scan_result.regularity:.0%})",
                    score=line_score,
                    passed=passed,
                )
            )
            scores.append(line_score)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "meter_type": self.meter_type.value,
                "foot_count": self.foot_count,
                "tolerance": self.tolerance,
                "min_score": self.min_score,
            },
        )

    def _foot_length_name(self, count: int) -> str:
        """Get the name for a foot count."""
        names = {
            1: "monometer",
            2: "dimeter",
            3: "trimeter",
            4: "tetrameter",
            5: "pentameter",
            6: "hexameter",
            7: "heptameter",
            8: "octameter",
        }
        return names.get(count, f"{count}-foot")

    def describe(self) -> str:
        meter_name = f"{self.meter_type.value} {self._foot_length_name(self.foot_count)}"
        if self.tolerance > 0:
            return f"{meter_name.title()} (±{self.tolerance} feet)"
        return meter_name.title()

    def instruction(self) -> str:
        """Generate plain English instruction."""
        meter_name = self.meter_type.value
        length_name = self._foot_length_name(self.foot_count)
        expected_syllables = get_expected_syllables(self.meter_type, self.foot_count)

        foot_desc = {
            MeterType.IAMB: "unstressed-stressed (da-DUM)",
            MeterType.TROCHEE: "stressed-unstressed (DUM-da)",
            MeterType.ANAPEST: "unstressed-unstressed-stressed (da-da-DUM)",
            MeterType.DACTYL: "stressed-unstressed-unstressed (DUM-da-da)",
            MeterType.SPONDEE: "stressed-stressed (DUM-DUM)",
            MeterType.PYRRHIC: "unstressed-unstressed (da-da)",
            MeterType.AMPHIBRACH: "unstressed-stressed-unstressed (da-DUM-da)",
        }.get(self.meter_type, "")

        return (
            f"Write in {meter_name} {length_name}: each line should have "
            f"{self.foot_count} {meter_name} feet ({foot_desc}), "
            f"totaling approximately {expected_syllables} syllables per line."
        )


class MeterPattern(Constraint):
    """
    Constraint for poems with varying meter per stanza or line group.

    Useful for forms like Common Meter (alternating 8-6-8-6 syllables)
    or ballads with specific metrical patterns.

    Example:
        >>> # Common Meter: iambic 4-3-4-3 feet pattern
        >>> pattern = MeterPattern(
        ...     meter_type=MeterType.IAMB,
        ...     foot_pattern=[4, 3, 4, 3],
        ... )
    """

    name = "Meter Pattern"
    constraint_type = ConstraintType.PROSODIC

    def __init__(
        self,
        meter_type: MeterType,
        foot_pattern: list[int],
        tolerance: int = 0,
        min_score: float = 0.7,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize meter pattern constraint.

        Args:
            meter_type: The type of metrical foot
            foot_pattern: List of foot counts that repeats across lines
            tolerance: Allowed deviation in foot count
            min_score: Minimum meter match score per line
            weight: Relative weight for composition
        """
        super().__init__(weight)
        self.meter_type = meter_type
        self.foot_pattern = foot_pattern
        self.tolerance = tolerance
        self.min_score = min_score

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        rubric: list[RubricItem] = []
        scores: list[float] = []

        for i, line in enumerate(structure.lines):
            # Get expected feet for this line (cycling through pattern)
            expected_feet = self.foot_pattern[i % len(self.foot_pattern)]

            # Score the line
            line_score = meter_score(
                line,
                self.meter_type,
                expected_feet,
                self.tolerance,
            )

            scan_result = scan_line(line)
            passed = line_score >= self.min_score

            expected_syllables = get_expected_syllables(self.meter_type, expected_feet)

            rubric.append(
                RubricItem(
                    criterion=f"Line {i + 1} meter",
                    expected=f"{expected_feet} {self.meter_type.value} feet ({expected_syllables} syllables)",
                    actual=f"{scan_result.foot_count} feet ({scan_result.syllable_count} syllables)",
                    score=line_score,
                    passed=passed,
                )
            )
            scores.append(line_score)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "meter_type": self.meter_type.value,
                "foot_pattern": self.foot_pattern,
            },
        )

    def describe(self) -> str:
        pattern_str = "-".join(str(f) for f in self.foot_pattern)
        return f"{self.meter_type.value.title()} pattern: {pattern_str} feet"

    def instruction(self) -> str:
        """Generate plain English instruction."""
        syllable_pattern = [
            get_expected_syllables(self.meter_type, f) for f in self.foot_pattern
        ]
        pattern_str = "-".join(str(s) for s in syllable_pattern)
        return (
            f"Write in {self.meter_type.value} meter with a repeating syllable "
            f"pattern of {pattern_str} syllables per line."
        )
