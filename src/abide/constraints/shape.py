"""
Shape and visual structure constraints for poetry.

Provides constraints for poems with visual/shape requirements like
diamante, concrete poetry, and other forms where line length matters.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from abide.constraints.base import Constraint
from abide.constraints.types import (
    ConstraintType,
    RubricItem,
    VerificationResult,
)
from abide.primitives import count_line_syllables

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class ShapeType(Enum):
    """Predefined shape patterns for visual poetry."""

    DIAMOND = "diamond"  # Lines get longer then shorter (diamante)
    TRIANGLE_UP = "triangle_up"  # Lines get progressively longer
    TRIANGLE_DOWN = "triangle_down"  # Lines get progressively shorter
    HOURGLASS = "hourglass"  # Short-long-short pattern
    RECTANGLE = "rectangle"  # All lines same length
    CUSTOM = "custom"  # User-defined pattern


class MeasureMode(Enum):
    """How to measure line length."""

    CHARACTERS = "characters"  # Count characters (including spaces)
    CHARACTERS_NO_SPACE = "characters_no_space"  # Count non-space characters
    SYLLABLES = "syllables"  # Count syllables
    WORDS = "words"  # Count words


class LineShape(Constraint):
    """
    Constraint on line length pattern for visual/shape poetry.

    Can measure by characters, syllables, or words.
    Supports predefined shapes (diamond, triangle) or custom patterns.

    Examples:
        >>> # Diamante - lines get longer then shorter
        >>> shape = LineShape(shape_type=ShapeType.DIAMOND, num_lines=7)
        >>> result = shape.verify(poem)

        >>> # Custom pattern with exact lengths
        >>> shape = LineShape(lengths=[1, 2, 3, 4, 3, 2, 1], mode=MeasureMode.WORDS)
        >>> result = shape.verify(poem)

        >>> # Triangle with relative comparison
        >>> shape = LineShape(shape_type=ShapeType.TRIANGLE_UP, num_lines=5)
    """

    name = "Line Shape"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        shape_type: ShapeType | None = None,
        num_lines: int | None = None,
        lengths: list[int] | None = None,
        mode: MeasureMode = MeasureMode.SYLLABLES,
        tolerance: int = 0,
        relative: bool = True,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize line shape constraint.

        Args:
            shape_type: Predefined shape pattern (DIAMOND, TRIANGLE_UP, etc.)
            num_lines: Number of lines for shape generation (used with shape_type)
            lengths: Explicit length for each line (mutually exclusive with shape_type)
            mode: How to measure line length (characters, syllables, words)
            tolerance: Allowed deviation from expected length
            relative: If True, check relative ordering rather than exact lengths
            weight: Relative weight for composition
        """
        super().__init__(weight)
        self.shape_type = shape_type
        self.num_lines = num_lines
        self.lengths = lengths
        self.mode = mode
        self.tolerance = tolerance
        self.relative = relative

        # Generate expected lengths from shape_type if not provided
        if lengths is None and shape_type is not None and num_lines is not None:
            self.lengths = self._generate_shape(shape_type, num_lines)

    def _generate_shape(self, shape_type: ShapeType, num_lines: int) -> list[int]:
        """Generate length pattern for a shape type."""
        if shape_type == ShapeType.DIAMOND:
            # 1, 2, 3, 4, 3, 2, 1 for 7 lines
            mid = (num_lines + 1) // 2
            lengths = list(range(1, mid + 1))
            if num_lines % 2 == 0:
                lengths += list(range(mid, 0, -1))
            else:
                lengths += list(range(mid - 1, 0, -1))
            return lengths

        elif shape_type == ShapeType.TRIANGLE_UP:
            # 1, 2, 3, 4, 5 for 5 lines
            return list(range(1, num_lines + 1))

        elif shape_type == ShapeType.TRIANGLE_DOWN:
            # 5, 4, 3, 2, 1 for 5 lines
            return list(range(num_lines, 0, -1))

        elif shape_type == ShapeType.HOURGLASS:
            # 3, 2, 1, 2, 3 for 5 lines
            mid = (num_lines + 1) // 2
            lengths = list(range(mid, 0, -1))
            if num_lines % 2 == 0:
                lengths += list(range(1, mid + 1))
            else:
                lengths += list(range(2, mid + 1))
            return lengths

        elif shape_type == ShapeType.RECTANGLE:
            # All same length (we'll use 5 as default)
            return [5] * num_lines

        return [1] * num_lines

    def _measure_line(self, line: str) -> int:
        """Measure a line according to the configured mode."""
        if self.mode == MeasureMode.CHARACTERS:
            return len(line)
        elif self.mode == MeasureMode.CHARACTERS_NO_SPACE:
            return len(line.replace(" ", ""))
        elif self.mode == MeasureMode.SYLLABLES:
            return count_line_syllables(line)
        elif self.mode == MeasureMode.WORDS:
            return len(line.split())
        return len(line)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        rubric: list[RubricItem] = []
        scores: list[float] = []

        actual_lengths = [self._measure_line(line) for line in structure.lines]

        if self.lengths is None:
            # No specific lengths, just report what we found
            rubric.append(
                RubricItem(
                    criterion="Line lengths",
                    expected="any",
                    actual=str(actual_lengths),
                    score=1.0,
                    passed=True,
                )
            )
            return VerificationResult(
                score=1.0,
                passed=True,
                rubric=rubric,
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"actual_lengths": actual_lengths, "mode": self.mode.value},
            )

        # Check line count matches
        if len(actual_lengths) != len(self.lengths):
            rubric.append(
                RubricItem(
                    criterion="Line count",
                    expected=str(len(self.lengths)),
                    actual=str(len(actual_lengths)),
                    score=0.5,
                    passed=False,
                )
            )
            scores.append(0.5)

        if self.relative:
            # Check relative ordering (each line compared to next)
            score = self._check_relative_shape(actual_lengths, rubric)
            scores.append(score)
        else:
            # Check exact lengths with tolerance
            for i, (actual, expected) in enumerate(zip(actual_lengths, self.lengths)):
                diff = abs(actual - expected)
                passed = diff <= self.tolerance

                if diff == 0:
                    line_score = 1.0
                elif diff <= self.tolerance:
                    line_score = 1.0 - (diff / (self.tolerance + 1)) * 0.2
                else:
                    line_score = max(0.0, 1.0 - diff / max(expected, 1))

                rubric.append(
                    RubricItem(
                        criterion=f"Line {i + 1} length",
                        expected=f"{expected} (±{self.tolerance})",
                        actual=str(actual),
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
                "expected_lengths": self.lengths,
                "actual_lengths": actual_lengths,
                "mode": self.mode.value,
                "shape_type": self.shape_type.value if self.shape_type else None,
            },
        )

    def _check_relative_shape(
        self, actual: list[int], rubric: list[RubricItem]
    ) -> float:
        """Check that lines follow the relative shape pattern."""
        if not self.lengths or len(actual) < 2:
            return 1.0

        correct = 0
        total = 0

        for i in range(len(self.lengths) - 1):
            if i >= len(actual) - 1:
                break

            expected_direction = self._compare(self.lengths[i], self.lengths[i + 1])
            actual_direction = self._compare(actual[i], actual[i + 1])

            total += 1
            if expected_direction == actual_direction:
                correct += 1
                passed = True
            else:
                passed = False

            direction_names = {-1: "shorter", 0: "equal", 1: "longer"}
            rubric.append(
                RubricItem(
                    criterion=f"Line {i + 1} → {i + 2}",
                    expected=f"L{i + 2} should be {direction_names[expected_direction]} than L{i + 1}",
                    actual=f"L{i + 2} is {direction_names[actual_direction]} ({actual[i]} → {actual[i + 1]})",
                    score=1.0 if passed else 0.0,
                    passed=passed,
                )
            )

        return correct / total if total > 0 else 1.0

    def _compare(self, a: int, b: int) -> int:
        """Compare two values: -1 if a > b, 0 if equal, 1 if a < b."""
        if a < b:
            return 1
        elif a > b:
            return -1
        return 0

    def describe(self) -> str:
        mode_name = self.mode.value
        if self.shape_type:
            return f"Lines follow {self.shape_type.value} shape (measured by {mode_name})"
        elif self.lengths:
            return f"Line lengths: {self.lengths} ({mode_name})"
        return f"Line shape pattern ({mode_name})"

    def instruction(self) -> str:
        """Generate plain English instruction."""
        mode_name = self.mode.value
        if self.shape_type == ShapeType.DIAMOND:
            return f"Write lines that form a diamond shape: start short, grow to the middle, then shrink back (measured by {mode_name})."
        elif self.shape_type == ShapeType.TRIANGLE_UP:
            return f"Write lines that progressively get longer from first to last (measured by {mode_name})."
        elif self.shape_type == ShapeType.TRIANGLE_DOWN:
            return f"Write lines that progressively get shorter from first to last (measured by {mode_name})."
        elif self.shape_type == ShapeType.HOURGLASS:
            return f"Write lines that form an hourglass shape: start long, shrink to the middle, then grow back (measured by {mode_name})."
        elif self.lengths:
            if self.relative:
                return f"Follow this relative length pattern: {self.lengths} (measured by {mode_name})."
            else:
                return f"Write lines with these exact lengths: {self.lengths} {mode_name} (±{self.tolerance})."
        return f"Follow the specified line shape pattern."


class LineLengthRange(Constraint):
    """
    Constraint on line length within a range.

    Useful for forms that require lines to stay within bounds
    without being exact.

    Example:
        >>> # All lines between 8-12 syllables
        >>> constraint = LineLengthRange(min_length=8, max_length=12)
    """

    name = "Line Length Range"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        mode: MeasureMode = MeasureMode.SYLLABLES,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.min_length = min_length
        self.max_length = max_length
        self.mode = mode

    def _measure_line(self, line: str) -> int:
        """Measure a line according to the configured mode."""
        if self.mode == MeasureMode.CHARACTERS:
            return len(line)
        elif self.mode == MeasureMode.CHARACTERS_NO_SPACE:
            return len(line.replace(" ", ""))
        elif self.mode == MeasureMode.SYLLABLES:
            return count_line_syllables(line)
        elif self.mode == MeasureMode.WORDS:
            return len(line.split())
        return len(line)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        rubric: list[RubricItem] = []
        scores: list[float] = []

        for i, line in enumerate(structure.lines):
            length = self._measure_line(line)

            # Check bounds
            below_min = self.min_length is not None and length < self.min_length
            above_max = self.max_length is not None and length > self.max_length
            passed = not below_min and not above_max

            if passed:
                score = 1.0
            else:
                # Partial credit based on how far out of bounds
                if below_min:
                    score = length / self.min_length
                else:  # above_max
                    score = self.max_length / length

            bounds = []
            if self.min_length is not None:
                bounds.append(f"≥{self.min_length}")
            if self.max_length is not None:
                bounds.append(f"≤{self.max_length}")

            rubric.append(
                RubricItem(
                    criterion=f"Line {i + 1} length",
                    expected=" and ".join(bounds) if bounds else "any",
                    actual=str(length),
                    score=score,
                    passed=passed,
                )
            )
            scores.append(score)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "min_length": self.min_length,
                "max_length": self.max_length,
                "mode": self.mode.value,
            },
        )

    def describe(self) -> str:
        bounds = []
        if self.min_length is not None:
            bounds.append(f"≥{self.min_length}")
        if self.max_length is not None:
            bounds.append(f"≤{self.max_length}")
        return f"Line lengths {' and '.join(bounds)} ({self.mode.value})"

    def instruction(self) -> str:
        mode_name = self.mode.value
        if self.min_length is not None and self.max_length is not None:
            return f"Each line should have between {self.min_length} and {self.max_length} {mode_name}."
        elif self.min_length is not None:
            return f"Each line should have at least {self.min_length} {mode_name}."
        elif self.max_length is not None:
            return f"Each line should have at most {self.max_length} {mode_name}."
        return "Line lengths are unconstrained."
