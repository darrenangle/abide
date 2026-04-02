"""
Structural constraints for poem verification.

Line count, stanza count, stanza sizes, syllables per line, etc.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from abide.constraints._validation import (
    require_nonnegative,
    require_positive,
    require_positive_numeric_bound,
)
from abide.constraints.base import Constraint, NumericConstraint
from abide.constraints.types import (
    ConstraintType,
    NumericBound,
    RubricItem,
    VerificationResult,
)
from abide.primitives import count_line_syllables

if TYPE_CHECKING:
    from collections.abc import Sequence

    from abide.primitives import PoemStructure


class LineCount(NumericConstraint):
    """
    Constraint on the number of lines in a poem.

    Examples:
        >>> constraint = LineCount(14)  # Exactly 14 lines (sonnet)
        >>> constraint = LineCount(NumericBound.at_least(10))  # At least 10
    """

    name = "Line Count"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        bound: int | NumericBound,
        weight: float = 1.0,
        sigma: float = 0.9,
    ) -> None:
        if isinstance(bound, int):
            bound = NumericBound.exact(bound)
        require_positive_numeric_bound(bound, "line count")
        super().__init__(bound, weight, sigma)

    def _get_actual_value(self, structure: PoemStructure) -> int:
        return structure.line_count

    def _get_criterion_name(self) -> str:
        return "lines"

    def instruction(self) -> str:
        """Plain English instruction for LLM prompts."""
        return f"Write exactly {self.bound.describe()} lines."


class StanzaCount(NumericConstraint):
    """
    Constraint on the number of stanzas in a poem.

    Examples:
        >>> constraint = StanzaCount(6)  # Exactly 6 stanzas (villanelle)
        >>> constraint = StanzaCount(NumericBound.at_least(2))
    """

    name = "Stanza Count"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        bound: int | NumericBound,
        weight: float = 1.0,
        sigma: float = 1.0,
    ) -> None:
        if isinstance(bound, int):
            bound = NumericBound.exact(bound)
        require_positive_numeric_bound(bound, "stanza count")
        super().__init__(bound, weight, sigma)

    def _get_actual_value(self, structure: PoemStructure) -> int:
        return structure.stanza_count

    def _get_criterion_name(self) -> str:
        return "stanzas"

    def instruction(self) -> str:
        """Plain English instruction for LLM prompts."""
        return f"Organize the poem into {self.bound.describe()} stanzas (verse paragraphs separated by blank lines)."


class StanzaSizes(Constraint):
    """
    Constraint on the size (line count) of each stanza.

    Examples:
        >>> constraint = StanzaSizes([3, 3, 3, 3, 3, 4])  # Villanelle
        >>> constraint = StanzaSizes([6, 6, 6, 6, 6, 6, 3])  # Sestina
    """

    name = "Stanza Sizes"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        expected_sizes: Sequence[int],
        weight: float = 1.0,
        sigma: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.expected_sizes = tuple(expected_sizes)
        if not self.expected_sizes or any(size <= 0 for size in self.expected_sizes):
            raise ValueError("expected_sizes must contain at least one positive stanza size")
        require_positive(sigma, "sigma")
        self.sigma = sigma

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        actual_sizes = structure.stanza_sizes

        rubric: list[RubricItem] = []
        scores: list[float] = []

        # First check stanza count
        if len(actual_sizes) != len(self.expected_sizes):
            stanza_score = 0.0
            # Partial credit based on how many stanzas match
            min_len = min(len(actual_sizes), len(self.expected_sizes))
            if min_len > 0:
                matching = sum(
                    1 for i in range(min_len) if actual_sizes[i] == self.expected_sizes[i]
                )
                # Quadratic penalty for stricter GRPO training
                linear_stanza = matching / len(self.expected_sizes)
                stanza_score = linear_stanza**2
        else:
            stanza_score = 1.0

        rubric.append(
            RubricItem(
                criterion="Stanza count",
                expected=str(len(self.expected_sizes)),
                actual=str(len(actual_sizes)),
                score=stanza_score,
                passed=len(actual_sizes) == len(self.expected_sizes),
            )
        )
        scores.append(stanza_score)

        # Check each stanza size
        for i, expected in enumerate(self.expected_sizes):
            if i < len(actual_sizes):
                actual = actual_sizes[i]
                passed = actual == expected
                if passed:
                    score = 1.0
                else:
                    diff = abs(actual - expected)
                    score = math.exp(-0.5 * (diff / self.sigma) ** 2)
            else:
                actual = 0
                passed = False
                score = 0.0

            rubric.append(
                RubricItem(
                    criterion=f"Stanza {i + 1} size",
                    expected=str(expected),
                    actual=str(actual),
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
                "expected_sizes": list(self.expected_sizes),
                "actual_sizes": list(actual_sizes),
            },
        )

    def describe(self) -> str:
        sizes_str = ", ".join(str(s) for s in self.expected_sizes)
        return f"Has stanza sizes [{sizes_str}]"

    def instruction(self) -> str:
        """Plain English instruction for LLM prompts."""
        parts = []
        for i, size in enumerate(self.expected_sizes):
            parts.append(f"stanza {i + 1} has {size} lines")
        return "Structure the stanzas so that " + ", ".join(parts) + "."


class GroupedStanzas(Constraint):
    """
    Constraint on repeated stanza/group sizes with an optional final tail.

    This supports forms that are usually written in uniform stanzas but may
    also appear as a single block of lines or with a short closing tail.
    """

    name = "Grouped Stanzas"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        group_size: int,
        min_groups: int = 1,
        *,
        allow_single_block_chunking: bool = True,
        allowed_tail_sizes: Sequence[int] = (),
        weight: float = 1.0,
        sigma: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if min_groups < 0:
            raise ValueError("min_groups must be non-negative")

        self.group_size = group_size
        self.min_groups = min_groups
        self.allow_single_block_chunking = allow_single_block_chunking
        self.allowed_tail_sizes = tuple(sorted(set(allowed_tail_sizes)))
        self.sigma = sigma

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        actual_sizes, chunked_single_block = self._infer_group_sizes(structure)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        full_groups = sum(1 for size in actual_sizes if size == self.group_size)
        count_passed = full_groups >= self.min_groups
        count_score = 1.0 if count_passed else (full_groups / max(1, self.min_groups)) ** 2

        rubric.append(
            RubricItem(
                criterion="Minimum full groups",
                expected=f"at least {self.min_groups} groups of {self.group_size} lines",
                actual=f"{full_groups} group(s) of {self.group_size} lines",
                score=count_score,
                passed=count_passed,
            )
        )
        scores.append(count_score)

        if not actual_sizes:
            return VerificationResult(
                score=count_score,
                passed=False,
                rubric=rubric,
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={
                    "group_size": self.group_size,
                    "min_groups": self.min_groups,
                    "actual_sizes": [],
                    "chunked_single_block": chunked_single_block,
                },
            )

        tail_sizes = set(self.allowed_tail_sizes)
        for index, actual in enumerate(actual_sizes):
            is_last = index == len(actual_sizes) - 1
            last_group_ok = is_last and actual in tail_sizes
            passed = actual == self.group_size or last_group_ok

            if passed:
                score = 1.0
            else:
                valid_targets = [self.group_size, *self.allowed_tail_sizes]
                diff = min(abs(actual - target) for target in valid_targets)
                score = math.exp(-0.5 * (diff / self.sigma) ** 2)

            if is_last and tail_sizes:
                expected = f"{self.group_size} or tail {sorted(tail_sizes)}"
            else:
                expected = str(self.group_size)

            rubric.append(
                RubricItem(
                    criterion=f"Group {index + 1} size",
                    expected=expected,
                    actual=str(actual),
                    score=score,
                    passed=passed,
                )
            )
            scores.append(score)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(item.passed for item in rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "group_size": self.group_size,
                "min_groups": self.min_groups,
                "actual_sizes": actual_sizes,
                "allowed_tail_sizes": list(self.allowed_tail_sizes),
                "chunked_single_block": chunked_single_block,
            },
        )

    def _infer_group_sizes(self, structure: PoemStructure) -> tuple[list[int], bool]:
        if structure.stanza_count > 1 or not self.allow_single_block_chunking:
            return list(structure.stanza_sizes), False

        full_groups, remainder = divmod(structure.line_count, self.group_size)
        sizes = [self.group_size] * full_groups
        if remainder:
            sizes.append(remainder)
        return sizes, True

    def describe(self) -> str:
        desc = f"Has at least {self.min_groups} groups of {self.group_size} lines"
        if self.allowed_tail_sizes:
            tails = ", ".join(str(size) for size in self.allowed_tail_sizes)
            desc += f", with an optional final tail of {tails} lines"
        return desc

    def instruction(self) -> str:
        desc = (
            f"Organize the poem into at least {self.min_groups} groups of {self.group_size} lines"
        )
        if self.allowed_tail_sizes:
            tails = " or ".join(str(size) for size in self.allowed_tail_sizes)
            desc += f", optionally ending with a final group of {tails} lines"
        return desc + "."


class SyllablesPerLine(Constraint):
    """
    Constraint on syllable count for each line.

    Examples:
        >>> constraint = SyllablesPerLine([5, 7, 5])  # Haiku
        >>> constraint = SyllablesPerLine([10] * 14)  # Sonnet (iambic pentameter)
    """

    name = "Syllables Per Line"
    constraint_type = ConstraintType.PROSODIC

    def __init__(
        self,
        expected_syllables: Sequence[int],
        weight: float = 1.0,
        sigma: float = 1.0,
        tolerance: int = 0,
    ) -> None:
        """
        Initialize syllable constraint.

        Args:
            expected_syllables: Expected syllable count for each line
            weight: Relative weight for composition
            sigma: Standard deviation for Gaussian decay
            tolerance: Allow +/- this many syllables and still pass
        """
        super().__init__(weight)
        self.expected_syllables = tuple(expected_syllables)
        if not self.expected_syllables or any(count <= 0 for count in self.expected_syllables):
            raise ValueError("expected_syllables must contain at least one positive syllable count")
        require_positive(sigma, "sigma")
        require_nonnegative(tolerance, "tolerance")
        self.sigma = sigma
        self.tolerance = tolerance

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        for i, expected in enumerate(self.expected_syllables):
            if i < len(structure.lines):
                line = structure.lines[i]
                actual = count_line_syllables(line)

                diff = abs(actual - expected)
                passed = diff <= self.tolerance

                if passed:
                    score = 1.0
                else:
                    # Gaussian decay from tolerance boundary
                    adjusted_diff = diff - self.tolerance
                    score = math.exp(-0.5 * (adjusted_diff / self.sigma) ** 2)
            else:
                actual = 0
                passed = False
                score = 0.0

            rubric.append(
                RubricItem(
                    criterion=f"Line {i + 1} syllables",
                    expected=str(expected) + (f" ± {self.tolerance}" if self.tolerance else ""),
                    actual=str(actual),
                    score=score,
                    passed=passed,
                    explanation=structure.lines[i][:50] + "..."
                    if i < len(structure.lines) and len(structure.lines[i]) > 50
                    else (structure.lines[i] if i < len(structure.lines) else ""),
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
        )

    def describe(self) -> str:
        if len(set(self.expected_syllables)) == 1:
            return f"Has {self.expected_syllables[0]} syllables per line"
        syllables_str = ", ".join(str(s) for s in self.expected_syllables[:5])
        if len(self.expected_syllables) > 5:
            syllables_str += ", ..."
        return f"Has syllable pattern [{syllables_str}]"

    def instruction(self) -> str:
        """Plain English instruction for LLM prompts."""
        if len(set(self.expected_syllables)) == 1:
            count = self.expected_syllables[0]
            tolerance_str = f" (plus or minus {self.tolerance})" if self.tolerance else ""
            return f"Each line should have approximately {count} syllables{tolerance_str}."
        syllables_str = "-".join(str(s) for s in self.expected_syllables)
        return f"Follow the syllable pattern {syllables_str} (syllables per line in order)."


class TotalSyllables(NumericConstraint):
    """
    Constraint on total syllable count in poem.
    """

    name = "Total Syllables"
    constraint_type = ConstraintType.PROSODIC

    def __init__(
        self,
        bound: int | NumericBound,
        weight: float = 1.0,
        sigma: float = 5.0,
    ) -> None:
        if isinstance(bound, int):
            bound = NumericBound.exact(bound)
        super().__init__(bound, weight, sigma)

    def _get_actual_value(self, structure: PoemStructure) -> int:
        return sum(count_line_syllables(line) for line in structure.lines)

    def _get_criterion_name(self) -> str:
        return "total syllables"

    def instruction(self) -> str:
        """Plain English instruction for LLM prompts."""
        return f"The entire poem should contain {self.bound.describe()} syllables total."
