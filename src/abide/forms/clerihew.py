"""
Clerihew form template.

A humorous 4-line biographical poem.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from abide.constraints import (
    Constraint,
    ConstraintType,
    RubricItem,
    VerificationResult,
)
from abide.primitives import rhyme_score

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Clerihew(Constraint):
    """
    Clerihew: 4-line humorous biographical poem.

    Structure:
    - 4 lines total
    - Rhyme scheme: AABB (two couplets)
    - Line 1 must be/contain a person's name
    - Irregular meter (intentionally uneven)
    - Humorous or whimsical content

    Famous example:
        "Sir Humphry Davy / Abominated gravy..."
        - Edmund Clerihew Bentley

    Examples:
        >>> clerihew = Clerihew()
        >>> result = clerihew.verify(poem)
    """

    name = "Clerihew"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.5,
        strict: bool = False,
    ) -> None:
        """
        Initialize clerihew constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_threshold = rhyme_threshold
        self.strict = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        rubric: list[RubricItem] = []
        violations = 0

        # Check line count (exactly 4)
        if structure.line_count == 4:
            rubric.append(
                RubricItem(
                    criterion="Line count",
                    expected="4",
                    actual=str(structure.line_count),
                    score=1.0,
                    passed=True,
                )
            )
        else:
            rubric.append(
                RubricItem(
                    criterion="Line count",
                    expected="4",
                    actual=str(structure.line_count),
                    score=0.0,
                    passed=False,
                )
            )
            violations += 1

        if structure.line_count < 4:
            # Can't verify further constraints without enough lines
            return VerificationResult(
                score=0.05,
                passed=False,
                rubric=rubric,
                constraint_name=self.name,
                constraint_type=self.constraint_type,
            )

        # Check first line contains a name (capitalized word)
        first_line = structure.lines[0]
        has_name = self._contains_name(first_line)

        if has_name:
            rubric.append(
                RubricItem(
                    criterion="First line contains name",
                    expected="capitalized proper noun",
                    actual=first_line,
                    score=1.0,
                    passed=True,
                )
            )
        else:
            rubric.append(
                RubricItem(
                    criterion="First line contains name",
                    expected="capitalized proper noun",
                    actual=first_line,
                    score=0.0,
                    passed=False,
                )
            )
            violations += 1

        # Check AABB rhyme scheme
        end_words = [self._get_end_word(line) for line in structure.lines[:4]]

        # First couplet (lines 1-2)
        if len(end_words) >= 2:
            rhyme_12 = rhyme_score(end_words[0], end_words[1])
            passed_12 = rhyme_12 >= self.rhyme_threshold

            rubric.append(
                RubricItem(
                    criterion="Lines 1-2 rhyme (A-A)",
                    expected=f"rhyme >= {self.rhyme_threshold}",
                    actual=f"'{end_words[0]}' / '{end_words[1]}' = {rhyme_12:.2f}",
                    score=rhyme_12,
                    passed=passed_12,
                )
            )
            if not passed_12:
                violations += 1

        # Second couplet (lines 3-4)
        if len(end_words) >= 4:
            rhyme_34 = rhyme_score(end_words[2], end_words[3])
            passed_34 = rhyme_34 >= self.rhyme_threshold

            rubric.append(
                RubricItem(
                    criterion="Lines 3-4 rhyme (B-B)",
                    expected=f"rhyme >= {self.rhyme_threshold}",
                    actual=f"'{end_words[2]}' / '{end_words[3]}' = {rhyme_34:.2f}",
                    score=rhyme_34,
                    passed=passed_34,
                )
            )
            if not passed_34:
                violations += 1

        # Steep penalty scoring: 0 violations = 1.0, 1 = 0.5, 2 = 0.25, 3+ = 0.05
        if violations == 0:
            overall_score = 1.0
        elif violations == 1:
            overall_score = 0.5
        elif violations == 2:
            overall_score = 0.25
        else:  # 3 or more violations
            overall_score = 0.05

        overall_passed = violations == 0 if self.strict else overall_score >= 0.5

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
        )

    def _contains_name(self, line: str) -> bool:
        """Check if line contains a proper name (capitalized word)."""
        # Look for capitalized words that aren't at start of line
        words = line.split()
        if not words:
            return False

        # First word being capitalized is expected, look for others
        # or check if first word looks like a name/title
        capitalized = [w for w in words if w and w[0].isupper()]

        # Common name patterns: "Sir X", "Dr. X", "Name Name"
        if len(capitalized) >= 2:
            return True

        # Check for title patterns
        titles = ["sir", "dr", "mr", "mrs", "ms", "lord", "lady", "king", "queen"]
        first_word = words[0].lower().rstrip(".")
        if first_word in titles and len(words) >= 2:
            return True

        # Single capitalized word that looks like a name
        if len(words) >= 1 and words[0][0].isupper():
            # Check if it's not just a common word
            common_starters = [
                "the",
                "a",
                "an",
                "i",
                "we",
                "they",
                "he",
                "she",
                "it",
            ]
            if words[0].lower() not in common_starters:
                return True

        return False

    def _get_end_word(self, line: str) -> str:
        """Get last word of line."""
        words = re.findall(r"\b\w+\b", line)
        return words[-1].lower() if words else ""

    def describe(self) -> str:
        return "Clerihew: 4 lines with AABB rhyme, first line contains a name"
