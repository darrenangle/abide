"""
Clerihew form template.

Structural proxy for the 4-line clerihew shell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    EndRhymePairs,
    LineCount,
    RubricItem,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class _FirstLineContainsName(Constraint):
    """Constraint that the first line appears to contain a proper name."""

    name = "First Line Contains Name"
    constraint_type = ConstraintType.SEMANTIC

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        if not structure.lines:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[
                    RubricItem(
                        criterion="First line contains name",
                        expected="capitalized proper noun",
                        actual="(line missing)",
                        score=0.0,
                        passed=False,
                    )
                ],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
            )

        first_line = structure.lines[0]
        passed = Clerihew._contains_name(first_line)
        score = 1.0 if passed else 0.0
        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[
                RubricItem(
                    criterion="First line contains name",
                    expected="capitalized proper noun",
                    actual=first_line,
                    score=score,
                    passed=passed,
                )
            ],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
        )

    def describe(self) -> str:
        return "First line contains a proper name"


class Clerihew(Constraint):
    """
    Clerihew structural proxy.

    Structure:
    - 4 lines total
    - Rhyme scheme: AABB (two couplets)
    - Line 1 must be/contain a person's name
    - Irregular meter (intentionally uneven)

    This verifier checks line count, a first-line-name proxy, and rhyme only.

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
        self._line_count = LineCount(4, weight=2.0)
        self._name = _FirstLineContainsName(weight=2.0)
        self._rhyme = EndRhymePairs(
            [(0, 1), (2, 3)],
            threshold=rhyme_threshold,
            weight=2.0,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._name, 2.0),
            (self._rhyme, 2.0),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([constraint for constraint, _ in constraints])
        else:
            self._constraint = WeightedSum(
                constraints,
                threshold=0.5,
                required_indices=list(range(len(constraints))),
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

    @staticmethod
    def _contains_name(line: str) -> bool:
        """Check if line contains a plausible name-style phrase."""
        words = [word.strip(".,;:!?\"'()[]{}") for word in line.split()]
        words = [word for word in words if word]
        if not words:
            return False

        capitalized_positions = [idx for idx, word in enumerate(words) if word[0].isupper()]

        # Common title patterns: "Sir X", "Dr X"
        titles = ["sir", "dr", "mr", "mrs", "ms", "lord", "lady", "king", "queen"]
        first_word = words[0].lower().rstrip(".")
        if first_word in titles and len(words) >= 2 and words[1][0].isupper():
            return True

        # A name later in the line or a multi-token capitalized phrase is a
        # stronger proxy than bare sentence-start capitalization.
        if len(capitalized_positions) >= 2:
            return True
        return any(position > 0 for position in capitalized_positions)

    def describe(self) -> str:
        return "Clerihew: 4 lines with AABB rhyme, first line contains a name"
