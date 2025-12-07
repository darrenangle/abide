"""
Composition operators for combining constraints.

Logical operators (AND, OR, NOT) and aggregation (WeightedSum, AtLeast).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints.base import Constraint
from abide.constraints.types import (
    ConstraintType,
    RubricItem,
    VerificationResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from abide.primitives import PoemStructure


class And(Constraint):
    """
    Logical AND: all constraints must pass.

    Score is minimum of child scores (strict) or product (probabilistic).
    """

    name = "AND"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        constraints: Sequence[Constraint],
        mode: str = "min",  # "min", "product", "mean"
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.constraints = list(constraints)
        self.mode = mode

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        results = [c.verify(poem) for c in self.constraints]

        scores = [r.score for r in results]

        if self.mode == "min":
            overall_score = min(scores) if scores else 0.0
        elif self.mode == "product":
            overall_score = 1.0
            for s in scores:
                overall_score *= s
        else:  # mean
            overall_score = sum(scores) / len(scores) if scores else 0.0

        overall_passed = all(r.passed for r in results)

        # Combine rubrics
        rubric: list[RubricItem] = []
        for result in results:
            rubric.extend(result.rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"child_scores": scores, "mode": self.mode},
        )

    def describe(self) -> str:
        descs = [c.describe() for c in self.constraints]
        return " AND ".join(descs)


class Or(Constraint):
    """
    Logical OR: at least one constraint must pass.

    Score is maximum of child scores.
    """

    name = "OR"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        constraints: Sequence[Constraint],
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.constraints = list(constraints)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        results = [c.verify(poem) for c in self.constraints]

        scores = [r.score for r in results]
        overall_score = max(scores) if scores else 0.0
        overall_passed = any(r.passed for r in results)

        # Only include rubric from best-scoring constraint
        best_idx = scores.index(max(scores)) if scores else 0
        rubric = results[best_idx].rubric if results else []

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"child_scores": scores},
        )

    def describe(self) -> str:
        descs = [c.describe() for c in self.constraints]
        return " OR ".join(descs)


class Not(Constraint):
    """
    Logical NOT: inverts a constraint.

    Passes when child fails, fails when child passes.
    Score = 1 - child_score.
    """

    name = "NOT"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        constraint: Constraint,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.constraint = constraint

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self.constraint.verify(poem)

        overall_score = 1.0 - result.score
        overall_passed = not result.passed

        # Modify rubric to indicate negation
        rubric = [
            RubricItem(
                criterion=f"NOT: {r.criterion}",
                expected=f"NOT {r.expected}",
                actual=r.actual,
                score=1.0 - r.score,
                passed=not r.passed,
                explanation=r.explanation,
            )
            for r in result.rubric
        ]

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
        )

    def describe(self) -> str:
        return f"NOT ({self.constraint.describe()})"


class WeightedSum(Constraint):
    """
    Weighted combination of constraints.

    Score = sum(weight_i * score_i) / sum(weight_i)

    This is the primary composition method for form templates.
    """

    name = "Weighted Sum"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        constraints: Sequence[tuple[Constraint, float]] | Sequence[Constraint],
        threshold: float = 0.5,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize weighted sum.

        Args:
            constraints: Either list of (constraint, weight) tuples,
                        or list of constraints (uses constraint.weight)
            threshold: Score threshold for overall pass
            weight: Relative weight for further composition
        """
        super().__init__(weight)
        self.threshold = threshold

        # Handle both input formats
        self.weighted_constraints: list[tuple[Constraint, float]] = []
        for item in constraints:
            if isinstance(item, tuple):
                self.weighted_constraints.append(item)
            else:
                self.weighted_constraints.append((item, item.weight))

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        results: list[tuple[VerificationResult, float]] = []

        for constraint, weight in self.weighted_constraints:
            result = constraint.verify(poem)
            results.append((result, weight))

        # Compute weighted sum
        total_weight = sum(w for _, w in results)
        if total_weight == 0:
            overall_score = 0.0
        else:
            weighted_sum = sum(r.score * w for r, w in results)
            overall_score = weighted_sum / total_weight

        overall_passed = overall_score >= self.threshold

        # Combine rubrics with weight info
        rubric: list[RubricItem] = []
        for result, weight in results:
            for item in result.rubric:
                rubric.append(
                    RubricItem(
                        criterion=f"[w={weight:.1f}] {item.criterion}",
                        expected=item.expected,
                        actual=item.actual,
                        score=item.score,
                        passed=item.passed,
                        explanation=item.explanation,
                    )
                )

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "child_scores": [(r.score, w) for r, w in results],
                "threshold": self.threshold,
            },
        )

    def describe(self) -> str:
        parts = [f"{c.describe()} (w={w:.1f})" for c, w in self.weighted_constraints]
        return "Weighted: " + " + ".join(parts)


class AtLeast(Constraint):
    """
    At least N constraints must pass.

    Useful for flexible form definitions.
    """

    name = "At Least"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        n: int,
        constraints: Sequence[Constraint],
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.n = n
        self.constraints = list(constraints)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        results = [c.verify(poem) for c in self.constraints]

        passed_count = sum(1 for r in results if r.passed)
        overall_passed = passed_count >= self.n

        # Score based on how many passed relative to required
        overall_score = 1.0 if self.n == 0 else min(1.0, passed_count / self.n)

        # Combine rubrics
        rubric: list[RubricItem] = []
        for result in results:
            rubric.extend(result.rubric)

        rubric.insert(
            0,
            RubricItem(
                criterion="Constraints passed",
                expected=f"at least {self.n}",
                actual=str(passed_count),
                score=overall_score,
                passed=overall_passed,
            ),
        )

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"n": self.n, "passed_count": passed_count},
        )

    def describe(self) -> str:
        return f"At least {self.n} of: " + ", ".join(c.describe() for c in self.constraints)


class AtMost(Constraint):
    """
    At most N constraints can pass.

    Useful for exclusion rules.
    """

    name = "At Most"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        n: int,
        constraints: Sequence[Constraint],
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.n = n
        self.constraints = list(constraints)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        results = [c.verify(poem) for c in self.constraints]

        passed_count = sum(1 for r in results if r.passed)
        overall_passed = passed_count <= self.n

        # Score inversely related to excess
        if passed_count <= self.n:
            overall_score = 1.0
        else:
            excess = passed_count - self.n
            overall_score = max(0.0, 1.0 - excess / len(self.constraints))

        # Combine rubrics
        rubric: list[RubricItem] = []
        for result in results:
            rubric.extend(result.rubric)

        rubric.insert(
            0,
            RubricItem(
                criterion="Constraints passed",
                expected=f"at most {self.n}",
                actual=str(passed_count),
                score=overall_score,
                passed=overall_passed,
            ),
        )

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"n": self.n, "passed_count": passed_count},
        )

    def describe(self) -> str:
        return f"At most {self.n} of: " + ", ".join(c.describe() for c in self.constraints)
