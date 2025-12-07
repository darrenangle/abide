"""
Abstract base constraint class.

All constraints inherit from Constraint and implement:
- verify(): Check poem and return VerificationResult
- describe(): Human-readable description
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from abide.constraints.types import (
    BoundType,
    ConstraintType,
    NumericBound,
    RubricItem,
    VerificationResult,
)
from abide.primitives import PoemStructure, parse_structure

if TYPE_CHECKING:
    pass


class Constraint(ABC):
    """
    Abstract base class for all constraints.

    Constraints verify that a poem adheres to specific rules and
    produce spectrum rewards suitable for RL training.
    """

    # Class-level attributes
    constraint_type: ConstraintType = ConstraintType.STRUCTURAL
    name: str = "Constraint"

    def __init__(self, weight: float = 1.0) -> None:
        """
        Initialize constraint.

        Args:
            weight: Relative weight for composition (default 1.0)
        """
        self.weight = weight

    @abstractmethod
    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        """
        Verify poem against this constraint.

        Args:
            poem: Raw poem text or parsed structure

        Returns:
            VerificationResult with score, pass/fail, and rubric
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """
        Return human-readable description of what this constraint checks.

        Should be phrased as a requirement, e.g., "Has exactly 14 lines"
        """
        ...

    def _ensure_structure(self, poem: str | PoemStructure) -> PoemStructure:
        """Convert string to PoemStructure if needed."""
        if isinstance(poem, str):
            return parse_structure(poem)
        return poem

    def to_dict(self) -> dict[str, Any]:
        """Serialize constraint to dictionary."""
        return {
            "type": self.__class__.__name__,
            "constraint_type": self.constraint_type.name,
            "name": self.name,
            "weight": self.weight,
            "description": self.describe(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight})"


class NumericConstraint(Constraint):
    """
    Base class for constraints checking numeric properties.

    Provides standard scoring with configurable decay for near-misses.
    """

    def __init__(
        self,
        bound: NumericBound,
        weight: float = 1.0,
        sigma: float = 2.0,
    ) -> None:
        """
        Initialize numeric constraint.

        Args:
            bound: The numeric bound to check
            weight: Relative weight for composition
            sigma: Standard deviation for Gaussian decay scoring
        """
        super().__init__(weight)
        self.bound = bound
        self.sigma = sigma

    @abstractmethod
    def _get_actual_value(self, structure: PoemStructure) -> int:
        """Extract the numeric value to check from the poem."""
        ...

    @abstractmethod
    def _get_criterion_name(self) -> str:
        """Return the name of the criterion being checked."""
        ...

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        """Verify the numeric constraint."""
        structure = self._ensure_structure(poem)
        actual = self._get_actual_value(structure)

        passed = self.bound.check(actual)
        score = self._compute_score(actual)

        rubric_item = RubricItem(
            criterion=self._get_criterion_name(),
            expected=self.bound.describe(),
            actual=str(actual),
            score=score,
            passed=passed,
        )

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[rubric_item],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"actual": actual, "bound": self.bound.describe()},
        )

    def _compute_score(self, actual: int) -> float:
        """
        Compute score using Gaussian decay for near-misses.

        Perfect match = 1.0, decay based on distance from expected.
        """
        if self.bound.bound_type == BoundType.EXACT:
            if actual == self.bound.value:
                return 1.0
            diff = abs(actual - self.bound.value)  # type: ignore
            return math.exp(-0.5 * (diff / self.sigma) ** 2)

        elif self.bound.bound_type == BoundType.MIN:
            if actual >= self.bound.value:  # type: ignore
                return 1.0
            diff = self.bound.value - actual  # type: ignore
            return math.exp(-0.5 * (diff / self.sigma) ** 2)

        elif self.bound.bound_type == BoundType.MAX:
            if actual <= self.bound.value:  # type: ignore
                return 1.0
            diff = actual - self.bound.value  # type: ignore
            return math.exp(-0.5 * (diff / self.sigma) ** 2)

        elif self.bound.bound_type == BoundType.RANGE:
            if self.bound.min_value <= actual <= self.bound.max_value:  # type: ignore
                return 1.0
            if actual < self.bound.min_value:  # type: ignore
                diff = self.bound.min_value - actual  # type: ignore
            else:
                diff = actual - self.bound.max_value  # type: ignore
            return math.exp(-0.5 * (diff / self.sigma) ** 2)

        return 0.0

    def describe(self) -> str:
        return f"Has {self.bound.describe()} {self._get_criterion_name().lower()}"
