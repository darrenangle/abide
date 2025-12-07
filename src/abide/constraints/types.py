"""
Core types for the constraint algebra.

Provides foundational types used by all constraints:
- ConstraintType: Categories of constraints
- BoundType: How numeric bounds work
- VerificationResult: What constraints return
- RubricItem: Detailed explanations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ConstraintType(Enum):
    """Categories of poetic constraints."""

    STRUCTURAL = auto()  # Line count, stanza count, etc.
    RELATIONAL = auto()  # Rhyme, refrain, end-word patterns
    PROSODIC = auto()  # Meter, syllables, stress
    SONIC = auto()  # Assonance, consonance, alliteration
    SEMANTIC = auto()  # Word choice, themes (future)
    COMPOSITE = auto()  # Combined constraints


class BoundType(Enum):
    """Types of numeric bounds."""

    EXACT = auto()  # Must equal exactly
    MIN = auto()  # Must be at least
    MAX = auto()  # Must be at most
    RANGE = auto()  # Must be within range


@dataclass(frozen=True)
class NumericBound:
    """
    Specification for a numeric constraint bound.

    Examples:
        >>> exact_14 = NumericBound(bound_type=BoundType.EXACT, value=14)
        >>> at_least_3 = NumericBound(bound_type=BoundType.MIN, value=3)
        >>> between_5_10 = NumericBound(bound_type=BoundType.RANGE, min_value=5, max_value=10)
    """

    bound_type: BoundType
    value: int | None = None  # For EXACT, MIN, MAX
    min_value: int | None = None  # For RANGE
    max_value: int | None = None  # For RANGE

    def __post_init__(self) -> None:
        """Validate bound configuration."""
        if self.bound_type == BoundType.RANGE:
            if self.min_value is None or self.max_value is None:
                raise ValueError("RANGE bounds require min_value and max_value")
            if self.min_value > self.max_value:
                raise ValueError("min_value must be <= max_value")
        elif self.bound_type in (BoundType.EXACT, BoundType.MIN, BoundType.MAX):
            if self.value is None:
                raise ValueError(f"{self.bound_type.name} bounds require value")

    def check(self, actual: int) -> bool:
        """Check if actual value satisfies the bound."""
        if self.bound_type == BoundType.EXACT:
            return actual == self.value
        elif self.bound_type == BoundType.MIN:
            return actual >= self.value  # type: ignore
        elif self.bound_type == BoundType.MAX:
            return actual <= self.value  # type: ignore
        elif self.bound_type == BoundType.RANGE:
            return self.min_value <= actual <= self.max_value  # type: ignore
        return False

    def describe(self) -> str:
        """Human-readable description of the bound."""
        if self.bound_type == BoundType.EXACT:
            return f"exactly {self.value}"
        elif self.bound_type == BoundType.MIN:
            return f"at least {self.value}"
        elif self.bound_type == BoundType.MAX:
            return f"at most {self.value}"
        elif self.bound_type == BoundType.RANGE:
            return f"between {self.min_value} and {self.max_value}"
        return "unknown bound"

    @classmethod
    def exact(cls, value: int) -> NumericBound:
        """Create an EXACT bound."""
        return cls(bound_type=BoundType.EXACT, value=value)

    @classmethod
    def at_least(cls, value: int) -> NumericBound:
        """Create a MIN bound."""
        return cls(bound_type=BoundType.MIN, value=value)

    @classmethod
    def at_most(cls, value: int) -> NumericBound:
        """Create a MAX bound."""
        return cls(bound_type=BoundType.MAX, value=value)

    @classmethod
    def between(cls, min_val: int, max_val: int) -> NumericBound:
        """Create a RANGE bound."""
        return cls(bound_type=BoundType.RANGE, min_value=min_val, max_value=max_val)


@dataclass
class RubricItem:
    """
    A single item in a verification rubric.

    Explains what was checked, what was expected, what was found,
    and whether it passed.
    """

    criterion: str  # What was checked (e.g., "Line count")
    expected: str  # What was expected (e.g., "exactly 14")
    actual: str  # What was found (e.g., "14")
    score: float  # Score for this item [0, 1]
    passed: bool  # Whether this item passed
    explanation: str = ""  # Additional context

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        result = f"[{status}] {self.criterion}: expected {self.expected}, got {self.actual}"
        if self.explanation:
            result += f" ({self.explanation})"
        return result


@dataclass
class VerificationResult:
    """
    Result of verifying a poem against a constraint.

    Contains the overall score, pass/fail status, and detailed rubric.
    """

    score: float  # Overall score [0, 1]
    passed: bool  # Whether the constraint is satisfied
    rubric: list[RubricItem] = field(default_factory=list)
    constraint_name: str = ""
    constraint_type: ConstraintType = ConstraintType.STRUCTURAL
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.constraint_name}: {self.score:.2f}"]
        for item in self.rubric:
            lines.append(f"  {item}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "score": self.score,
            "passed": self.passed,
            "constraint_name": self.constraint_name,
            "constraint_type": self.constraint_type.name,
            "rubric": [
                {
                    "criterion": r.criterion,
                    "expected": r.expected,
                    "actual": r.actual,
                    "score": r.score,
                    "passed": r.passed,
                    "explanation": r.explanation,
                }
                for r in self.rubric
            ],
            "details": self.details,
        }
