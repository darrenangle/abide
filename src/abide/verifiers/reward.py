"""
Reward function interface for verifiers framework.

Provides PoeticFormReward which wraps any constraint as a
verifiers-compatible reward function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from abide.constraints import Constraint, VerificationResult


@dataclass
class RewardOutput:
    """
    Output format compatible with verifiers framework.

    Attributes:
        score: Reward score in [0, 1]
        passed: Whether the constraint was satisfied
        rubric: Human-readable explanation of the score
        metadata: Additional information about the verification
    """

    score: float
    passed: bool
    rubric: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "score": self.score,
            "passed": self.passed,
            "rubric": self.rubric,
            "metadata": self.metadata,
        }


class PoeticFormReward:
    """
    Verifiers-compatible reward function for poetic form verification.

    Wraps any abide Constraint as a callable reward function that
    can be used with the verifiers framework.

    Example:
        >>> from abide.forms import Haiku
        >>> reward_fn = PoeticFormReward(Haiku())
        >>> output = reward_fn("An old silent pond\\nA frog jumps in\\nSplash!")
        >>> print(f"Score: {output.score:.2f}")
    """

    def __init__(
        self,
        constraint: Constraint,
        name: str | None = None,
        normalize_score: bool = True,
        include_rubric: bool = True,
    ) -> None:
        """
        Initialize reward function.

        Args:
            constraint: The poetic form constraint to verify against
            name: Optional name for this reward function
            normalize_score: Whether to ensure score is in [0, 1]
            include_rubric: Whether to include detailed rubric in output
        """
        self.constraint = constraint
        self.name = name or constraint.name
        self.normalize_score = normalize_score
        self.include_rubric = include_rubric

    def __call__(self, text: str) -> RewardOutput:
        """
        Evaluate text against the constraint.

        Args:
            text: The text (poem) to evaluate

        Returns:
            RewardOutput with score, pass/fail, and rubric
        """
        result = self.constraint.verify(text)
        return self._format_output(result)

    def verify(self, text: str) -> VerificationResult:
        """
        Get full verification result.

        Args:
            text: The text (poem) to evaluate

        Returns:
            Full VerificationResult with all details
        """
        return self.constraint.verify(text)

    def _format_output(self, result: VerificationResult) -> RewardOutput:
        """Format VerificationResult as RewardOutput."""
        score = result.score
        if self.normalize_score:
            score = max(0.0, min(1.0, score))

        if self.include_rubric:
            rubric_lines = [str(item) for item in result.rubric]
            rubric = "\n".join(rubric_lines)
        else:
            rubric = ""

        return RewardOutput(
            score=score,
            passed=result.passed,
            rubric=rubric,
            metadata={
                "constraint_name": result.constraint_name,
                "constraint_type": result.constraint_type.name,
                "details": result.details,
            },
        )

    def describe(self) -> str:
        """Get description of what this reward function checks."""
        return self.constraint.describe()


def make_reward_function(
    constraint: Constraint,
    **kwargs: Any,
) -> Callable[[str], RewardOutput]:
    """
    Factory function to create a reward function from a constraint.

    Args:
        constraint: The constraint to wrap
        **kwargs: Additional arguments for PoeticFormReward

    Returns:
        Callable reward function
    """
    return PoeticFormReward(constraint, **kwargs)
