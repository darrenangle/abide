"""
Abide: Composable constraint algebra for specifying and auto-verifying poetic forms.

Abide provides a framework for defining poetic form constraints and verifying
whether text adheres to those constraints. It produces spectrum rewards suitable
for reinforcement learning and includes auto-generated rubrics explaining
verification results.

Example:
    >>> from abide.forms import Haiku
    >>> from abide import verify
    >>>
    >>> poem = '''
    ... An old silent pond
    ... A frog jumps into the pond
    ... Splash! Silence again
    ... '''
    >>> result = verify(poem, Haiku())
    >>> print(f"Score: {result.score:.2f}")
    Score: 0.95
"""

from abide.constraints import (
    Constraint,
    ConstraintType,
    VerificationResult,
)
from abide.specs import FormSpec
from abide.version import __version__


def verify(poem: str, constraint: Constraint) -> VerificationResult:
    """
    Verify a poem against a constraint.

    Args:
        poem: The poem text to verify
        constraint: The constraint to check against

    Returns:
        VerificationResult with score, pass/fail, and rubric
    """
    return constraint.verify(poem)


__all__ = [
    "Constraint",
    "ConstraintType",
    "FormSpec",
    "VerificationResult",
    "__version__",
    "verify",
]
