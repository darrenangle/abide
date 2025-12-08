"""
Verifiers framework compatibility layer.

This module provides integration with the verifiers framework
(https://github.com/PrimeIntellect-ai/verifiers) for using abide
constraints as reward functions in RL training.

The verifiers framework expects:
- A dataset with 'prompt' and optionally 'answer'/'info' columns
- Reward functions with signature: (prompt, completion, answer, state) -> float
- A Rubric combining multiple reward functions

Example:
    >>> import verifiers as vf
    >>> from abide.evals.verifiers_compat import make_abide_rubric
    >>> from abide.forms import Haiku
    >>>
    >>> rubric = make_abide_rubric([Haiku()])
    >>> env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from abide.constraints import Constraint


def make_reward_func(
    constraint: Constraint,
    extract_poem: Callable[[list[dict[str, str]]], str] | None = None,
) -> Callable[[str, list[dict[str, str]], str, dict[str, Any]], float]:
    """
    Create a verifiers-compatible reward function from an abide constraint.

    The verifiers framework expects reward functions with this signature:
        (prompt, completion, answer, state) -> float

    This wrapper extracts the poem text from the completion and runs
    verification against the constraint.

    Args:
        constraint: Abide constraint to use for verification
        extract_poem: Optional function to extract poem from completion.
                     Default extracts content from last assistant message.

    Returns:
        Reward function compatible with verifiers.Rubric
    """
    if extract_poem is None:

        def extract_poem(completion: list[dict[str, str]]) -> str:
            """Extract poem text from completion (last assistant message)."""
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            return ""

    def reward_func(
        prompt: str | list[dict[str, str]],
        completion: list[dict[str, str]],
        answer: str,
        state: dict[str, Any],
    ) -> float:
        """Verify poem against constraint and return score."""
        poem = extract_poem(completion)
        if not poem:
            return 0.0

        result = constraint.verify(poem)
        return result.score

    # Set function name for debugging
    reward_func.__name__ = f"abide_{constraint.name.lower().replace(' ', '_')}"
    reward_func.__doc__ = f"Verify poem against {constraint.name}: {constraint.describe()}"

    return reward_func


def make_pass_func(
    constraint: Constraint,
    extract_poem: Callable[[list[dict[str, str]]], str] | None = None,
) -> Callable[[str, list[dict[str, str]], str, dict[str, Any]], float]:
    """
    Create a binary (pass/fail) reward function.

    Returns 1.0 if constraint passes, 0.0 otherwise.
    Useful for strict evaluation scenarios.
    """
    if extract_poem is None:

        def extract_poem(completion: list[dict[str, str]]) -> str:
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            return ""

    def pass_func(
        prompt: str | list[dict[str, str]],
        completion: list[dict[str, str]],
        answer: str,
        state: dict[str, Any],
    ) -> float:
        """Return 1.0 if constraint passes, 0.0 otherwise."""
        poem = extract_poem(completion)
        if not poem:
            return 0.0

        result = constraint.verify(poem)
        return 1.0 if result.passed else 0.0

    pass_func.__name__ = f"abide_{constraint.name.lower().replace(' ', '_')}_pass"
    return pass_func


def make_constraint_funcs(
    constraints: list[Constraint],
    include_pass: bool = False,
) -> list[Callable[..., float]]:
    """
    Create reward functions for multiple constraints.

    Args:
        constraints: List of abide constraints
        include_pass: If True, include binary pass/fail functions too

    Returns:
        List of reward functions
    """
    funcs = []
    for constraint in constraints:
        funcs.append(make_reward_func(constraint))
        if include_pass:
            funcs.append(make_pass_func(constraint))
    return funcs


# Verifiers Rubric integration (requires verifiers package)
def make_abide_rubric(
    constraints: list[Constraint],
    weights: list[float] | None = None,
) -> Any:
    """
    Create a verifiers Rubric from abide constraints.

    This requires the verifiers package to be installed.

    Args:
        constraints: List of abide constraints
        weights: Optional weights for each constraint (default: equal weights)

    Returns:
        verifiers.Rubric instance

    Raises:
        ImportError: If verifiers package is not installed
    """
    try:
        import verifiers as vf
    except ImportError as e:
        raise ImportError(
            "verifiers package required for Rubric integration. " "Install with: uv add verifiers"
        ) from e

    funcs = [make_reward_func(c) for c in constraints]
    weights = weights or [1.0] * len(constraints)

    return vf.Rubric(funcs=funcs, weights=weights)


def make_forms_mini_rubric(
    strict: bool = False,
) -> Any:
    """
    Create a rubric for the forms-mini evaluation.

    Uses the standard forms from MAJOR_FORMS_MINI.

    Args:
        strict: If True, use binary pass/fail scoring

    Returns:
        verifiers.Rubric instance
    """
    from abide.evals.forms_mini import MAJOR_FORMS_MINI

    constraints = [spec.constraint for spec in MAJOR_FORMS_MINI]

    if strict:
        funcs = [make_pass_func(c) for c in constraints]
    else:
        funcs = [make_reward_func(c) for c in constraints]

    weights = [1.0] * len(constraints)

    try:
        import verifiers as vf

        return vf.Rubric(funcs=funcs, weights=weights)
    except ImportError:
        # Return dict of functions if verifiers not installed
        return {
            "funcs": funcs,
            "weights": weights,
        }
