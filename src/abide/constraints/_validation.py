"""Shared constructor validation helpers for public constraints."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from abide.constraints.types import BoundType, NumericBound

if TYPE_CHECKING:
    from collections.abc import Sequence


def require_positive(value: int | float, name: str) -> None:
    """Reject nonpositive numeric parameters early."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def require_nonnegative(value: int | float, name: str) -> None:
    """Reject negative numeric parameters early."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def require_probability(value: float, name: str) -> None:
    """Reject probability-like parameters outside [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def require_positive_numeric_bound(bound: NumericBound, name: str) -> None:
    """Reject numeric bounds that would allow empty poems."""
    if bound.bound_type == BoundType.RANGE:
        if bound.min_value is None or bound.max_value is None:
            raise ValueError(f"{name} bound must be positive")
        if bound.min_value <= 0 or bound.max_value <= 0:
            raise ValueError(f"{name} bound must be positive")
        return

    if bound.value is None or bound.value <= 0:
        raise ValueError(f"{name} bound must be positive")


def require_line_indices(indices: Sequence[int], name: str) -> tuple[int, ...]:
    """Reject empty or negative line-index collections."""
    values = tuple(indices)
    if not values:
        raise ValueError(f"{name} must contain at least one line index")
    if any(index < 0 for index in values):
        raise ValueError(f"{name} indices must be non-negative")
    return values


def require_line_pairs(line_pairs: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Reject empty or malformed line-pair collections."""
    if not line_pairs:
        raise ValueError("line_pairs must contain at least one pair")

    validated: list[tuple[int, int]] = []
    for pair in line_pairs:
        try:
            line_a, line_b = pair
        except (TypeError, ValueError):
            raise ValueError("line_pairs must be pairs of two line indices") from None
        if line_a < 0 or line_b < 0:
            raise ValueError("line pair indices must be non-negative")
        validated.append((line_a, line_b))
    return tuple(validated)


def require_rotation(rotation: Sequence[int], num_words: int) -> tuple[int, ...]:
    """Reject malformed end-word rotations."""
    values = tuple(rotation)
    if len(values) != num_words:
        raise ValueError("rotation must contain exactly num_words indices")
    if set(values) != set(range(num_words)):
        raise ValueError("rotation must be a permutation of 0..num_words-1")
    return values


def require_alphabetic_text(value: str, name: str) -> None:
    """Reject strings that do not contain any alphabetic characters."""
    if not any(char.isalpha() for char in value):
        raise ValueError(f"{name} must contain at least one alphabetic character")


def require_word_list(words: Sequence[str], name: str) -> tuple[str, ...]:
    """Reject empty required-word lists and blank word entries."""
    values = tuple(words)
    if not values:
        raise ValueError(f"{name} must contain at least one word")
    if any(not re.search(r"\w", word) for word in values):
        raise ValueError(f"{name} must contain non-empty words")
    return values
