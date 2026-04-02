"""Shared constructor validation helpers for form configuration."""

from __future__ import annotations


def require_positive(value: int, name: str) -> None:
    """Reject nonpositive integer parameters early."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def require_nonnegative(value: int, name: str) -> None:
    """Reject negative integer parameters early."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def require_at_most(value: int, maximum: int, name: str) -> None:
    """Reject integer parameters above a supported maximum."""
    if value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def require_ordered_bounds(
    upper_name: str,
    upper_value: int | None,
    lower_name: str,
    lower_value: int,
) -> None:
    """Reject inverted lower/upper bound pairs."""
    if upper_value is None:
        return
    require_positive(upper_value, upper_name)
    if upper_value < lower_value:
        raise ValueError(f"{upper_name} must be greater than or equal to {lower_name}")
