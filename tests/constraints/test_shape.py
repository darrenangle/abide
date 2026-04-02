"""Tests for shape constraints."""

import pytest

from abide.constraints import LineLengthRange, LineShape
from abide.constraints.shape import MeasureMode, ShapeType


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: LineShape(),
            "must specify lengths or shape_type with num_lines",
        ),
        (
            lambda: LineShape(shape_type=ShapeType.DIAMOND),
            "must specify lengths or shape_type with num_lines",
        ),
        (
            lambda: LineShape(shape_type=ShapeType.DIAMOND, num_lines=0),
            "num_lines must be positive",
        ),
        (
            lambda: LineShape(lengths=[]),
            "lengths must contain at least one positive length",
        ),
        (
            lambda: LineShape(lengths=[1, 0, 1]),
            "lengths must contain at least one positive length",
        ),
        (
            lambda: LineShape(lengths=[1, 2, 1], shape_type=ShapeType.DIAMOND, num_lines=3),
            "lengths cannot be combined with shape_type or num_lines",
        ),
        (
            lambda: LineLengthRange(),
            "must specify min_length or max_length",
        ),
        (
            lambda: LineLengthRange(min_length=0, max_length=1),
            "min_length must be positive",
        ),
        (
            lambda: LineLengthRange(min_length=4, max_length=3),
            "max_length must be greater than or equal to min_length",
        ),
    ],
)
def test_shape_constraints_reject_invalid_constructor_values(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_line_length_range_rejects_empty_poem() -> None:
    constraint = LineLengthRange(min_length=1, max_length=4, mode=MeasureMode.WORDS)
    result = constraint.verify("")

    assert result.passed is False
    assert result.score == 0.0
