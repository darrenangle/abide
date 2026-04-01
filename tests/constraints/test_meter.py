"""Tests for meter constraints."""

from abide.constraints import Meter, MeterPattern
from abide.primitives import FootLength, MeterType


def test_meter_constraint_accepts_iambic_pentameter_line() -> None:
    constraint = Meter(MeterType.IAMB, FootLength.PENTAMETER, min_score=0.8)
    result = constraint.verify("Shall I compare thee to a summer's day")

    assert result.passed is True
    assert result.score >= 0.8
    assert "iamb" in result.rubric[0].actual.lower()


def test_meter_constraint_rejects_wrong_meter_line() -> None:
    constraint = Meter(MeterType.IAMB, FootLength.PENTAMETER, min_score=0.8)
    result = constraint.verify("Tyger Tyger burning bright")

    assert result.passed is False
    assert result.score < 0.8


def test_meter_pattern_accepts_common_meter_excerpt() -> None:
    poem = "\n".join(
        [
            "Amazing grace how sweet the sound",
            "That saved a wretch like me",
            "I once was lost but now am found",
            "Was blind but now I see",
        ]
    )
    constraint = MeterPattern(MeterType.IAMB, [4, 3, 4, 3], min_score=0.6)
    result = constraint.verify(poem)

    assert result.passed is True
    assert result.score >= 0.6
