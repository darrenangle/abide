"""Tests for meter and scansion primitives."""

from abide.primitives import MeterType, detect_meter, meter_score, scan_line


def test_scan_line_identifies_shakespeare_example_as_iambic_pentameter() -> None:
    line = "Shall I compare thee to a summer's day"
    result = scan_line(line)

    assert result.binary_pattern == "1101110101"
    assert result.syllable_count == 10
    assert result.dominant_meter == MeterType.IAMB
    assert result.foot_count == 5
    assert result.regularity >= 0.8


def test_scan_line_identifies_trochaic_line() -> None:
    line = "Tyger Tyger burning bright"
    result = scan_line(line)

    assert result.dominant_meter == MeterType.TROCHEE
    assert result.foot_count == 4
    assert result.regularity >= 0.9


def test_meter_score_prefers_matching_meter() -> None:
    iambic_line = "Shall I compare thee to a summer's day"
    trochaic_line = "Tyger Tyger burning bright"

    assert meter_score(iambic_line, MeterType.IAMB, expected_feet=5) > meter_score(
        iambic_line,
        MeterType.TROCHEE,
        expected_feet=5,
    )
    assert meter_score(trochaic_line, MeterType.TROCHEE, expected_feet=4) > meter_score(
        trochaic_line,
        MeterType.IAMB,
        expected_feet=4,
    )


def test_detect_meter_uses_line_level_alignment() -> None:
    meter, confidence = detect_meter("Shall I compare thee to a summer's day")

    assert meter == MeterType.IAMB
    assert confidence >= 0.8
