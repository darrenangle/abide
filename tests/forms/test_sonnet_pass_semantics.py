from abide.forms.sonnet import Sonnet


def test_generic_sonnet_accepts_valid_ten_syllable_lines() -> None:
    line = "The silver river carries us at dawn"
    poem = "\n".join([line] * 14)

    result = Sonnet(strict=False).verify(poem)

    assert result.passed is True


def test_generic_sonnet_rejects_wrong_meter_false_positive() -> None:
    line = "silver rivers drift away"
    poem = "\n".join([line] * 14)

    result = Sonnet(strict=False).verify(poem)

    assert result.score >= 0.6
    assert result.passed is False
