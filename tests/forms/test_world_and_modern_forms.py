import pytest

from abide.forms.modern import Skeltonic
from abide.forms.world import Naani, Rispetto


def test_naani_accepts_valid_four_line_example() -> None:
    poem = "\n".join(
        [
            "sun sun sun sun sun",
            "sun sun sun sun sun",
            "sun sun sun sun sun",
            "sun sun sun sun sun",
        ]
    )

    result = Naani().verify(poem)

    assert result.passed is True
    assert 20 <= result.details["total_syllables"] <= 25


def test_naani_retain_partial_credit_but_fail_wrong_line_count() -> None:
    poem = "\n".join(
        [
            "sun sun sun sun sun sun sun",
            "sun sun sun sun sun sun sun",
            "sun sun sun sun sun sun sun",
        ]
    )

    result = Naani().verify(poem)

    assert result.passed is False
    assert 0.4 < result.score < 0.5


def test_skeltonic_accepts_valid_min_line_example() -> None:
    poem = "\n".join(
        [
            "small cat",
            "hard hat",
            "old mat",
            "soft bell",
            "grim tell",
            "thin shell",
            "pale moon",
            "swift tune",
            "dark noon",
            "clear light",
        ]
    )

    result = Skeltonic().verify(poem)

    assert result.passed is True
    assert result.details["line_count"] == 10


def test_skeltonic_rejects_high_scoring_wrong_line_count() -> None:
    poem = "\n".join(
        [
            "cat bat",
            "hat bat",
            "mat bat",
            "bell fell",
            "tell fell",
            "shell fell",
            "soon moon",
            "tune moon",
            "noon moon",
        ]
    )

    result = Skeltonic().verify(poem)

    assert result.passed is False
    assert result.score > 0.5


def test_skeltonic_rejects_identical_end_word_repetition() -> None:
    poem = "\n".join(
        [
            "small wind rain",
            "brief dust rain",
            "thin ash rain",
            "bare dusk rain",
            "hard frost rain",
            "small branch rain",
            "cold pond rain",
            "thin cloud rain",
            "brief field rain",
            "dark road rain",
        ]
    )

    result = Skeltonic().verify(poem)

    assert result.passed is False
    assert result.score >= 0.55


def test_rispetto_rejects_unsupported_variants_instead_of_falling_back() -> None:
    with pytest.raises(ValueError, match="variant must be one of: sicilian, tuscan"):
        Rispetto(variant="bogus")
