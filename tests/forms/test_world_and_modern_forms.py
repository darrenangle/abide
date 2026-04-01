from abide.forms.modern import Skeltonic
from abide.forms.world import Naani


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


def test_naani_rejects_high_scoring_wrong_line_count() -> None:
    poem = "\n".join(
        [
            "sun sun sun sun sun sun sun",
            "sun sun sun sun sun sun sun",
            "sun sun sun sun sun sun sun",
        ]
    )

    result = Naani().verify(poem)

    assert result.passed is False
    assert result.score > 0.9


def test_skeltonic_accepts_valid_min_line_example() -> None:
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
            "light bright",
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
    assert result.score > 0.9
