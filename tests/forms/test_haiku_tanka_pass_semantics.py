"""Direct regressions for Haiku/Tanka canonical pass semantics."""

from __future__ import annotations

import pytest

from abide import verify
from abide.forms import Haiku, Tanka
from tests.fixtures.poems import HAIKU_SYNTHETIC_PERFECT, TANKA_SYNTHETIC_PERFECT


@pytest.mark.parametrize(
    ("factory", "poem"),
    [
        (lambda: Haiku(strict=False), HAIKU_SYNTHETIC_PERFECT),
        (lambda: Tanka(strict=False), TANKA_SYNTHETIC_PERFECT),
    ],
    ids=["haiku", "tanka"],
)
def test_japanese_short_forms_accept_valid_lenient_examples(
    factory: callable,
    poem: str,
) -> None:
    result = verify(poem, factory())

    assert result.passed is True
    assert result.details["canonical_requirements_passed"] is True


@pytest.mark.parametrize(
    ("factory", "poem"),
    [
        (
            lambda: Haiku(strict=False),
            HAIKU_SYNTHETIC_PERFECT + "\nQuiet echoes gather",
        ),
        (
            lambda: Tanka(strict=False),
            TANKA_SYNTHETIC_PERFECT + "\nCandles dim against the shore",
        ),
    ],
    ids=["haiku-extra-line", "tanka-extra-line"],
)
def test_japanese_short_forms_reject_extra_trailing_lines_in_lenient_mode(
    factory: callable,
    poem: str,
) -> None:
    result = verify(poem, factory())

    assert result.score > 0.9
    assert result.passed is False
    assert result.details["canonical_requirements_passed"] is False


@pytest.mark.parametrize(
    ("factory", "poem"),
    [
        (
            lambda: Haiku(strict=False),
            "The morning sun glows\n\nCherry blossoms gently fall\nSpring has come at last",
        ),
        (
            lambda: Tanka(strict=False),
            "The autumn moon shines\nCasting silver on the lake\n\nGentle ripples spread\nWhile the night birds call softly\nDreams drift on the quiet waves",
        ),
    ],
    ids=["haiku-two-stanzas", "tanka-two-stanzas"],
)
def test_japanese_short_forms_reject_multiple_stanzas_in_lenient_mode(
    factory: callable,
    poem: str,
) -> None:
    result = verify(poem, factory())

    assert result.score > 0.9
    assert result.passed is False
    assert result.details["canonical_requirements_passed"] is False
