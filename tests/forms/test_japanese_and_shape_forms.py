import pytest

from abide.forms import (
    Cinquain,
    Diamante,
    Etheree,
    Katauta,
    ReverseEtheree,
    Sedoka,
    Senryu,
    WordCinquain,
)


def _mono_line(count: int, word: str = "sun") -> str:
    return " ".join([word] * count)


SENRYU_PERFECT = "\n".join([_mono_line(5), _mono_line(7), _mono_line(5)])
SENRYU_NEAR_MISS = "\n".join([_mono_line(6), _mono_line(7), _mono_line(5)])

KATAUTA_PERFECT = "\n".join([_mono_line(5), _mono_line(7), _mono_line(7)])
KATAUTA_NEAR_MISS = "\n".join([_mono_line(5), _mono_line(8), _mono_line(7)])

SEDOKA_PERFECT = "\n".join(
    [
        _mono_line(5),
        _mono_line(7),
        _mono_line(7),
        _mono_line(5),
        _mono_line(7),
        _mono_line(7),
    ]
)
SEDOKA_NEAR_MISS = "\n".join(
    [
        _mono_line(5),
        _mono_line(7),
        _mono_line(7),
        _mono_line(6),
        _mono_line(7),
        _mono_line(7),
    ]
)

DIAMANTE_PERFECT = "\n".join(
    [
        "stone",
        "calm bright",
        "drift glow sing",
        "earth wind fire rain",
        "turn move hum",
        "soft warm",
        "ember",
    ]
)
DIAMANTE_NEAR_MISS = "\n".join(
    [
        "stone",
        "calm bright",
        "drift glow sing",
        "earth wind fire rain light",
        "turn move hum",
        "soft warm",
        "ember",
    ]
)

CINQUAIN_PERFECT = "\n".join(
    [_mono_line(2), _mono_line(4), _mono_line(6), _mono_line(8), _mono_line(2)]
)
CINQUAIN_NEAR_MISS = "\n".join(
    [_mono_line(2), _mono_line(4), _mono_line(7), _mono_line(8), _mono_line(2)]
)

WORD_CINQUAIN_PERFECT = "\n".join(
    [
        "stone",
        "calm bright",
        "drift glow sing",
        "earth wind fire rain",
        "ember",
    ]
)
WORD_CINQUAIN_NEAR_MISS = "\n".join(
    [
        "stone",
        "calm bright",
        "drift glow sing",
        "earth wind fire rain light",
        "ember",
    ]
)

ETHEREE_PERFECT = "\n".join([_mono_line(count) for count in range(1, 11)])
ETHEREE_NEAR_MISS = "\n".join(
    [_mono_line(1), _mono_line(2), _mono_line(3), _mono_line(4), _mono_line(6)]
    + [_mono_line(count) for count in range(6, 11)]
)

REVERSE_ETHEREE_PERFECT = "\n".join([_mono_line(count) for count in range(10, 0, -1)])
REVERSE_ETHEREE_NEAR_MISS = "\n".join(
    [_mono_line(count) for count in range(10, 5, -1)]
    + [_mono_line(4)]
    + [_mono_line(count) for count in range(4, 0, -1)]
)


@pytest.mark.parametrize(
    ("factory", "poem"),
    [
        (Senryu, SENRYU_PERFECT),
        (Katauta, KATAUTA_PERFECT),
        (Sedoka, SEDOKA_PERFECT),
        (Diamante, DIAMANTE_PERFECT),
        (Cinquain, CINQUAIN_PERFECT),
        (WordCinquain, WORD_CINQUAIN_PERFECT),
        (Etheree, ETHEREE_PERFECT),
        (ReverseEtheree, REVERSE_ETHEREE_PERFECT),
    ],
)
def test_short_japanese_and_shape_forms_accept_valid_examples(factory, poem: str) -> None:
    result = factory().verify(poem)

    assert result.passed is True
    assert result.score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("factory", "poem"),
    [
        (lambda: Senryu(strict=False), SENRYU_NEAR_MISS),
        (lambda: Katauta(strict=False), KATAUTA_NEAR_MISS),
        (lambda: Sedoka(strict=False), SEDOKA_NEAR_MISS),
        (Diamante, DIAMANTE_NEAR_MISS),
        (Cinquain, CINQUAIN_NEAR_MISS),
        (WordCinquain, WORD_CINQUAIN_NEAR_MISS),
        (Etheree, ETHEREE_NEAR_MISS),
        (ReverseEtheree, REVERSE_ETHEREE_NEAR_MISS),
    ],
)
def test_short_japanese_and_shape_forms_reject_high_scoring_near_misses(factory, poem: str) -> None:
    result = factory().verify(poem)

    assert result.score > 0.7
    assert result.passed is False


def test_senryu_description_avoids_unverified_semantic_claims() -> None:
    assert "human" not in Senryu().describe().lower()


def test_word_cinquain_does_not_treat_punctuation_as_words() -> None:
    poem = "!!!\n???\n...\n!!!\n???"

    result = WordCinquain().verify(poem)

    assert result.passed is False
    assert result.score < 0.6
