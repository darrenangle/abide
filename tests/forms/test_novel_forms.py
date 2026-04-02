import pytest

from abide.forms.novel import (
    AlphabeticTerminus,
    BinaryBeat,
    ColorSpectrum,
    ConsonantCascade,
    DescendingStaircase,
    EchoEnd,
    ElementalVerse,
    ExclamationEcho,
    GoldenRatio,
    HourglassVerse,
    MonotoneMountain,
    NumberWord,
    NumericalEcho,
    OddEvenDance,
    PrimeVerse,
    QuestionQuest,
    SandwichSonnet,
    TemporalVerse,
    ThunderVerse,
    UniqueUtterance,
    VowelPilgrimage,
    WhisperPoem,
)


def test_novel_forms_accept_valid_examples() -> None:
    color_poem = "\n".join(
        [
            "A red lantern glows",
            "An orange banner sways",
            "Soft yellow pollen falls",
            "The green branch bends",
            "A blue harbor darkens",
            "An indigo ribbon lifts",
            "A violet window hums",
        ]
    )
    assert ColorSpectrum().verify(color_poem).passed is True

    element_poem = "\n".join(
        [
            "A gold lantern sways",
            "The silicon seal dries",
            "An iron hinge creaks",
            "Warm carbon dust drifts",
            "A lead bell sinks",
        ]
    )
    assert ElementalVerse().verify(element_poem).passed is True

    vowel_poem = "\n".join(
        [
            '"Apple glows',
            "'Elm bends",
            "(Ibis turns",
            '"Olive falls',
            "'Umbra drifts",
        ]
    )
    assert VowelPilgrimage().verify(vowel_poem).passed is True

    sandwich_poem = "\n".join(
        [
            "Begin again",
            "Hold the frame",
            "Middle line one",
            "Middle line two",
            "Middle line three",
            "Middle line four",
            "Begin again",
            "Hold the frame",
        ]
    )
    assert SandwichSonnet().verify(sandwich_poem).passed is True


def test_color_spectrum_requires_whole_color_words() -> None:
    poem = "\n".join(
        [
            "A bred mare runs",
            "The doorange swings",
            "An unyellowed page waits",
            "A greener branch bends",
            "The bluebird startles dawn",
            "An indigold thread glows",
            "Ultraviolet lamps hum",
        ]
    )

    result = ColorSpectrum().verify(poem)

    assert result.passed is False
    assert result.score < 0.2


def test_elemental_verse_requires_whole_element_words() -> None:
    poem = "\n".join(
        [
            "A golden lantern sways",
            "The siliconed seal dries",
            "An irony blooms",
            "Warm carbonara cools",
            "A leaden bell sinks",
        ]
    )

    result = ElementalVerse().verify(poem)

    assert result.passed is False
    assert result.score < 0.2


@pytest.mark.parametrize(
    "factory",
    [
        QuestionQuest,
        WhisperPoem,
        ThunderVerse,
        OddEvenDance,
        ElementalVerse,
        TemporalVerse,
        ExclamationEcho,
    ],
)
def test_empty_poem_does_not_receive_high_score_in_line_match_novel_forms(factory) -> None:
    result = factory().verify("")

    assert result.passed is False
    assert result.score <= 0.2


@pytest.mark.parametrize(
    ("form", "poem", "max_score"),
    [
        (QuestionQuest(), "?", 0.2),
        (ElementalVerse(), "gold", 0.2),
        (TemporalVerse(), "today", 0.2),
        (ExclamationEcho(), "!", 0.2),
        (OddEvenDance(), "one two three", 0.2),
        (ThunderVerse(), "x" * 60, 0.2),
        (WhisperPoem(), "alpha", 0.2),
        (WhisperPoem(), "alpha\nbravo", 0.2),
        (WhisperPoem(), "!!!\n???\n...\n!!!\n???", 0.6),
    ],
)
def test_min_line_novel_forms_do_not_reward_underlength_valid_prefixes(
    form,
    poem: str,
    max_score: float,
) -> None:
    result = form.verify(poem)

    assert result.passed is False
    assert result.score < max_score


@pytest.mark.parametrize(
    ("form", "poem"),
    [
        (PrimeVerse(), "alpha beta"),
        (HourglassVerse(), "alpha"),
        (DescendingStaircase(), "one two three four five six seven"),
        (NumericalEcho(), "one two three"),
        (BinaryBeat(), "alpha beta"),
        (GoldenRatio(), "alpha"),
    ],
)
def test_exact_word_pattern_novel_forms_do_not_reward_short_prefixes(form, poem: str) -> None:
    result = form.verify(poem)

    assert result.passed is False
    assert result.score < 0.2


def test_sandwich_sonnet_short_input_fails_instead_of_throwing() -> None:
    result = SandwichSonnet().verify("alpha\nbeta\ngamma")

    assert result.passed is False
    assert result.score < 0.1


def test_unique_utterance_does_not_hide_an_undocumented_min_word_threshold() -> None:
    poem = "\n".join(
        [
            "alpha beta gamma",
            "delta epsilon zeta",
            "eta theta iota",
            "kappa lambda mu",
            "nu xi omicron",
            "pi rho sigma",
        ]
    )

    result = UniqueUtterance().verify(poem)

    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.parametrize("poem", ["alpha", "alpha\nbravo"])
def test_unique_utterance_does_not_reward_too_few_words_for_its_line_floor(poem: str) -> None:
    result = UniqueUtterance().verify(poem)

    assert result.passed is False
    assert result.score < 0.2


def test_monotone_mountain_does_not_reward_short_monosyllabic_prefixes() -> None:
    result = MonotoneMountain().verify("cat dog")

    assert result.passed is False
    assert result.score < 0.2


@pytest.mark.parametrize(
    ("form", "poem", "max_score"),
    [
        (EchoEnd(), "alpha", 0.2),
        (ConsonantCascade(), "alpha\nbravo", 0.2),
        (ConsonantCascade(), "!!!\n???\n...\n!!!\n???", 0.2),
        (SandwichSonnet(), "!!!\n???\n...\n!!!\n???", 0.2),
    ],
)
def test_comparison_novel_forms_do_not_reward_vacuous_or_punctuation_only_samples(
    form,
    poem: str,
    max_score: float,
) -> None:
    result = form.verify(poem)

    assert result.passed is False
    assert result.score < max_score


def test_prime_verse_rejects_unsupported_line_counts_instead_of_wrapping_pattern() -> None:
    with pytest.raises(ValueError, match="num_lines must be between 1 and 6"):
        PrimeVerse(num_lines=7)


def test_number_word_rejects_unsupported_line_counts_instead_of_clamping() -> None:
    with pytest.raises(ValueError, match="num_lines must be between 1 and 10"):
        NumberWord(num_lines=12)


def test_novel_target_sequence_forms_reject_degenerate_constructor_values() -> None:
    with pytest.raises(ValueError, match="letters must contain at least one alphabetic character"):
        AlphabeticTerminus(letters="")

    with pytest.raises(ValueError, match="start_words must be positive"):
        DescendingStaircase(start_words=0)


def test_echo_end_rejects_invalid_explicit_target_letters() -> None:
    with pytest.raises(ValueError, match="letter must be a single alphabetic character"):
        EchoEnd(letter="")
