from abide.forms.novel import (
    ColorSpectrum,
    ElementalVerse,
    SandwichSonnet,
    UniqueUtterance,
    VowelPilgrimage,
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
