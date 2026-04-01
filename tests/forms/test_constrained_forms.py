from abide.forms import (
    Abecedarian,
    Anaphora,
    Lipogram,
    Mesostic,
    PalindromePoem,
    Univocalic,
)
from abide.forms.catalog import instantiate_form


def test_abecedarian_ignores_leading_punctuation() -> None:
    poem = "\"Alpha\n'Bravo\n(Charlie"

    result = Abecedarian(letters="ABC").verify(poem)

    assert result.passed is True
    assert result.score == 1.0


def test_lipogram_and_univocalic_enforce_letter_constraints() -> None:
    lipogram_pass = Lipogram(forbidden="E").verify("A calm dusk\nBright sun")
    lipogram_fail = Lipogram(forbidden="E").verify("See the tree")
    univocalic_pass = Univocalic(vowel="E").verify("deep green\nserene breeze")
    univocalic_fail = Univocalic(vowel="E").verify("green field\nclear breeze")

    assert lipogram_pass.passed is True
    assert lipogram_fail.passed is False
    assert univocalic_pass.passed is True
    assert univocalic_fail.passed is False


def test_mesostic_requires_target_letters_in_the_middle() -> None:
    good = Mesostic(word="POEM").verify("aPex\nclOud\ntrEe\naMber")
    bad = Mesostic(word="POEM").verify("atlas\nclOud\ntrEe\naMber")

    assert good.passed is True
    assert bad.passed is False


def test_anaphora_auto_detect_prefers_longest_repeated_opening() -> None:
    poem = (
        "I have a dream today\n"
        "I have a dream tonight\n"
        "I have a dream still\n"
        "The road is dark"
    )

    result = Anaphora(min_repeats=3, min_lines=4).verify(poem)

    assert result.passed is True
    assert result.details["phrase"] == "i have a dream"
    assert result.details["repeats"] == 3


def test_palindrome_poem_supports_word_and_letter_levels() -> None:
    word_result = PalindromePoem(level="word", min_lines=4).verify("alpha\nbeta\nbeta\nalpha")
    letter_result = PalindromePoem(level="letter", min_lines=3).verify("level\nrotor\ncivic")
    bad_result = PalindromePoem(level="word", min_lines=4).verify("alpha\nbeta\ngamma\nalpha")

    assert word_result.passed is True
    assert letter_result.passed is True
    assert bad_result.passed is False


def test_catalog_can_instantiate_constrained_forms_with_defaults() -> None:
    mesostic = instantiate_form("Mesostic")
    anaphora = instantiate_form("Anaphora")

    assert isinstance(mesostic, Mesostic)
    assert mesostic.target_word == "POEM"
    assert isinstance(anaphora, Anaphora)
    assert anaphora.target_phrase == "i am"
    assert anaphora.min_lines == 4
