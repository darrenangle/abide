from abide.forms.constrained import Abecedarian, Mesostic
from abide.forms.free_verse import FreeVerse


def test_abecedarian_and_mesostic_accept_valid_exact_examples() -> None:
    abecedarian = "Alpha\nBravo\nCharlie\nDelta"
    result = Abecedarian(letters="ABCD").verify(abecedarian)
    assert result.passed is True

    mesostic = "aPex\nclOud\ntrEe\naMber"
    result = Mesostic(word="POEM").verify(mesostic)
    assert result.passed is True


def test_exact_count_constrained_forms_reject_missing_trailing_lines() -> None:
    abecedarian = "Alpha\nBravo\nCharlie"
    result = Abecedarian(letters="ABCD").verify(abecedarian)
    assert result.passed is False
    assert result.score < 0.8

    mesostic = "aPex\nclOud\ntrEe"
    result = Mesostic(word="POEM").verify(mesostic)
    assert result.passed is False
    assert result.score < 0.8


def test_free_verse_enforces_configured_word_bounds_canonically() -> None:
    valid = "\n".join(
        [
            "one two three",
            "four five six",
            "seven eight nine",
        ]
    )
    result = FreeVerse(min_lines=3, max_words_per_line=3).verify(valid)
    assert result.passed is True

    invalid = "\n".join(
        [
            "one two three four five six seven eight nine ten",
            "one two three four five six seven eight nine ten",
            "one two three four five six seven eight nine ten",
        ]
    )
    result = FreeVerse(min_lines=3, max_words_per_line=3).verify(invalid)
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["word_bound_violations"]
