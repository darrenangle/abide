import pytest

from abide.forms.mathematical import (
    CoprimeVerse,
    GoldenRatioVerse,
    PiKu,
    PythagoreanTercet,
    SelfReferential,
    SquareStanzas,
)


def test_mathematical_forms_accept_valid_exact_examples() -> None:
    golden = "\n".join(["a" * length for length in [10, 16, 26, 42, 68]])
    assert GoldenRatioVerse().verify(golden).passed is True

    pythagorean = "\n".join(
        [
            "one two three",
            "one two three four",
            "one two three four five",
        ]
    )
    assert PythagoreanTercet().verify(pythagorean).passed is True

    coprime = "\n".join(
        [
            "a b",
            "a b c",
            "a b c d",
            "a b c d e",
            "a b c d e f",
        ]
    )
    assert CoprimeVerse().verify(coprime).passed is True

    square = "one\n\none two three four\n\none two three four five six seven eight nine"
    assert SquareStanzas(num_stanzas=3).verify(square).passed is True

    self_referential = "\n".join(
        [
            "one",
            "two two",
            "three three three",
            "four four four four",
            "five five five five five",
        ]
    )
    assert SelfReferential().verify(self_referential).passed is True


def test_mathematical_forms_reject_trailing_prefix_passes() -> None:
    golden = "\n".join(["a" * length for length in [10, 16, 26, 42, 68, 5]])
    result = GoldenRatioVerse().verify(golden)
    assert result.passed is False
    assert result.score == 0.0

    pythagorean = "\n".join(
        [
            "one two three",
            "one two three four",
            "one two three four five",
            "garbage trailing line",
        ]
    )
    result = PythagoreanTercet().verify(pythagorean)
    assert result.passed is False
    assert result.score == 0.0

    coprime = "\n".join(
        [
            "a b",
            "a b c",
            "a b c d",
            "a b c d e",
            "a b c d e f",
            "extra extra",
        ]
    )
    result = CoprimeVerse().verify(coprime)
    assert result.passed is False
    assert result.score == 0.0

    square = (
        "one\n\none two three four\n\none two three four five six seven eight nine"
        "\n\nextra stanza words here"
    )
    result = SquareStanzas(num_stanzas=3).verify(square)
    assert result.passed is False
    assert result.score == 0.0

    self_referential = "\n".join(
        [
            "one",
            "two two",
            "three three three",
            "four four four four",
            "five five five five five",
            "extra junk",
        ]
    )
    result = SelfReferential().verify(self_referential)
    assert result.passed is False
    assert result.score == 0.0


def test_self_referential_counts_whole_number_tokens_only() -> None:
    poem = "\n".join(
        [
            "someone",
            "twotwo",
            "threethreethree",
            "fourfourfourfour",
            "fivefivefivefivefive",
        ]
    )

    result = SelfReferential().verify(poem)

    assert result.passed is False
    assert result.score == 0.05


def test_bounded_mathematical_forms_reject_unsupported_line_counts_instead_of_clamping() -> None:
    with pytest.raises(ValueError, match="num_lines must be between 1 and 20"):
        PiKu(num_lines=25)

    with pytest.raises(ValueError, match="num_lines must be between 1 and 9"):
        SelfReferential(num_lines=12)
