"""Constructor-contract tests for lexical constraints."""

import pytest

from abide.constraints import (
    Alliteration,
    AllWordsUnique,
    CharacterCount,
    CharacterPalindrome,
    CrossLineVowelWordCount,
    DoubleAcrostic,
    ExactCharacterBudget,
    ExactTotalCharacters,
    ExactTotalVowels,
    ExactWordCount,
    ForcedWords,
    LetterFrequency,
    LineEndsWith,
    LineStartsWith,
    MonosyllabicOnly,
    NoConsecutiveRepeats,
    NoSharedLetters,
    PositionalCharacter,
    VowelConsonantPattern,
    WordCount,
    WordLengthPattern,
    WordLengthStaircase,
)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: WordCount([]), "words_per_line must contain at least one positive count"),
        (lambda: WordCount(0), "words_per_line must contain at least one positive count"),
        (
            lambda: WordCount([3, 0]),
            "words_per_line must contain at least one positive count",
        ),
        (
            lambda: CharacterCount([]),
            "chars_per_line must contain at least one positive count",
        ),
        (
            lambda: CharacterCount(0),
            "chars_per_line must contain at least one positive count",
        ),
        (lambda: ForcedWords([]), "required_words must contain at least one word"),
        (lambda: ForcedWords([""]), "required_words must contain non-empty words"),
        (lambda: ForcedWords(["moon"], min_occurrences=0), "min_occurrences must be positive"),
        (lambda: WordLengthPattern([]), "pattern must contain at least one positive length"),
        (lambda: WordLengthPattern([1, 0]), "pattern must contain at least one positive length"),
        (
            lambda: WordLengthPattern([[3, 3], []]),
            "pattern must contain at least one positive length",
        ),
        (lambda: LineStartsWith([]), "patterns must contain at least one non-empty pattern"),
        (lambda: LineStartsWith(""), "patterns must contain at least one non-empty pattern"),
        (lambda: LineEndsWith([]), "patterns must contain at least one non-empty pattern"),
        (lambda: LineEndsWith(""), "patterns must contain at least one non-empty pattern"),
        (
            lambda: LetterFrequency("", min_percent=1),
            "letter must be a single alphabetic character",
        ),
        (lambda: LetterFrequency("AB"), "letter must be a single alphabetic character"),
        (
            lambda: LetterFrequency("A", min_percent=60, max_percent=40),
            "max_percent must be greater than or equal to min_percent",
        ),
        (
            lambda: Alliteration(letter="S", min_words=0),
            "min_words must be positive",
        ),
        (lambda: Alliteration(min_consecutive=0), "min_consecutive must be positive"),
        (lambda: NoConsecutiveRepeats(min_lines=0), "min_lines must be positive"),
        (
            lambda: VowelConsonantPattern(""),
            "pattern must contain only V and C characters",
        ),
        (
            lambda: VowelConsonantPattern("VX"),
            "pattern must contain only V and C characters",
        ),
        (lambda: ExactTotalCharacters(0), "total must be positive"),
        (lambda: ExactTotalVowels(0), "total must be positive"),
        (lambda: WordLengthStaircase(max_words=0), "max_words must be positive"),
        (lambda: CrossLineVowelWordCount(start_words=0), "start_words must be positive"),
        (lambda: NoSharedLetters([]), "pairs must contain at least one line pair"),
        (lambda: NoSharedLetters([(0, 1)]), "pair line numbers must be positive"),
        (
            lambda: NoSharedLetters("bogus"),
            "pairs must be 'consecutive', 'alternating', or a list of line pairs",
        ),
        (lambda: ExactWordCount(0), "total must be positive"),
        (
            lambda: DoubleAcrostic("", ""),
            "first_word must contain at least one alphabetic character",
        ),
        (lambda: MonosyllabicOnly(min_words=0), "min_words must be positive"),
        (lambda: ExactCharacterBudget("e", 0), "count must be positive"),
        (
            lambda: ExactCharacterBudget(["a"], 1),  # type: ignore[arg-type]
            "character must be a single character string",
        ),
        (
            lambda: PositionalCharacter([]),
            "positions must contain at least one \\(position, character\\) pair",
        ),
        (
            lambda: PositionalCharacter([(0, "a")]),
            "positions must be 1-based positive integers",
        ),
        (
            lambda: PositionalCharacter([(1, "ab")]),
            "position characters must be a single character string",
        ),
        (
            lambda: CharacterPalindrome(lines=[]),
            "lines must contain at least one 1-based line number",
        ),
        (
            lambda: CharacterPalindrome(lines=[0]),
            "lines must be 1-based positive integers",
        ),
    ],
)
def test_lexical_constraints_reject_invalid_constructor_values(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_nonuniform_word_count_penalizes_missing_expected_lines() -> None:
    result = WordCount([1, 2, 3]).verify("alpha")

    assert result.passed is False
    assert result.score == pytest.approx((1 / 3) ** 2)
    assert result.details["matches"] == 1
    assert result.details["linear_score"] == pytest.approx(1 / 3)
    assert "Missing 2 expected line(s)" in result.details["line_details"]


def test_uniform_word_count_keeps_open_ended_single_line_behavior() -> None:
    result = WordCount(1).verify("alpha")

    assert result.passed is True
    assert result.score == 1.0


def test_word_count_does_not_treat_punctuation_as_words() -> None:
    result = WordCount(1).verify("!!!")

    assert result.passed is False
    assert result.score == 0.0


def test_all_words_unique_penalizes_inputs_below_minimum_sample_size() -> None:
    result = AllWordsUnique(min_words=10).verify("alpha bravo")

    assert result.passed is False
    assert result.score == pytest.approx(0.04)
    assert result.details["adequacy"] == pytest.approx(0.2)
    assert result.details["linear_score"] == pytest.approx(0.2)


def test_monosyllabic_only_penalizes_inputs_below_minimum_sample_size() -> None:
    result = MonosyllabicOnly(min_words=10).verify("cat dog")

    assert result.passed is False
    assert result.score == pytest.approx(0.04)
    assert result.details["adequacy"] == pytest.approx(0.2)
    assert result.details["linear_score"] == pytest.approx(0.2)


def test_no_shared_letters_rejects_pairs_without_alphabetic_content() -> None:
    result = NoSharedLetters("consecutive").verify("!!!\n???\n...\n!!!\n???")

    assert result.passed is False
    assert result.score == 0.0
