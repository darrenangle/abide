import pytest

from abide.forms.hard import (
    AlternatingIsolation,
    ArithmeticVerse,
    CharacterPalindromePoem,
    CombinedChallenge,
    DescendingStaircasePoem,
    DoubleAcrosticPoem,
    ExactWordPoem,
    PositionalPoem,
    PrecisionHaiku,
    PrecisionVerse,
    StaircasePoem,
    TotalCharacterPoem,
    VowelBudgetPoem,
)


def test_positional_poem_rejects_malformed_positions_early() -> None:
    with pytest.raises(ValueError, match="positions must be a list of"):
        PositionalPoem(positions=[(1, "T"), 2])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: PrecisionVerse(chars_per_line=0), "chars_per_line must be positive"),
        (lambda: VowelBudgetPoem(min_lines=0), "min_lines must be positive"),
        (lambda: StaircasePoem(num_words=0), "num_words must be positive"),
        (lambda: DescendingStaircasePoem(num_words=0), "num_words must be positive"),
        (lambda: ArithmeticVerse(num_lines=0), "num_lines must be positive"),
        (lambda: ArithmeticVerse(start_words=0), "start_words must be positive"),
        (lambda: CharacterPalindromePoem(num_lines=0), "num_lines must be positive"),
        (
            lambda: DoubleAcrosticPoem(first_word="", last_word=""),
            "Words must contain at least one alphabetic character",
        ),
        (lambda: TotalCharacterPoem(total_chars=0), "total_chars must be positive"),
        (lambda: TotalCharacterPoem(min_lines=0), "min_lines must be positive"),
        (lambda: ExactWordPoem(word_count=0), "word_count must be positive"),
        (lambda: ExactWordPoem(min_lines=0), "min_lines must be positive"),
        (
            lambda: PrecisionHaiku(chars_per_line=(0, 0, 0)),
            "chars_per_line must contain exactly three positive integers",
        ),
        (lambda: CombinedChallenge(num_words=0), "num_words must be positive"),
    ],
)
def test_hard_forms_reject_degenerate_constructor_values(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_alternating_isolation_does_not_reward_punctuation_only_lines() -> None:
    result = AlternatingIsolation().verify("!!!\n???\n...\n!!!\n???")

    assert result.passed is False
    assert result.score < 0.2
