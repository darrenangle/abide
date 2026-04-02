"""Constructor-contract tests for lexical constraints."""

import pytest

from abide.constraints import ForcedWords, WordCount, WordLengthPattern


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: WordCount([]), "words_per_line must contain at least one positive count"),
        (lambda: WordCount(0), "words_per_line must contain at least one positive count"),
        (
            lambda: WordCount([3, 0]),
            "words_per_line must contain at least one positive count",
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
    ],
)
def test_lexical_constraints_reject_invalid_constructor_values(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
