import pytest

from abide.forms import DoubleFibonacci, FibonacciPoem, FreeVerse, ProsePoem, ReverseFibonacci


def _mono_line(count: int, word: str = "sun") -> str:
    return " ".join([word] * count)


FIBONACCI_PERFECT = "\n".join([_mono_line(count) for count in [1, 1, 2, 3, 5, 8]])
FIBONACCI_NEAR_MISS = "\n".join([_mono_line(count) for count in [1, 1, 2, 4, 5, 8]])

REVERSE_FIBONACCI_PERFECT = "\n".join([_mono_line(count) for count in [8, 5, 3, 2, 1, 1]])
REVERSE_FIBONACCI_NEAR_MISS = "\n".join([_mono_line(count) for count in [8, 5, 4, 2, 1, 1]])

DOUBLE_FIBONACCI_PERFECT = "\n".join(
    [_mono_line(count) for count in [1, 1, 2, 3, 5, 8, 8, 5, 3, 2, 1, 1]]
)
DOUBLE_FIBONACCI_NEAR_MISS = "\n".join(
    [_mono_line(count) for count in [1, 1, 2, 3, 5, 9, 8, 5, 3, 2, 1, 1]]
)

FREE_VERSE_VALID = """\
The road keeps its own counsel
under the late blue air

We follow slowly
and let the silence widen"""

FREE_VERSE_TOO_SHORT = """\
Only one line
left to carry"""

PROSE_POEM_VALID = """\
The room remembers rain. Light gathers on the table. Dust turns quietly in the window.

No one speaks aloud. The kettle cools by degrees. Evening settles into the walls."""

PROSE_POEM_LINE_BROKEN = """\
The room remembers rain.
Light gathers on the table.
Dust turns quietly in the window."""


@pytest.mark.parametrize(
    ("factory", "poem"),
    [
        (FibonacciPoem, FIBONACCI_PERFECT),
        (ReverseFibonacci, REVERSE_FIBONACCI_PERFECT),
        (DoubleFibonacci, DOUBLE_FIBONACCI_PERFECT),
    ],
)
def test_fibonacci_family_accepts_valid_examples(factory, poem: str) -> None:
    result = factory().verify(poem)

    assert result.passed is True
    assert result.score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("factory", "poem"),
    [
        (FibonacciPoem, FIBONACCI_NEAR_MISS),
        (ReverseFibonacci, REVERSE_FIBONACCI_NEAR_MISS),
        (DoubleFibonacci, DOUBLE_FIBONACCI_NEAR_MISS),
    ],
)
def test_fibonacci_family_rejects_high_scoring_near_misses(factory, poem: str) -> None:
    result = factory().verify(poem)

    assert result.score > 0.7
    assert result.passed is False


def test_free_verse_enforces_configured_structural_bounds() -> None:
    form = FreeVerse(min_lines=4, min_stanzas=2)

    assert form.verify(FREE_VERSE_VALID).passed is True
    assert form.verify(FREE_VERSE_TOO_SHORT).passed is False


def test_free_verse_word_bounds_do_not_treat_punctuation_as_words() -> None:
    form = FreeVerse(min_lines=5, min_words_per_line=1)
    poem = "!!!\n???\n...\n!!!\n???"

    result = form.verify(poem)

    assert result.passed is False
    assert result.details["word_bound_violations"]


def test_prose_poem_rejects_line_broken_verse() -> None:
    form = ProsePoem(min_paragraphs=1, min_sentences=3)

    valid = form.verify(PROSE_POEM_VALID)
    line_broken = form.verify(PROSE_POEM_LINE_BROKEN)

    assert valid.passed is True
    assert line_broken.passed is False
    assert line_broken.details["line_broken_paragraphs"] == 1


def test_free_form_descriptions_stay_structural() -> None:
    assert "rhyme" not in FreeVerse().describe().lower()
    assert ProsePoem().describe() == "Prose Poem: 1-10 paragraphs"
