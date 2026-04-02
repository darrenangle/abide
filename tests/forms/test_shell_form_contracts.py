import pytest

from abide.forms import (
    Aubade,
    BlankVerse,
    DramaticVerse,
    FreeVerse,
    HoratianOde,
    IrregularOde,
    Ode,
    PindaricOde,
    ProsePoem,
    Skeltonic,
)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Ode(min_lines=0), "min_lines must be positive"),
        (
            lambda: Ode(min_lines=5, max_lines=4),
            "max_lines must be greater than or equal to min_lines",
        ),
        (lambda: IrregularOde(min_lines=0, min_stanzas=1), "min_lines must be positive"),
        (lambda: IrregularOde(min_lines=1, min_stanzas=0), "min_stanzas must be positive"),
        (lambda: HoratianOde(stanza_count=0), "stanza_count must be positive"),
        (lambda: HoratianOde(stanza_size=0), "stanza_size must be positive"),
        (
            lambda: HoratianOde(stanza_size=4, rhyme_scheme="AB"),
            "rhyme_scheme length must match stanza_size",
        ),
        (lambda: PindaricOde(triads=0), "triads must be positive"),
        (lambda: PindaricOde(strophe_lines=0), "strophe_lines must be positive"),
        (lambda: PindaricOde(epode_lines=0), "epode_lines must be positive"),
        (lambda: Aubade(stanza_count=0), "stanza_count must be positive"),
        (lambda: Aubade(lines_per_stanza=7), "lines_per_stanza must be between 1 and 6"),
        (lambda: FreeVerse(min_lines=0), "min_lines must be positive"),
        (
            lambda: FreeVerse(min_lines=3, max_lines=2),
            "max_lines must be greater than or equal to min_lines",
        ),
        (lambda: FreeVerse(min_stanzas=0), "min_stanzas must be positive"),
        (
            lambda: FreeVerse(min_stanzas=2, max_stanzas=1),
            "max_stanzas must be greater than or equal to min_stanzas",
        ),
        (lambda: FreeVerse(min_words_per_line=-1), "min_words_per_line must be non-negative"),
        (
            lambda: FreeVerse(min_words_per_line=3, max_words_per_line=2),
            "max_words_per_line must be greater than or equal to min_words_per_line",
        ),
        (lambda: ProsePoem(min_paragraphs=0), "min_paragraphs must be positive"),
        (
            lambda: ProsePoem(min_paragraphs=2, max_paragraphs=1),
            "max_paragraphs must be greater than or equal to min_paragraphs",
        ),
        (lambda: ProsePoem(min_sentences=0), "min_sentences must be positive"),
        (
            lambda: ProsePoem(min_sentences=3, max_sentences=2),
            "max_sentences must be greater than or equal to min_sentences",
        ),
        (lambda: BlankVerse(min_lines=0), "min_lines must be positive"),
        (
            lambda: BlankVerse(min_lines=5, max_lines=4),
            "max_lines must be greater than or equal to min_lines",
        ),
        (lambda: BlankVerse(syllable_tolerance=-1), "syllable_tolerance must be non-negative"),
        (lambda: DramaticVerse(min_lines=0), "min_lines must be positive"),
        (lambda: Skeltonic(min_lines=0), "min_lines must be positive"),
        (
            lambda: Skeltonic(max_syllables_per_line=0),
            "max_syllables_per_line must be positive",
        ),
    ],
)
def test_shell_forms_reject_degenerate_constructor_bounds(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
