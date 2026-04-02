"""Direct regressions for the Epigram structural proxy."""

from __future__ import annotations

from abide import verify
from abide.forms import Epigram

THREE_LINE_ESSAY = """\
This line is very long and drifts through several clauses without any compactness at all
This second line also rambles beyond anything that feels epigrammatic or compressed in form
And the third line simply continues the prose without rhyme or a pointed turn of any kind"""

THREE_LINE_EPIGRAM = """\
Sharp bells ring
Small sparks sting
Night folds in"""

FOUR_LINE_NO_RHYME = """\
Alpha stone
Beta field
Gamma lantern
Delta harbor"""


def test_epigram_rejects_three_line_prose_blocks() -> None:
    result = verify(THREE_LINE_ESSAY, Epigram())

    assert result.passed is False
    assert result.score < 1.0
    assert result.details["line_count"] == 3


def test_epigram_accepts_compact_three_line_example() -> None:
    result = verify(THREE_LINE_EPIGRAM, Epigram())

    assert result.passed is True
    assert result.score == 1.0


def test_epigram_keeps_four_line_rhyme_requirement() -> None:
    result = verify(FOUR_LINE_NO_RHYME, Epigram())

    assert result.passed is False
