import pytest
from tests.fixtures.poems import (
    GHAZAL_SYNTHETIC_PERFECT,
    HAIKU_SYNTHETIC_PERFECT,
    SESTINA_SYNTHETIC_PERFECT,
    SONNET_SHAKESPEARE_18,
    TANKA_SYNTHETIC_PERFECT,
    VILLANELLE_SYNTHETIC_PERFECT,
)

from abide.inference import analyze_poem, infer_form


def test_analyze_poem_reports_basic_haiku_structure() -> None:
    analysis = analyze_poem(HAIKU_SYNTHETIC_PERFECT)

    assert analysis.structure.line_count == 3
    assert analysis.structure.stanza_count == 1
    assert analysis.syllable_pattern == [5, 7, 5]
    assert analysis.refrains == []
    assert {constraint.id for constraint in analysis.constraints} == {
        "line_count",
        "syllables_exact",
    }


def test_analyze_poem_detects_villanelle_structure_and_refrains() -> None:
    analysis = analyze_poem(VILLANELLE_SYNTHETIC_PERFECT)

    assert analysis.structure.line_count == 19
    assert analysis.structure.stanza_count == 6
    assert list(analysis.structure.stanza_sizes) == [3, 3, 3, 3, 3, 4]
    assert analysis.rhyme_scheme is not None
    assert analysis.refrains == [(0, [5, 11, 17]), (2, [8, 14, 18])]
    assert {
        "line_count",
        "stanza_count",
        "stanza_sizes",
        "syllables_uniform",
        "rhyme_scheme",
        "refrain_line_1",
        "refrain_line_3",
    } <= {constraint.id for constraint in analysis.constraints}
    assert analysis.verify_score() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("name", "poem"),
    [
        ("Haiku", HAIKU_SYNTHETIC_PERFECT),
        ("Tanka", TANKA_SYNTHETIC_PERFECT),
        ("Shakespeare Example", SONNET_SHAKESPEARE_18),
        ("Villanelle", VILLANELLE_SYNTHETIC_PERFECT),
        ("Sestina", SESTINA_SYNTHETIC_PERFECT),
        ("Ghazal", GHAZAL_SYNTHETIC_PERFECT),
    ],
)
def test_infer_form_returns_self_consistent_spec(name: str, poem: str) -> None:
    spec = infer_form(poem, name=name)

    assert spec.name == name
    assert spec.weighted_score(poem) == pytest.approx(1.0)
