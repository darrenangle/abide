import pytest

from abide.forms import (
    Haiku,
    Limerick,
    Pantoum,
    Rondeau,
    Sestina,
    Sonnet,
    Tanka,
    Triolet,
    Villanelle,
)
from tests.fixtures.poems import (
    HAIKU_SYNTHETIC_PERFECT,
    LIMERICK_SYNTHETIC_PERFECT,
    PANTOUM_SYNTHETIC_PERFECT,
    RONDEAU_MCCRAE_FLANDERS,
    SESTINA_SYNTHETIC_PERFECT,
    SONNET_SHAKESPEARE_18,
    TANKA_SYNTHETIC_PERFECT,
    TRIOLET_SYNTHETIC_PERFECT,
    VILLANELLE_SYNTHETIC_PERFECT,
)


def _append_trailing_line(poem: str) -> str:
    return poem.rstrip() + "\nSpare trailing line"


def _drop_last_nonempty_line(poem: str) -> str:
    lines = poem.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            del lines[idx]
            break
    return "\n".join(lines)


CASES = [
    ("haiku", Haiku(), HAIKU_SYNTHETIC_PERFECT),
    ("tanka", Tanka(), TANKA_SYNTHETIC_PERFECT),
    (
        "villanelle",
        Villanelle(strict=False, rhyme_threshold=0.5, refrain_threshold=0.8),
        VILLANELLE_SYNTHETIC_PERFECT,
    ),
    ("sestina", Sestina(strict=False), SESTINA_SYNTHETIC_PERFECT),
    (
        "triolet",
        Triolet(strict=False, rhyme_threshold=0.5, refrain_threshold=0.8),
        TRIOLET_SYNTHETIC_PERFECT,
    ),
    ("pantoum", Pantoum(strict=False), PANTOUM_SYNTHETIC_PERFECT),
    (
        "rondeau",
        Rondeau(strict=False, rhyme_threshold=0.5, refrain_threshold=0.7),
        RONDEAU_MCCRAE_FLANDERS,
    ),
    ("limerick", Limerick(strict=False, rhyme_threshold=0.5), LIMERICK_SYNTHETIC_PERFECT),
    ("sonnet", Sonnet(strict=False, syllable_tolerance=2), SONNET_SHAKESPEARE_18),
]

FLATTENING_CASES = [
    (
        "villanelle",
        Villanelle(strict=False, rhyme_threshold=0.5, refrain_threshold=0.8),
        VILLANELLE_SYNTHETIC_PERFECT,
    ),
    ("sestina", Sestina(strict=False), SESTINA_SYNTHETIC_PERFECT),
    ("pantoum", Pantoum(strict=False), PANTOUM_SYNTHETIC_PERFECT),
]


@pytest.mark.parametrize(("name", "form", "poem"), CASES, ids=[name for name, _, _ in CASES])
def test_fixture_mutation_harness_rejects_added_line(name: str, form, poem: str) -> None:
    baseline = form.verify(poem)
    assert baseline.passed is True, f"{name} baseline should pass before mutation"

    mutated = form.verify(_append_trailing_line(poem))

    assert mutated.passed is False


@pytest.mark.parametrize(("name", "form", "poem"), CASES, ids=[name for name, _, _ in CASES])
def test_fixture_mutation_harness_rejects_missing_line(name: str, form, poem: str) -> None:
    baseline = form.verify(poem)
    assert baseline.passed is True, f"{name} baseline should pass before mutation"

    mutated = form.verify(_drop_last_nonempty_line(poem))

    assert mutated.passed is False


def test_pantoum_rejects_flattened_single_block_layout() -> None:
    form = Pantoum(strict=False)

    baseline = form.verify(PANTOUM_SYNTHETIC_PERFECT)
    assert baseline.passed is True

    flattened = form.verify(PANTOUM_SYNTHETIC_PERFECT.replace("\n\n", "\n"))

    assert flattened.passed is False


@pytest.mark.parametrize(
    ("name", "form", "poem"),
    FLATTENING_CASES,
    ids=[name for name, _, _ in FLATTENING_CASES],
)
def test_fixture_mutation_harness_rejects_flattened_stanza_layouts(
    name: str, form, poem: str
) -> None:
    baseline = form.verify(poem)
    assert baseline.passed is True, f"{name} baseline should pass before flattening"

    flattened = form.verify(poem.replace("\n\n", "\n"))

    assert flattened.passed is False
