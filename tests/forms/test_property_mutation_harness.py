from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from abide.forms import (
    BluesPoem,
    Haiku,
    Limerick,
    Pantoum,
    Rondeau,
    Sestina,
    Sonnet,
    Tanka,
    TerzaRima,
    Triolet,
    Villanelle,
)
from tests.fixtures.poems import (
    BLUES_SYNTHETIC_PERFECT,
    HAIKU_SYNTHETIC_PERFECT,
    LIMERICK_SYNTHETIC_PERFECT,
    PANTOUM_SYNTHETIC_PERFECT,
    RONDEAU_MCCRAE_FLANDERS,
    SESTINA_SYNTHETIC_PERFECT,
    SONNET_SHAKESPEARE_18,
    TANKA_SYNTHETIC_PERFECT,
    TERZA_RIMA_SYNTHETIC_PERFECT,
    TRIOLET_SYNTHETIC_PERFECT,
    VILLANELLE_SYNTHETIC_PERFECT,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from abide.constraints import Constraint


@dataclass(frozen=True)
class FixtureMutationCase:
    name: str
    form_factory: Callable[[], Constraint]
    poem: str
    protected_drop_content_indices: tuple[int, ...] = ()


LINE_MUTATION_CASES = [
    FixtureMutationCase("haiku", lambda: Haiku(), HAIKU_SYNTHETIC_PERFECT),
    FixtureMutationCase("tanka", lambda: Tanka(), TANKA_SYNTHETIC_PERFECT),
    FixtureMutationCase(
        "villanelle",
        lambda: Villanelle(strict=False, rhyme_threshold=0.5, refrain_threshold=0.8),
        VILLANELLE_SYNTHETIC_PERFECT,
    ),
    FixtureMutationCase("sestina", lambda: Sestina(strict=False), SESTINA_SYNTHETIC_PERFECT),
    FixtureMutationCase(
        "triolet",
        lambda: Triolet(strict=False, rhyme_threshold=0.5, refrain_threshold=0.8),
        TRIOLET_SYNTHETIC_PERFECT,
    ),
    FixtureMutationCase("pantoum", lambda: Pantoum(strict=False), PANTOUM_SYNTHETIC_PERFECT),
    FixtureMutationCase(
        "rondeau",
        lambda: Rondeau(strict=False, rhyme_threshold=0.5, refrain_threshold=0.7),
        RONDEAU_MCCRAE_FLANDERS,
    ),
    FixtureMutationCase(
        "limerick",
        lambda: Limerick(strict=False, rhyme_threshold=0.5),
        LIMERICK_SYNTHETIC_PERFECT,
    ),
    FixtureMutationCase(
        "sonnet", lambda: Sonnet(strict=False, syllable_tolerance=2), SONNET_SHAKESPEARE_18
    ),
    FixtureMutationCase("blues", lambda: BluesPoem(strict=False), BLUES_SYNTHETIC_PERFECT),
    FixtureMutationCase(
        "terza-rima",
        lambda: TerzaRima(strict=False, rhyme_threshold=0.5),
        TERZA_RIMA_SYNTHETIC_PERFECT,
        protected_drop_content_indices=(12,),
    ),
]

STANZA_MUTATION_CASES = [
    FixtureMutationCase(
        "villanelle",
        lambda: Villanelle(strict=False, rhyme_threshold=0.5, refrain_threshold=0.8),
        VILLANELLE_SYNTHETIC_PERFECT,
    ),
    FixtureMutationCase("sestina", lambda: Sestina(strict=False), SESTINA_SYNTHETIC_PERFECT),
    FixtureMutationCase("pantoum", lambda: Pantoum(strict=False), PANTOUM_SYNTHETIC_PERFECT),
    FixtureMutationCase("blues", lambda: BluesPoem(strict=False), BLUES_SYNTHETIC_PERFECT),
    FixtureMutationCase(
        "terza-rima",
        lambda: TerzaRima(strict=False, rhyme_threshold=0.5),
        TERZA_RIMA_SYNTHETIC_PERFECT,
    ),
]

INSERTED_LINE = st.lists(
    st.sampled_from(
        [
            "spare",
            "echo",
            "lantern",
            "river",
            "hush",
            "stone",
            "ember",
            "shadow",
            "bright",
            "slow",
        ]
    ),
    min_size=1,
    max_size=6,
).map(" ".join)


def _baseline_passes(case: FixtureMutationCase) -> None:
    result = case.form_factory().verify(case.poem)
    assert result.passed is True, f"{case.name} baseline should pass before mutation"


def _insert_line(poem: str, insertion_index: int, line: str) -> str:
    lines = poem.splitlines()
    lines.insert(insertion_index, line)
    return "\n".join(lines)


def _drop_content_line(poem: str, content_index: int) -> str:
    lines = poem.splitlines()
    del lines[content_index]
    return "\n".join(lines)


def _collapse_selected_breaks(poem: str, selected_blank_indices: set[int]) -> str:
    lines = poem.splitlines()
    kept = [line for idx, line in enumerate(lines) if idx not in selected_blank_indices]
    return "\n".join(kept)


@pytest.mark.parametrize("case", LINE_MUTATION_CASES, ids=lambda case: case.name)
@hypothesis_settings(max_examples=8, deadline=None)
@given(data=st.data(), extra_line=INSERTED_LINE)
def test_property_mutation_harness_rejects_inserted_lines(
    case: FixtureMutationCase,
    data: st.DataObject,
    extra_line: str,
) -> None:
    _baseline_passes(case)

    insertion_index = data.draw(
        st.integers(min_value=0, max_value=len(case.poem.splitlines())),
        label="insertion_index",
    )
    mutated = _insert_line(case.poem, insertion_index, extra_line)

    assert case.form_factory().verify(mutated).passed is False


@pytest.mark.parametrize("case", LINE_MUTATION_CASES, ids=lambda case: case.name)
@hypothesis_settings(max_examples=8, deadline=None)
@given(data=st.data())
def test_property_mutation_harness_rejects_dropped_content_lines(
    case: FixtureMutationCase,
    data: st.DataObject,
) -> None:
    _baseline_passes(case)

    lines = case.poem.splitlines()
    available_indices = [
        idx
        for idx, line in enumerate(lines)
        if line.strip() and idx not in case.protected_drop_content_indices
    ]
    content_index = data.draw(
        st.sampled_from(available_indices),
        label="content_index",
    )
    mutated = _drop_content_line(case.poem, content_index)

    assert case.form_factory().verify(mutated).passed is False


@pytest.mark.parametrize("case", STANZA_MUTATION_CASES, ids=lambda case: case.name)
@hypothesis_settings(max_examples=8, deadline=None)
@given(data=st.data())
def test_property_mutation_harness_rejects_partial_stanza_break_collapse(
    case: FixtureMutationCase,
    data: st.DataObject,
) -> None:
    _baseline_passes(case)

    lines = case.poem.splitlines()
    blank_indices = [idx for idx, line in enumerate(lines) if not line.strip()]
    selected_blank_indices = set(
        data.draw(
            st.lists(
                st.sampled_from(blank_indices),
                min_size=1,
                max_size=len(blank_indices),
                unique=True,
            ),
            label="selected_blank_indices",
        )
    )
    mutated = _collapse_selected_breaks(case.poem, selected_blank_indices)

    assert case.form_factory().verify(mutated).passed is False
