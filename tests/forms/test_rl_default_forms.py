"""Generated adversarial checks for the curated RL-default form subset."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from abide import verify
from abide.forms.catalog import load_rl_default_form_instances
from tests.fixtures.poems import (
    BLUES_SYNTHETIC_PERFECT,
    CLERIHEW_SYNTHETIC_PERFECT,
    GHAZAL_SYNTHETIC_PERFECT,
    HAIKU_SYNTHETIC_PERFECT,
    LIMERICK_SYNTHETIC_PERFECT,
    PANTOUM_SYNTHETIC_PERFECT,
    RONDEAU_MCCRAE_FLANDERS,
    SESTINA_SYNTHETIC_PERFECT,
    SONNET_MILTON_BLINDNESS,
    SONNET_SHAKESPEARE_18,
    TANKA_SYNTHETIC_PERFECT,
    TERZA_RIMA_SYNTHETIC_PERFECT,
    TRIOLET_SYNTHETIC_PERFECT,
    VILLANELLE_SYNTHETIC_PERFECT,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _haiku_near_miss(poem: str) -> str:
    return poem + "\nQuiet echoes gather"


def _tanka_near_miss(poem: str) -> str:
    return "\n".join(poem.splitlines()[:-1])


def _limerick_near_miss(poem: str) -> str:
    lines = poem.splitlines()
    lines[-1] = "And drifted away in the storm"
    return "\n".join(lines)


def _shakespearean_near_miss(poem: str) -> str:
    lines = poem.splitlines()
    lines[-1] = lines[-2]
    return "\n".join(lines)


def _petrarchan_near_miss(poem: str) -> str:
    lines = poem.splitlines()
    lines[1] = lines[0]
    return "\n".join(lines)


def _villanelle_near_miss(poem: str) -> str:
    lines = poem.splitlines()
    lines[5] = "A lone bell falters in the rain"
    return "\n".join(lines)


def _ghazal_near_miss(poem: str) -> str:
    couplets = poem.split("\n\n")
    broken_couplet = couplets[2].splitlines()
    broken_couplet[1] = "The world is calm beneath the cedar plain"
    couplets[2] = "\n".join(broken_couplet)
    return "\n\n".join(couplets)


def _sestina_near_miss(poem: str) -> str:
    lines = poem.splitlines()
    first_stanza = [re.sub(r"\b\w+\b(?=[^\w]*$)", "stone", line) for line in lines[:6]]
    return "\n".join(first_stanza + lines[6:])


def _triolet_near_miss(poem: str) -> str:
    lines = poem.splitlines()
    lines[-1] = "And morning sinks beneath the harbor"
    return "\n".join(lines)


def _pantoum_near_miss(poem: str) -> str:
    stanzas = [stanza.splitlines() for stanza in poem.split("\n\n")]
    stanzas[-1][1] = "A stranger lingers near the shuttered door"
    stanzas[-1][3] = "The lantern gutters on the attic floor"
    return "\n\n".join("\n".join(stanza) for stanza in stanzas)


def _terza_rima_near_miss(poem: str) -> str:
    lines = poem.splitlines()
    lines[-1] = "A quiet lantern sleeps beneath the stone"
    return "\n".join(lines)


def _rondeau_near_miss(poem: str) -> str:
    lines = poem.splitlines()
    lines[8] = "We carry ashes through the wind."
    return "\n".join(lines)


def _clerihew_near_miss(poem: str) -> str:
    lines = poem.splitlines()
    lines[0] = "a patient lantern waited there"
    return "\n".join(lines)


def _blues_near_miss(poem: str) -> str:
    stanzas = [stanza.splitlines() for stanza in poem.split("\n\n")]
    stanzas[0][1] = "The station master counted out the rain"
    return "\n\n".join("\n".join(stanza) for stanza in stanzas)


RL_DEFAULT_FIXTURES: list[tuple[str, str, Callable[[str], str]]] = [
    ("Haiku", HAIKU_SYNTHETIC_PERFECT, _haiku_near_miss),
    ("Tanka", TANKA_SYNTHETIC_PERFECT, _tanka_near_miss),
    ("Limerick", LIMERICK_SYNTHETIC_PERFECT, _limerick_near_miss),
    ("ShakespeareanSonnet", SONNET_SHAKESPEARE_18, _shakespearean_near_miss),
    ("PetrarchanSonnet", SONNET_MILTON_BLINDNESS, _petrarchan_near_miss),
    ("Villanelle", VILLANELLE_SYNTHETIC_PERFECT, _villanelle_near_miss),
    ("Ghazal", GHAZAL_SYNTHETIC_PERFECT, _ghazal_near_miss),
    ("Sestina", SESTINA_SYNTHETIC_PERFECT, _sestina_near_miss),
    ("Triolet", TRIOLET_SYNTHETIC_PERFECT, _triolet_near_miss),
    ("Pantoum", PANTOUM_SYNTHETIC_PERFECT, _pantoum_near_miss),
    ("TerzaRima", TERZA_RIMA_SYNTHETIC_PERFECT, _terza_rima_near_miss),
    ("Rondeau", RONDEAU_MCCRAE_FLANDERS, _rondeau_near_miss),
    ("Clerihew", CLERIHEW_SYNTHETIC_PERFECT, _clerihew_near_miss),
    ("BluesPoem", BLUES_SYNTHETIC_PERFECT, _blues_near_miss),
]


@pytest.mark.parametrize(
    ("form_name", "positive_poem", "mutator"),
    RL_DEFAULT_FIXTURES,
    ids=[item[0] for item in RL_DEFAULT_FIXTURES],
)
def test_rl_default_forms_pass_positive_fixture_and_fail_single_property_mutation(
    form_name: str,
    positive_poem: str,
    mutator: Callable[[str], str],
) -> None:
    forms = load_rl_default_form_instances()
    form = forms[form_name]

    positive = verify(positive_poem, form)
    negative = verify(mutator(positive_poem), form)

    assert positive.passed is True
    assert positive.score >= 0.9
    assert negative.passed is False
    assert negative.score < positive.score
