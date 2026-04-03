"""
Catalog helpers and curated form subsets for poetry training.

This module centralizes:
- default kwargs used when a form needs explicit construction params
- the current RL rollout subset
- the smaller well-known subset used by the legacy verifiers trainer
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

    from abide.constraints import Constraint


RL_DEFAULT_FORM_DEFAULTS: dict[str, dict[str, Any]] = {
    "Haiku": {},
    "Tanka": {},
    "Limerick": {},
    "ShakespeareanSonnet": {
        "syllable_tolerance": 2,
        "rhyme_threshold": 0.4,
    },
    "PetrarchanSonnet": {},
    "Villanelle": {},
    "Ghazal": {},
    "Sestina": {},
    "Triolet": {},
    "Pantoum": {},
    "TerzaRima": {},
    "Rondeau": {},
    "Clerihew": {},
    "BluesPoem": {},
}

RL_DEFAULT_FORM_NAMES: tuple[str, ...] = tuple(RL_DEFAULT_FORM_DEFAULTS)

WELL_KNOWN_FORM_DEFAULTS: dict[str, dict[str, Any]] = {
    "Haiku": {},
    "Tanka": {},
    "Limerick": {},
    "ShakespeareanSonnet": {
        "syllable_tolerance": 2,
        "rhyme_threshold": 0.4,
    },
    "PetrarchanSonnet": {},
    "Villanelle": {},
    "Ghazal": {},
    "Sestina": {},
}

WELL_KNOWN_FORM_NAMES: tuple[str, ...] = tuple(WELL_KNOWN_FORM_DEFAULTS)

SPECIAL_FORM_KWARGS: dict[str, dict[str, Any]] = {
    "StaircasePoem": {"num_words": 7},
    "DescendingStaircasePoem": {"num_words": 7},
    "VowelBudgetPoem": {"vowel_count": 30},
    "PrecisionVerse": {"chars_per_line": 25},
    "ExactWordPoem": {"word_count": 20},
    "CharacterBudgetPoem": {"character": "e", "count": 10},
    "TotalCharacterPoem": {"total_chars": 100},
    "FibonacciVerse": {"num_lines": 5},
    "TriangularVerse": {"num_lines": 4},
    "PiKu": {"num_lines": 5},
    "PrecisionHaiku": {"chars_per_line": (15, 21, 15)},
    "ArithmeticVerse": {"start_words": 2, "num_lines": 5},
    "AlternatingIsolation": {},
    "DoubleAcrosticPoem": {"first_word": "POEM", "last_word": "RIME"},
    "CombinedChallenge": {"num_words": 4},
    "Lipogram": {"forbidden": "e"},
    "Univocalic": {"vowel": "a"},
    "Mesostic": {"word": "POEM"},
    "Anaphora": {"phrase": "I am", "min_lines": 4},
    "ModularVerse": {"k": 1, "n": 3, "min_words": 15},
    "CoprimeVerse": {"num_lines": 4},
    "SquareStanzas": {"num_stanzas": 4},
    "SelfReferential": {"num_lines": 4},
    "GoldenRatioVerse": {"num_lines": 6},
    "PythagoreanTercet": {"num_stanzas": 2},
}


def instantiate_form(
    form_name: str,
    *,
    training_profile: bool = False,
    **overrides: Any,
) -> Constraint:
    """Instantiate a form by exported name."""
    forms_module = importlib.import_module("abide.forms")
    try:
        form_class = cast("type[Constraint]", getattr(forms_module, form_name))
    except AttributeError as exc:
        raise KeyError(f"Unknown form: {form_name}") from exc

    kwargs: dict[str, Any] = {}
    kwargs.update(SPECIAL_FORM_KWARGS.get(form_name, {}))
    if training_profile:
        kwargs.update(RL_DEFAULT_FORM_DEFAULTS.get(form_name, {}))
        kwargs.update(WELL_KNOWN_FORM_DEFAULTS.get(form_name, {}))
    kwargs.update(overrides)

    return form_class(**kwargs)


def load_form_instances(
    form_names: Iterable[str] | None = None,
    *,
    training_profile: bool = False,
) -> dict[str, Constraint]:
    """Load a mapping of form names to instances."""
    forms_module = importlib.import_module("abide.forms")
    names = list(form_names) if form_names is not None else list(forms_module.__all__)

    instances: dict[str, Constraint] = {}
    for name in names:
        try:
            instances[name] = instantiate_form(name, training_profile=training_profile)
        except TypeError:
            if form_names is not None:
                raise
            continue

    return instances


def load_rl_default_form_instances() -> dict[str, Constraint]:
    """Load the current curated RL-default subset with its tuned defaults."""
    return load_form_instances(RL_DEFAULT_FORM_NAMES, training_profile=True)


def load_well_known_form_instances() -> dict[str, Constraint]:
    """Load the smaller well-known subset with its tuned defaults."""
    return load_form_instances(WELL_KNOWN_FORM_NAMES, training_profile=True)


__all__ = [
    "RL_DEFAULT_FORM_DEFAULTS",
    "RL_DEFAULT_FORM_NAMES",
    "SPECIAL_FORM_KWARGS",
    "WELL_KNOWN_FORM_DEFAULTS",
    "WELL_KNOWN_FORM_NAMES",
    "instantiate_form",
    "load_form_instances",
    "load_rl_default_form_instances",
    "load_well_known_form_instances",
]
