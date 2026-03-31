"""
Catalog and support tiers for poetic forms.

This module centralizes:
- support-tier classification for exported forms
- default kwargs used when a form needs explicit construction params
- the conservative training-safe subset used by RL scripts
"""

from __future__ import annotations

import importlib
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

    from abide.constraints import Constraint


class FormSupportTier(str, Enum):
    """Reliability/support tier for a form verifier."""

    TRAINING_SAFE = "training_safe"
    BEST_EFFORT = "best_effort"
    EXPERIMENTAL = "experimental"


TRAINING_SAFE_FORM_DEFAULTS: dict[str, dict[str, Any]] = {
    "Haiku": {},
    "Tanka": {},
    "Limerick": {},
    "ShakespeareanSonnet": {
        "syllable_tolerance": 2,
        "rhyme_threshold": 0.4,
    },
    "PetrarchanSonnet": {},
    "Villanelle": {},
    "Sestina": {},
    "Triolet": {},
    "Pantoum": {},
    "TerzaRima": {},
    "Rondeau": {},
    "Clerihew": {},
    "BluesPoem": {},
}

TRAINING_SAFE_FORM_NAMES: tuple[str, ...] = tuple(TRAINING_SAFE_FORM_DEFAULTS)

_HISTORICAL_FORM_NAMES: tuple[str, ...] = (
    "Sonnet",
    "ShakespeareanSonnet",
    "PetrarchanSonnet",
    "Haiku",
    "Senryu",
    "Tanka",
    "Limerick",
    "Couplet",
    "Tercet",
    "Quatrain",
    "FreeVerse",
    "BlankVerse",
    "Ballad",
    "BalladStanza",
    "Ode",
    "Villanelle",
    "Ghazal",
    "Sestina",
    "Rondeau",
    "Rondel",
    "Triolet",
    "Pantoum",
    "TerzaRima",
    "OttavaRima",
    "RhymeRoyal",
    "SpenserianStanza",
    "SpenserianSonnet",
    "CurtalSonnet",
    "CaudateSonnet",
    "HeroicCouplet",
    "HeroicQuatrain",
    "EnvelopeQuatrain",
    "Aubade",
    "Elegiac",
    "Epigram",
    "Cinquain",
    "Clerihew",
    "Rispetto",
    "Canzone",
    "Rubai",
    "Rubaiyat",
    "Diamante",
    "Etheree",
    "ReverseEtheree",
    "Ballade",
    "DoubleBallade",
    "ChantRoyal",
    "Kyrielle",
    "KyrielleSonnet",
    "BurnsStanza",
    "OneginStanza",
    "Lai",
    "Virelai",
    "Roundel",
    "Rondelet",
    "Rondine",
    "Tritina",
    "Quatina",
    "Quintina",
    "SandwichSonnet",
    "CrownOfSonnets",
    "Tanaga",
    "Katauta",
    "Sedoka",
    "Naani",
    "Seguidilla",
    "BluesPoem",
    "Bop",
    "ProsePoem",
    "DramaticVerse",
    "Monostich",
    "Distich",
    "Triplet",
    "Terzanelle",
    "SapphicStanza",
    "SapphicOde",
    "PindaricOde",
    "HoratianOde",
    "IrregularOde",
    "LiteraryBallad",
    "BroadBallad",
    "Skeltonic",
)

BEST_EFFORT_FORM_NAMES: tuple[str, ...] = tuple(
    name for name in _HISTORICAL_FORM_NAMES if name not in TRAINING_SAFE_FORM_DEFAULTS
)

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
    "PrecisionHaiku": {"chars_per_line": 17},
    "ArithmeticVerse": {"start": 2, "diff": 2, "num_lines": 5},
    "PositionalPoem": {"positions": [1, 2, 3]},
    "IsolatedCouplet": {"position": 3},
    "AlternatingIsolation": {"num_lines": 6},
    "DoubleAcrosticPoem": {"word": "POETRY"},
    "CombinedChallenge": {"num_lines": 4},
    "Lipogram": {"forbidden": "e"},
    "Univocalic": {"vowel": "a"},
    "Mesostic": {"spine": "POEM"},
    "Anaphora": {"phrase": "I am", "num_lines": 4},
    "ModularVerse": {"modulus": 3, "num_lines": 6},
    "CoprimeVerse": {"base": 6, "num_lines": 4},
    "SquareStanzas": {"size": 4},
    "SelfReferential": {"num_lines": 4},
    "GoldenRatioVerse": {"num_lines": 6},
    "PythagoreanTercet": {"scale": 2},
}


def get_form_support_tier(form_name: str) -> FormSupportTier:
    """Return the support tier for a form."""
    if form_name in TRAINING_SAFE_FORM_DEFAULTS:
        return FormSupportTier.TRAINING_SAFE
    if form_name in BEST_EFFORT_FORM_NAMES:
        return FormSupportTier.BEST_EFFORT
    return FormSupportTier.EXPERIMENTAL


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
        kwargs.update(TRAINING_SAFE_FORM_DEFAULTS.get(form_name, {}))
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


def load_training_safe_form_instances() -> dict[str, Constraint]:
    """Load the conservative training-safe subset with its tuned defaults."""
    return load_form_instances(TRAINING_SAFE_FORM_NAMES, training_profile=True)


__all__ = [
    "BEST_EFFORT_FORM_NAMES",
    "SPECIAL_FORM_KWARGS",
    "TRAINING_SAFE_FORM_DEFAULTS",
    "TRAINING_SAFE_FORM_NAMES",
    "FormSupportTier",
    "get_form_support_tier",
    "instantiate_form",
    "load_form_instances",
    "load_training_safe_form_instances",
]
