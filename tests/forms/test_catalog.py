"""Tests for the form catalog helpers."""

import abide.forms as forms_module
from abide.forms.catalog import (
    RL_DEFAULT_FORM_NAMES,
    instantiate_form,
    load_form_instances,
    load_rl_default_form_instances,
)


def test_load_rl_default_form_instances_returns_exact_catalog_set() -> None:
    forms = load_rl_default_form_instances()
    assert tuple(forms.keys()) == RL_DEFAULT_FORM_NAMES


def test_rl_default_catalog_uses_tuned_shakespeare_defaults() -> None:
    form = load_rl_default_form_instances()["ShakespeareanSonnet"]
    assert getattr(form, "syllable_tolerance", None) == 2
    assert getattr(form, "rhyme_threshold", None) == 0.4


def test_catalog_instantiates_positional_poem_without_runtime_exception() -> None:
    form = instantiate_form("PositionalPoem")
    poem = "\n".join(["Three"] * 4)

    result = form.verify(poem)

    assert result.passed is True


def test_load_form_instances_covers_all_exported_forms() -> None:
    forms = load_form_instances()

    assert set(forms) == set(forms_module.__all__)
