"""Tests for the form catalog helpers."""

from abide.forms.catalog import (
    RL_DEFAULT_FORM_NAMES,
    load_rl_default_form_instances,
)


def test_load_rl_default_form_instances_returns_exact_catalog_set() -> None:
    forms = load_rl_default_form_instances()
    assert tuple(forms.keys()) == RL_DEFAULT_FORM_NAMES


def test_rl_default_catalog_uses_tuned_shakespeare_defaults() -> None:
    form = load_rl_default_form_instances()["ShakespeareanSonnet"]
    assert getattr(form, "syllable_tolerance", None) == 2
    assert getattr(form, "rhyme_threshold", None) == 0.4
