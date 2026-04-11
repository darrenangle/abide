"""Tests for the form catalog helpers."""

import abide.forms as forms_module
from abide.forms.catalog import (
    RL_DEFAULT_FORM_NAMES,
    WELL_KNOWN_FORM_NAMES,
    WELL_KNOWN_LONG_FORM_NAMES,
    WELL_KNOWN_SHORT_FORM_NAMES,
    instantiate_form,
    load_form_instances,
    load_rl_default_form_instances,
    load_well_known_form_instances,
    load_well_known_long_form_instances,
    load_well_known_short_form_instances,
)


def test_load_rl_default_form_instances_returns_exact_catalog_set() -> None:
    forms = load_rl_default_form_instances()
    assert tuple(forms.keys()) == RL_DEFAULT_FORM_NAMES


def test_rl_default_catalog_uses_tuned_shakespeare_defaults() -> None:
    form = load_rl_default_form_instances()["ShakespeareanSonnet"]
    assert getattr(form, "syllable_tolerance", None) == 2
    assert getattr(form, "rhyme_threshold", None) == 0.4


def test_load_well_known_form_instances_returns_exact_catalog_set() -> None:
    forms = load_well_known_form_instances()
    assert tuple(forms.keys()) == WELL_KNOWN_FORM_NAMES


def test_load_well_known_short_form_instances_returns_exact_catalog_set() -> None:
    forms = load_well_known_short_form_instances()
    assert tuple(forms.keys()) == WELL_KNOWN_SHORT_FORM_NAMES


def test_load_well_known_long_form_instances_returns_exact_catalog_set() -> None:
    forms = load_well_known_long_form_instances()
    assert tuple(forms.keys()) == WELL_KNOWN_LONG_FORM_NAMES


def test_well_known_subset_is_stable_and_smaller_than_rl_default() -> None:
    assert set(WELL_KNOWN_FORM_NAMES).issubset(set(RL_DEFAULT_FORM_NAMES))
    assert len(WELL_KNOWN_FORM_NAMES) < len(RL_DEFAULT_FORM_NAMES)


def test_well_known_short_and_long_sets_partition_well_known_subset() -> None:
    assert set(WELL_KNOWN_SHORT_FORM_NAMES).isdisjoint(set(WELL_KNOWN_LONG_FORM_NAMES))
    assert set(WELL_KNOWN_SHORT_FORM_NAMES) | set(WELL_KNOWN_LONG_FORM_NAMES) == set(
        WELL_KNOWN_FORM_NAMES
    )


def test_catalog_instantiates_positional_poem_without_runtime_exception() -> None:
    form = instantiate_form("PositionalPoem")
    poem = "\n".join(["Three"] * 4)

    result = form.verify(poem)

    assert result.passed is True


def test_load_form_instances_covers_all_exported_forms() -> None:
    forms = load_form_instances()

    assert set(forms) == set(forms_module.__all__)


def test_catalog_combined_challenge_defaults_are_satisfiable() -> None:
    form = instantiate_form("CombinedChallenge")

    assert "EXACTLY 10 vowels total" in form.describe()

    result = form.verify("a ae eau ioui")

    assert result.passed is True


def test_catalog_terzanelle_no_longer_has_internal_line_count_mismatch() -> None:
    form = instantiate_form("Terzanelle")
    poem = "\n".join([f"line {idx} same rhyme" for idx in range(1, 20)])

    result = form.verify(poem)
    failed_criteria = {
        item["criterion"] for item in result.to_dict()["rubric"] if item["passed"] is False
    }

    assert "[w=2.0] Line count for scheme" not in failed_criteria
