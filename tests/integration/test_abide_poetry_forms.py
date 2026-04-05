"""Tests for the installable Abide verifiers environment package."""

from __future__ import annotations

import abide_poetry_forms


def test_load_environment_defaults_to_well_known_subset() -> None:
    env = abide_poetry_forms.load_environment(num_prompts=8, seed=7)

    assert env.dataset.num_rows == 8
    form_names = {row["info"]["form_name"] for row in env.dataset}
    assert form_names.issubset(set(abide_poetry_forms.resolve_prime_rl_form_instances()))


def test_load_environment_single_form_routes_metadata_exactly() -> None:
    env = abide_poetry_forms.load_environment(num_prompts=4, seed=11, form_name="Haiku")

    assert {row["info"]["form_name"] for row in env.dataset} == {"Haiku"}


def test_normalize_generated_poem_strips_tags_and_code_fences() -> None:
    raw = "<think>draft</think>```poem\nline one\nline two\n```<end_of_turn>"

    assert abide_poetry_forms.normalize_generated_poem(raw) == "line one\nline two"
