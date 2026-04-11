"""Tests for the installable Abide verifiers environment package."""

from __future__ import annotations

import abide_poetry_forms


def test_load_environment_defaults_to_well_known_subset() -> None:
    env = abide_poetry_forms.load_environment(num_prompts=8, seed=7)

    assert env.dataset.num_rows == 8
    form_names = {row["info"]["form_name"] for row in env.dataset}
    assert form_names.issubset(set(abide_poetry_forms.resolve_prime_rl_form_instances()))


def test_load_environment_supports_short_and_long_well_known_subsets() -> None:
    short_env = abide_poetry_forms.load_environment(
        num_prompts=6, seed=7, form_set="well_known_short"
    )
    long_env = abide_poetry_forms.load_environment(
        num_prompts=6, seed=7, form_set="well_known_long"
    )

    assert {row["info"]["form_name"] for row in short_env.dataset}.issubset(
        {"Haiku", "Tanka", "Limerick"}
    )
    assert {row["info"]["form_name"] for row in long_env.dataset}.issubset(
        {"ShakespeareanSonnet", "PetrarchanSonnet", "Villanelle", "Ghazal", "Sestina"}
    )


def test_load_environment_single_form_routes_metadata_exactly() -> None:
    env = abide_poetry_forms.load_environment(num_prompts=4, seed=11, form_name="Haiku")

    assert {row["info"]["form_name"] for row in env.dataset} == {"Haiku"}


def test_build_prompt_records_are_deterministic_and_evenly_distributed() -> None:
    records_a = abide_poetry_forms.build_prime_rl_prompt_records(
        num_prompts=6,
        seed=19,
        form_set="well_known_short",
    )
    records_b = abide_poetry_forms.build_prime_rl_prompt_records(
        num_prompts=6,
        seed=19,
        form_set="well_known_short",
    )

    assert records_a == records_b
    assert [row["info"]["form_name"] for row in records_a].count("Haiku") == 2
    assert [row["info"]["form_name"] for row in records_a].count("Tanka") == 2
    assert [row["info"]["form_name"] for row in records_a].count("Limerick") == 2


def test_normalize_generated_poem_strips_tags_and_code_fences() -> None:
    raw = "<think>draft</think>```poem\nline one\nline two\n```<end_of_turn>"

    assert abide_poetry_forms.normalize_generated_poem(raw) == "line one\nline two"
