"""Integration tests for the verifier-gated SFT warmup dataset builder."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from scripts import build_synthetic_sft_dataset

import abide_poetry_forms
from abide.forms.catalog import RL_DEFAULT_FORM_NAMES

if TYPE_CHECKING:
    from pathlib import Path


def test_build_synthetic_sft_records_cover_rl_default_forms() -> None:
    records = abide_poetry_forms.build_synthetic_sft_records(
        form_set="rl_default",
        prompt_variants_per_seed=1,
        seed=7,
    )

    assert {record["form_name"] for record in records} == set(RL_DEFAULT_FORM_NAMES)
    assert all(record["verifier_passed"] for record in records)
    assert all(len(record["messages"]) == 2 for record in records)
    assert all(record["messages"][0]["role"] == "user" for record in records)
    assert all(record["messages"][1]["role"] == "assistant" for record in records)
    assert all(record["response"] == record["messages"][1]["content"] for record in records)


def test_build_synthetic_sft_records_reject_missing_synthetic_coverage() -> None:
    with pytest.raises(ValueError, match="Missing seed poems"):
        abide_poetry_forms.build_synthetic_sft_records(
            form_set="rl_default",
            prompt_variants_per_seed=1,
            include_public_domain=False,
        )


def test_build_synthetic_sft_records_support_explicit_abecedarian_form() -> None:
    records = abide_poetry_forms.build_synthetic_sft_records(
        form_names="Abecedarian",
        prompt_variants_per_seed=1,
        seed=7,
        include_public_domain=False,
    )

    assert len(records) == 2
    assert {record["form_name"] for record in records} == {"Abecedarian"}
    assert all(record["verifier_passed"] for record in records)


def test_build_synthetic_sft_dataset_cli_writes_jsonl_and_summary(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.jsonl"
    summary_path = tmp_path / "summary.json"

    exit_code = build_synthetic_sft_dataset.main(
        [
            "--form-names",
            "Haiku,Tanka",
            "--prompt-variants-per-seed",
            "2",
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert summary_path.exists()

    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 4
    assert {row["form_name"] for row in rows} == {"Haiku", "Tanka"}

    summary = json.loads(summary_path.read_text())
    assert summary["num_records"] == 4
    assert summary["form_counts"] == {"Haiku": 2, "Tanka": 2}
