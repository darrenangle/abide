"""Integration tests for the codex-spark warmup task and validation flow."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts import build_codex_spark_retry_tasks as build_retry_tasks_cli
from scripts import build_codex_spark_warmup_tasks as build_tasks_cli
from scripts import validate_codex_spark_warmup

from abide.forms.catalog import load_form_instances
from abide.training.codex_spark_warmup import (
    build_codex_spark_retry_tasks,
    validate_codex_spark_candidates,
)
from abide.training.codex_spark_warmup import (
    build_codex_spark_warmup_tasks as build_tasks,
)
from tests.fixtures.poems import (
    GHAZAL_SYNTHETIC_PERFECT,
    SESTINA_SYNTHETIC_PERFECT,
    SONNET_MILTON_BLINDNESS,
    TANKA_SYNTHETIC_PERFECT,
    VILLANELLE_SYNTHETIC_PERFECT,
)

if TYPE_CHECKING:
    from pathlib import Path


_VALID_POEMS_BY_FORM = {
    "Tanka": TANKA_SYNTHETIC_PERFECT,
    "PetrarchanSonnet": SONNET_MILTON_BLINDNESS,
    "Villanelle": VILLANELLE_SYNTHETIC_PERFECT,
    "Ghazal": GHAZAL_SYNTHETIC_PERFECT,
    "Sestina": SESTINA_SYNTHETIC_PERFECT,
}


def test_build_codex_spark_warmup_tasks_balances_forms() -> None:
    tasks = build_tasks(tasks_per_form=2, seed=7)

    assert len(tasks) == 10
    assert {task["form_name"] for task in tasks} == set(_VALID_POEMS_BY_FORM)
    assert all(task["task_id"] for task in tasks)
    assert all(task["messages"][0]["role"] == "user" for task in tasks)
    assert all("actual poem" in task["prompt"] for task in tasks)


def test_abecedarian_tasks_include_form_specific_quality_guidance() -> None:
    task = build_tasks(
        form_names="Abecedarian", tasks_per_form=1, seed=7, prompt_mode="brief_only"
    )[0]

    assert "real opening word" in task["prompt"]
    assert "coherent scene" in task["prompt"]
    assert "Theme:" in task["prompt"]
    assert "Tone:" in task["prompt"]


def test_missing_hard_forms_get_explicit_anti_scaffold_guidance() -> None:
    task = build_tasks(form_names="Canzone", tasks_per_form=1, seed=7, prompt_mode="brief_only")[0]

    assert "actual poem" in task["prompt"]
    assert "label, counter, heading, or exposed constraint trick" in task["prompt"]
    assert "earned inside the poem" in task["prompt"]


def test_build_codex_spark_warmup_tasks_can_target_all_exported_forms() -> None:
    tasks = build_tasks(form_set="all_forms", tasks_per_form=1, seed=7, prompt_mode="brief_only")

    assert len(tasks) == len(load_form_instances())
    assert {task["form_name"] for task in tasks} == set(load_form_instances())
    assert all("exact structural brief" in task["prompt"] for task in tasks)


def test_validate_codex_spark_candidates_accepts_valid_poems() -> None:
    tasks = build_tasks(tasks_per_form=1, seed=9)
    candidates = [
        {
            "task_id": task["task_id"],
            "response": _VALID_POEMS_BY_FORM[task["form_name"]],
        }
        for task in tasks
    ]

    validated_rows, summary = validate_codex_spark_candidates(
        candidates,
        task_rows=tasks,
        max_seed_similarity=1.0,
    )

    assert len(validated_rows) == len(tasks)
    assert summary["num_failures"] == 0
    assert summary["num_validated"] == len(tasks)


def test_validate_codex_spark_candidates_rejects_degenerate_but_valid_poem() -> None:
    tasks = build_tasks(form_names="PrimeVerse", tasks_per_form=1, seed=9, prompt_mode="brief_only")
    candidates = [
        {
            "task_id": tasks[0]["task_id"],
            "response": "a a\na a a\na a a a a\na a a a a a a\na a a a a a a a a a a\na a a a a a a a a a a a a",
        }
    ]

    validated_rows, summary = validate_codex_spark_candidates(
        candidates,
        task_rows=tasks,
    )

    assert validated_rows == []
    assert summary["num_failures"] == 1
    assert summary["failures"][0]["error"] == "degenerate_response"
    assert "count_form_placeholder_shell" in summary["failures"][0]["quality_issues"]


def test_codex_spark_warmup_clis_round_trip(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    validated_path = tmp_path / "validated.jsonl"

    build_exit_code = build_tasks_cli.main(
        [
            "--tasks-per-form",
            "1",
            "--output",
            str(tasks_path),
        ]
    )
    assert build_exit_code == 0

    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    candidate_rows = [
        {
            "task_id": task["task_id"],
            "response": _VALID_POEMS_BY_FORM[task["form_name"]],
        }
        for task in tasks
    ]
    with candidates_path.open("w") as handle:
        for row in candidate_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    validate_exit_code = validate_codex_spark_warmup.main(
        [
            "--tasks",
            str(tasks_path),
            "--candidates",
            str(candidates_path),
            "--output",
            str(validated_path),
            "--max-seed-similarity",
            "1.0",
        ]
    )
    assert validate_exit_code == 0

    validated_rows = [
        json.loads(line) for line in validated_path.read_text().splitlines() if line.strip()
    ]
    assert len(validated_rows) == len(tasks)


def test_validate_codex_spark_candidates_rejects_near_duplicate_peers() -> None:
    tasks = build_tasks(form_names="PetrarchanSonnet", tasks_per_form=2, seed=9)
    candidate_rows = [
        {
            "task_id": tasks[0]["task_id"],
            "response": SONNET_MILTON_BLINDNESS,
        },
        {
            "task_id": tasks[1]["task_id"],
            "response": SONNET_MILTON_BLINDNESS.replace(
                "When I consider how my light is spent,",
                "Now I consider how my light is spent,",
                1,
            ),
        },
    ]

    validated_rows, summary = validate_codex_spark_candidates(
        candidate_rows,
        task_rows=tasks,
        max_seed_similarity=1.0,
        max_peer_similarity=0.9,
    )

    assert len(validated_rows) == 1
    assert summary["num_failures"] == 1
    assert summary["failures"][0]["error"] == "too_close_to_peer"


def test_build_codex_spark_retry_tasks_includes_validator_feedback() -> None:
    tasks = build_tasks(form_names="PrimeVerse", tasks_per_form=1, seed=9, prompt_mode="brief_only")
    candidate_rows = [
        {
            "task_id": tasks[0]["task_id"],
            "response": "a a\na a a\na a a a a\na a a a a a a\na a a a a a a a a a a\na a a a a a a a a a a a a",
        }
    ]
    _, summary = validate_codex_spark_candidates(candidate_rows, task_rows=tasks)

    retry_tasks = build_codex_spark_retry_tasks(
        task_rows=tasks,
        candidate_rows=candidate_rows,
        failure_rows=summary["failures"],
    )

    assert len(retry_tasks) == 1
    retry_task = retry_tasks[0]
    assert retry_task["retry_round"] == 1
    assert len(retry_task["messages"]) == 3
    assert "low-quality generation" in retry_task["prompt"]
    assert "actual poem" in retry_task["prompt"]


def test_build_codex_spark_retry_cli_round_trip(tmp_path: Path) -> None:
    tasks = build_tasks(form_names="PrimeVerse", tasks_per_form=1, seed=9, prompt_mode="brief_only")
    task_path = tmp_path / "tasks.jsonl"
    candidate_path = tmp_path / "candidates.jsonl"
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "retry.jsonl"

    with task_path.open("w") as handle:
        for row in tasks:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with candidate_path.open("w") as handle:
        handle.write(
            json.dumps(
                {
                    "task_id": tasks[0]["task_id"],
                    "response": "a a\na a a\na a a a a\na a a a a a a\na a a a a a a a a a a\na a a a a a a a a a a a a",
                },
                sort_keys=True,
            )
            + "\n"
        )

    _, summary = validate_codex_spark_candidates(
        [
            {
                "task_id": tasks[0]["task_id"],
                "response": "a a\na a a\na a a a a\na a a a a a a\na a a a a a a a a a a\na a a a a a a a a a a a a",
            }
        ],
        task_rows=tasks,
    )
    summary_path.write_text(json.dumps(summary, sort_keys=True))

    exit_code = build_retry_tasks_cli.main(
        [
            "--tasks",
            str(task_path),
            "--candidates",
            str(candidate_path),
            "--summary",
            str(summary_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    retry_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(retry_rows) == 1
    assert retry_rows[0]["task_id"] == tasks[0]["task_id"]


def test_build_codex_spark_retry_tasks_supports_judge_reject_feedback() -> None:
    tasks = build_tasks(form_names="Haiku", tasks_per_form=1, seed=9)
    candidate_rows = [
        {
            "task_id": tasks[0]["task_id"],
            "response": "Cold rain taps the steps\nA blue cup cools beside the door\nDawn gathers in steam",
        }
    ]
    failure_rows = [
        {
            "task_id": tasks[0]["task_id"],
            "form_name": "Haiku",
            "error": "judge_reject",
            "judge_label": "template_scaffold",
            "judge_reason": "Reads like a template more than a poem.",
        }
    ]

    retry_tasks = build_codex_spark_retry_tasks(
        task_rows=tasks,
        candidate_rows=candidate_rows,
        failure_rows=failure_rows,
    )

    assert len(retry_tasks) == 1
    assert "poemness judge" in retry_tasks[0]["prompt"]
    assert "template_scaffold" in retry_tasks[0]["prompt"]
