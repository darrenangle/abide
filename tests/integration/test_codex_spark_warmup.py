"""Integration tests for the codex-spark warmup task and validation flow."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts import build_codex_spark_warmup_tasks as build_tasks_cli
from scripts import validate_codex_spark_warmup

from abide.training.codex_spark_warmup import (
    build_codex_spark_warmup_tasks as build_tasks,
)
from abide.training.codex_spark_warmup import (
    validate_codex_spark_candidates,
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
