"""Integration tests for the lightweight poemness judge task flow."""

from __future__ import annotations

import json

from scripts import build_poemness_judge_tasks as build_judge_tasks_cli
from scripts import merge_poemness_judge_results as merge_judge_results_cli

from abide.training.poemness_judge import (
    build_poemness_judge_tasks,
    summarize_poemness_judgments,
)


def _validated_rows() -> list[dict[str, object]]:
    return [
        {
            "task_id": "haiku-001",
            "form_name": "Haiku",
            "prompt": "Write a haiku about rain.",
            "response": "Cold rain taps the steps\nA blue cup cools beside the door\nDawn gathers in steam",
            "structural_brief": "Haiku: 3 lines with 5-7-5 syllable pattern",
            "verifier_passed": True,
            "verifier_score": 1.0,
        },
        {
            "task_id": "primeverse-001",
            "form_name": "PrimeVerse",
            "prompt": "Write a prime verse.",
            "response": "a a\na a a\na a a a a\na a a a a a a\na a a a a a a a a a a\na a a a a a a a a a a a a",
            "structural_brief": "Prime Verse: 6 lines with prime-number word counts",
            "verifier_passed": True,
            "verifier_score": 1.0,
        },
        {
            "task_id": "villanelle-001",
            "form_name": "Villanelle",
            "prompt": "Write a villanelle about dusk.",
            "response": "The lamp stays lit when evening folds the square\nThe buses hiss and empty down the lane\nI keep one cup of warmth against the air",
            "structural_brief": "Villanelle: 19 lines with two refrains",
            "verifier_passed": True,
            "verifier_score": 1.0,
        },
    ]


def _retry_summary() -> dict[str, object]:
    return {
        "failures": [
            {
                "task_id": "primeverse-001",
                "error": "degenerate_response",
                "quality_issues": ["count_form_placeholder_shell"],
            }
        ]
    }


def test_build_poemness_judge_tasks_includes_rejects_and_controls() -> None:
    tasks = build_poemness_judge_tasks(
        validated_rows=_validated_rows(),
        retry_summary=_retry_summary(),
        accepted_sample_size=1,
        seed=5,
    )

    assert len(tasks) == 2
    assert {task["source"] for task in tasks} == {"heuristic_reject", "accepted_control"}
    assert any(task["task_id"] == "primeverse-001" for task in tasks)
    assert all("judge_rubric" in task for task in tasks)


def test_summarize_poemness_judgments_records_disagreements() -> None:
    tasks = build_poemness_judge_tasks(
        validated_rows=_validated_rows(),
        retry_summary=_retry_summary(),
        accepted_sample_size=1,
        seed=5,
    )
    judgments = [
        {
            "judge_task_id": "primeverse-001--judge",
            "judge_passed": False,
            "label": "placeholder_text",
            "reason": "Repeated single-letter filler.",
        },
        {
            "judge_task_id": f"{next(task['task_id'] for task in tasks if task['source'] == 'accepted_control')}--judge",
            "judge_passed": False,
            "label": "not_poem_like",
            "reason": "Control disagreement.",
        },
    ]

    summary = summarize_poemness_judgments(tasks, judgments)

    assert summary["judge_rejects"] == 2
    assert summary["by_label"]["placeholder_text"] == 1
    assert len(summary["disagreements"]) == 1
    assert summary["disagreements"][0]["source"] == "accepted_control"


def test_poemness_judge_clis_round_trip(tmp_path) -> None:
    validated_path = tmp_path / "validated.jsonl"
    retry_summary_path = tmp_path / "retry.summary.json"
    tasks_path = tmp_path / "judge_tasks.jsonl"
    merged_path = tmp_path / "judgments.jsonl"

    with validated_path.open("w") as handle:
        for row in _validated_rows():
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    retry_summary_path.write_text(json.dumps(_retry_summary(), sort_keys=True))

    build_exit = build_judge_tasks_cli.main(
        [
            "--validated",
            str(validated_path),
            "--retry-summary",
            str(retry_summary_path),
            "--accepted-sample-size",
            "1",
            "--output",
            str(tasks_path),
        ]
    )
    assert build_exit == 0

    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    judgment_inputs = tmp_path / "judge_shard.jsonl"
    with judgment_inputs.open("w") as handle:
        for task in tasks:
            handle.write(
                json.dumps(
                    {
                        "judge_task_id": task["judge_task_id"],
                        "judge_passed": task["source"] == "accepted_control",
                        "label": "poem_like"
                        if task["source"] == "accepted_control"
                        else "placeholder_text",
                        "reason": "test",
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    merge_exit = merge_judge_results_cli.main(
        [
            "--tasks",
            str(tasks_path),
            "--inputs",
            str(judgment_inputs),
            "--output",
            str(merged_path),
        ]
    )
    assert merge_exit == 0
    merged = [json.loads(line) for line in merged_path.read_text().splitlines() if line.strip()]
    assert len(merged) == len(tasks)
