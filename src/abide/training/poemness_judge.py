"""Task builders and summaries for a lightweight poemness/anti-gaming judge."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

JUDGE_RUBRIC = (
    "Judge whether the assistant response is a useful poem-like training example rather than a "
    "placeholder, verifier-gaming shell, or heavily templated scaffold. Reject outputs that are "
    "mostly filler tokens, repeated lines not demanded by the form, obvious lexical padding, or "
    "mechanical template substitution. Accept outputs that read like an actual poem even if they "
    "are simple."
)

JUDGE_LABELS: tuple[str, ...] = (
    "poem_like",
    "placeholder_text",
    "overly_repetitive",
    "template_scaffold",
    "semantic_thinness",
    "not_poem_like",
)


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    return [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]


def write_jsonl_rows(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return output


def build_poemness_judge_tasks(
    *,
    validated_rows: list[dict[str, Any]],
    retry_summary: dict[str, Any] | None = None,
    accepted_sample_size: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build judge tasks from the quality-gated corpus plus rejected disagreements."""
    retry_failures = list((retry_summary or {}).get("failures", []))
    rejected_task_ids = {str(failure.get("task_id", "")).strip() for failure in retry_failures}
    retry_failure_by_task_id = {
        str(failure.get("task_id", "")).strip(): failure
        for failure in retry_failures
        if str(failure.get("task_id", "")).strip()
    }
    validated_by_task_id = {
        str(row.get("task_id", "")).strip(): row
        for row in validated_rows
        if str(row.get("task_id", "")).strip()
    }

    tasks: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()

    for task_id in sorted(rejected_task_ids):
        row = validated_by_task_id.get(task_id)
        if row is None:
            continue
        failure = retry_failure_by_task_id[task_id]
        tasks.append(
            {
                "judge_task_id": f"{task_id}--judge",
                "task_id": task_id,
                "form_name": row["form_name"],
                "prompt": row["prompt"],
                "response": row["response"],
                "structural_brief": row["structural_brief"],
                "verifier_passed": row["verifier_passed"],
                "verifier_score": row["verifier_score"],
                "source": "heuristic_reject",
                "heuristic_failure": failure.get("error"),
                "judge_rubric": JUDGE_RUBRIC,
                "allowed_labels": list(JUDGE_LABELS),
            }
        )
        seen_task_ids.add(task_id)

    remaining_rows = [
        row for row in validated_rows if str(row.get("task_id", "")).strip() not in seen_task_ids
    ]
    rng = random.Random(seed)
    rng.shuffle(remaining_rows)
    for row in remaining_rows[:accepted_sample_size]:
        task_id = str(row["task_id"])
        tasks.append(
            {
                "judge_task_id": f"{task_id}--judge",
                "task_id": task_id,
                "form_name": row["form_name"],
                "prompt": row["prompt"],
                "response": row["response"],
                "structural_brief": row["structural_brief"],
                "verifier_passed": row["verifier_passed"],
                "verifier_score": row["verifier_score"],
                "source": "accepted_control",
                "judge_rubric": JUDGE_RUBRIC,
                "allowed_labels": list(JUDGE_LABELS),
            }
        )

    return tasks


def summarize_poemness_judgments(
    tasks: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize judge outputs against the prepared task sheet."""
    task_by_id = {str(task["judge_task_id"]): task for task in tasks}
    accepted = 0
    rejected = 0
    by_source: Counter[str] = Counter()
    by_label: Counter[str] = Counter()
    disagreements: list[dict[str, Any]] = []

    for judgment in judgments:
        judge_task_id = str(judgment.get("judge_task_id", "")).strip()
        if not judge_task_id or judge_task_id not in task_by_id:
            continue
        task = task_by_id[judge_task_id]
        passed = bool(judgment.get("judge_passed", False))
        if passed:
            accepted += 1
        else:
            rejected += 1
        by_source[str(task.get("source", "unknown"))] += 1
        by_label[str(judgment.get("label", "unknown"))] += 1

        if task.get("source") == "accepted_control" and not passed:
            disagreements.append(
                {
                    "judge_task_id": judge_task_id,
                    "task_id": task["task_id"],
                    "form_name": task["form_name"],
                    "source": "accepted_control",
                    "label": judgment.get("label"),
                    "reason": judgment.get("reason"),
                }
            )
        if task.get("source") == "heuristic_reject" and passed:
            disagreements.append(
                {
                    "judge_task_id": judge_task_id,
                    "task_id": task["task_id"],
                    "form_name": task["form_name"],
                    "source": "heuristic_reject",
                    "label": judgment.get("label"),
                    "reason": judgment.get("reason"),
                }
            )

    return {
        "num_tasks": len(tasks),
        "num_judgments": len(judgments),
        "judge_accepts": accepted,
        "judge_rejects": rejected,
        "by_source": dict(sorted(by_source.items())),
        "by_label": dict(sorted(by_label.items())),
        "disagreements": disagreements,
    }


__all__ = [
    "JUDGE_LABELS",
    "JUDGE_RUBRIC",
    "build_poemness_judge_tasks",
    "read_jsonl_rows",
    "summarize_poemness_judgments",
    "write_jsonl_rows",
]
