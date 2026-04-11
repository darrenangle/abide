#!/usr/bin/env python3
"""Build retry tasks from the merged structural and model-judge decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abide.training.codex_spark_warmup import (
    build_codex_spark_retry_tasks,
    read_jsonl_rows,
    write_jsonl_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_TASKS = "data/sft/codex_spark_full_surface_tasks.jsonl"
DEFAULT_CANDIDATES = "data/sft/codex_spark_full_surface_candidates.jsonl"
DEFAULT_JUDGE_TASKS = "data/sft/codex_spark_full_surface_judge_tasks.jsonl"
DEFAULT_JUDGMENTS = "data/sft/codex_spark_full_surface_judgments.jsonl"
DEFAULT_OUTPUT = "data/sft/codex_spark_full_surface_adjudicated_retry_tasks.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--judge-tasks", default=DEFAULT_JUDGE_TASKS)
    parser.add_argument("--judgments", default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    task_rows = read_jsonl_rows(args.tasks)
    candidate_rows = read_jsonl_rows(args.candidates)
    judge_tasks = read_jsonl_rows(args.judge_tasks)
    judgments = read_jsonl_rows(args.judgments)
    judge_task_by_id = {str(task["judge_task_id"]): task for task in judge_tasks}

    failure_rows: list[dict[str, object]] = []
    for judgment in judgments:
        if bool(judgment.get("judge_passed", False)):
            continue
        judge_task_id = str(judgment.get("judge_task_id", "")).strip()
        if not judge_task_id or judge_task_id not in judge_task_by_id:
            continue
        judge_task = judge_task_by_id[judge_task_id]
        failure_rows.append(
            {
                "task_id": judge_task["task_id"],
                "form_name": judge_task["form_name"],
                "error": "judge_reject",
                "judge_label": judgment.get("label"),
                "judge_reason": judgment.get("reason"),
            }
        )

    retry_tasks = build_codex_spark_retry_tasks(
        task_rows=task_rows,
        candidate_rows=candidate_rows,
        failure_rows=failure_rows,
    )
    output_path = write_jsonl_rows(retry_tasks, args.output)
    summary = {
        "num_retry_tasks": len(retry_tasks),
        "judge_reject_count": len(failure_rows),
    }
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote retry tasks to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
