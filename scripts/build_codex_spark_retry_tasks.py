#!/usr/bin/env python3
"""Build retry tasks for rejected codex-authored warmup drafts."""

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


DEFAULT_TASKS = "data/sft/codex_spark_hard_form_tasks.jsonl"
DEFAULT_CANDIDATES = "data/sft/codex_spark_hard_form_candidates.jsonl"
DEFAULT_SUMMARY = "data/sft/codex_spark_hard_form_validated.summary.json"
DEFAULT_OUTPUT = "data/sft/codex_spark_hard_form_retry_tasks.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    task_rows = read_jsonl_rows(args.tasks)
    candidate_rows = read_jsonl_rows(args.candidates)
    summary = json.loads(Path(args.summary).read_text())
    failure_rows = list(summary.get("failures", []))

    retry_tasks = build_codex_spark_retry_tasks(
        task_rows=task_rows,
        candidate_rows=candidate_rows,
        failure_rows=failure_rows,
    )
    output_path = write_jsonl_rows(retry_tasks, args.output)
    print(
        json.dumps(
            {
                "num_retry_tasks": len(retry_tasks),
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
