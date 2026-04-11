#!/usr/bin/env python3
"""Merge and summarize JSONL poemness-judge outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abide.training.poemness_judge import (
    read_jsonl_rows,
    summarize_poemness_judgments,
    write_jsonl_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_TASKS = "data/sft/codex_spark_full_surface_judge_tasks.jsonl"
DEFAULT_OUTPUT = "data/sft/codex_spark_full_surface_judgments.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    tasks = read_jsonl_rows(args.tasks)
    judgments: list[dict[str, object]] = []
    for raw_path in args.inputs:
        path = Path(raw_path)
        if not path.exists():
            continue
        judgments.extend(read_jsonl_rows(path))
    judgments.sort(key=lambda row: str(row.get("judge_task_id", "")))

    output_path = write_jsonl_rows(judgments, args.output)
    summary = summarize_poemness_judgments(tasks, judgments)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote merged judgments to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
