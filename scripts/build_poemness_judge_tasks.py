#!/usr/bin/env python3
"""Build JSONL judge tasks for a small-model poemness audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abide.training.poemness_judge import (
    build_poemness_judge_tasks,
    read_jsonl_rows,
    write_jsonl_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_VALIDATED = "data/sft/codex_spark_full_surface_validated.jsonl"
DEFAULT_RETRY_SUMMARY = "data/sft/codex_spark_full_surface_quality_validated.summary.json"
DEFAULT_OUTPUT = "data/sft/codex_spark_full_surface_judge_tasks.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated", default=DEFAULT_VALIDATED)
    parser.add_argument("--retry-summary", default=DEFAULT_RETRY_SUMMARY)
    parser.add_argument("--accepted-sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validated_rows = read_jsonl_rows(args.validated)
    retry_summary = json.loads(Path(args.retry_summary).read_text())
    tasks = build_poemness_judge_tasks(
        validated_rows=validated_rows,
        retry_summary=retry_summary,
        accepted_sample_size=args.accepted_sample_size,
        seed=args.seed,
    )
    output_path = write_jsonl_rows(tasks, args.output)
    summary = {
        "num_tasks": len(tasks),
        "accepted_control_count": sum(1 for task in tasks if task["source"] == "accepted_control"),
        "heuristic_reject_count": sum(1 for task in tasks if task["source"] == "heuristic_reject"),
    }
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote tasks to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
