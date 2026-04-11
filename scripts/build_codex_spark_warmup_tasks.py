#!/usr/bin/env python3
"""Build task sheets for codex-spark-authored warmup poems."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abide.training.codex_spark_warmup import (
    HARD_FORM_NAMES,
    build_codex_spark_warmup_tasks,
    write_codex_spark_warmup_tasks,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_OUTPUT = "data/sft/codex_spark_hard_form_tasks.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form-names", default=",".join(HARD_FORM_NAMES))
    parser.add_argument("--tasks-per-form", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tasks = build_codex_spark_warmup_tasks(
        form_names=args.form_names,
        tasks_per_form=args.tasks_per_form,
        seed=args.seed,
    )
    output_path = write_codex_spark_warmup_tasks(tasks, args.output)
    summary = {
        "num_tasks": len(tasks),
        "form_names": sorted({task["form_name"] for task in tasks}),
        "tasks_per_form": args.tasks_per_form,
    }
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote tasks to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
