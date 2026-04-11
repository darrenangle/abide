#!/usr/bin/env python3
"""Validate codex-spark-authored warmup poems against Abide verifiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abide.training.codex_spark_warmup import (
    read_jsonl_rows,
    validate_codex_spark_candidates,
    write_jsonl_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_TASKS = "data/sft/codex_spark_hard_form_tasks.jsonl"
DEFAULT_CANDIDATES = "data/sft/codex_spark_hard_form_candidates.jsonl"
DEFAULT_OUTPUT = "data/sft/codex_spark_hard_form_validated.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--allow-degenerate", action="store_true")
    parser.add_argument("--max-seed-similarity", type=float, default=0.92)
    parser.add_argument("--max-peer-similarity", type=float, default=0.9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    task_rows = read_jsonl_rows(args.tasks)
    candidate_rows = read_jsonl_rows(args.candidates)
    validated_rows, summary = validate_codex_spark_candidates(
        candidate_rows,
        task_rows=task_rows,
        require_passed=not args.allow_failed,
        min_score=args.min_score,
        require_unique_per_form=not args.allow_duplicates,
        max_seed_similarity=args.max_seed_similarity,
        max_peer_similarity=args.max_peer_similarity,
        enforce_quality=not args.allow_degenerate,
    )
    output_path = write_jsonl_rows(validated_rows, args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote validated rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
