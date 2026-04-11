#!/usr/bin/env python3
"""Merge per-form codex-spark warmup candidate shards into one JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_INPUTS = (
    "data/sft/codex_spark_tanka_candidates.jsonl",
    "data/sft/codex_spark_petrarchansonnet_candidates.jsonl",
    "data/sft/codex_spark_villanelle_candidates.jsonl",
    "data/sft/codex_spark_ghazal_candidates.jsonl",
    "data/sft/codex_spark_sestina_candidates.jsonl",
)
DEFAULT_OUTPUT = "data/sft/codex_spark_hard_form_candidates.jsonl"


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=list(DEFAULT_INPUTS))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows: list[dict[str, object]] = []
    for raw_path in args.inputs:
        path = Path(raw_path)
        if not path.exists():
            continue
        rows.extend(read_rows(path))

    rows.sort(key=lambda row: str(row["task_id"]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    print(json.dumps({"num_rows": len(rows), "output": str(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
