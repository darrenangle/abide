#!/usr/bin/env python3
"""Build curated supplemental seed poems from the full-surface codex artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_VALIDATED = "data/sft/codex_spark_full_surface_validated.jsonl"
DEFAULT_JUDGMENTS = "data/sft/codex_spark_full_surface_all_judgments.jsonl"
DEFAULT_RETRY_CURATED = "data/sft/codex_spark_full_surface_retry_curated_poemlike.jsonl"
DEFAULT_OUTPUT = "data/sft/full_surface_supplemental_seeds.jsonl"


def _read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    return [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]


def _write_jsonl_rows(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return output


def _first_line_topic(poem: str) -> str:
    for raw_line in poem.splitlines():
        line = raw_line.strip()
        if line:
            compact = re.sub(r"\s+", " ", line)
            compact = compact.rstrip(".,;:!?")
            if len(compact) > 80:
                compact = compact[:77].rstrip() + "..."
            return f'the image "{compact}"'
    return "a single vivid scene"


def _default_tone(form_name: str) -> str:
    if form_name in {"Clerihew", "Limerick"}:
        return "wry"
    if form_name in {"Rubai", "RhymeRoyal", "Ballad", "Ballade", "BroadBallad"}:
        return "reverent"
    if form_name in {"Skeltonic"}:
        return "playful"
    return "meditative"


def _seed_row_from_validated_row(
    row: dict[str, Any],
    *,
    source_id_suffix: str,
    source_dataset: str,
) -> dict[str, Any]:
    form_name = str(row["form_name"])
    response = str(row["response"]).strip()
    return {
        "form_name": form_name,
        "source_id": f"{row['task_id']}_{source_id_suffix}",
        "poem": response,
        "topic": _first_line_topic(response),
        "tone": _default_tone(form_name),
        "source_kind": "synthetic",
        "source_dataset": source_dataset,
        "task_id": str(row["task_id"]),
        "structural_brief": str(row.get("structural_brief", "")),
    }


def _merge_manual_seed_rows(
    seed_rows: dict[str, dict[str, Any]],
    manual_seed_rows: list[dict[str, Any]],
) -> None:
    for row in manual_seed_rows:
        form_name = str(row["form_name"])
        seed_rows[form_name] = row


def build_full_surface_supplemental_seed_rows(
    *,
    validated_rows: list[dict[str, Any]],
    judgment_rows: list[dict[str, Any]],
    retry_curated_rows: list[dict[str, Any]],
    manual_seed_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    validated_by_task_id = {str(row["task_id"]): row for row in validated_rows}
    seed_rows_by_form: dict[str, dict[str, Any]] = {}

    for judgment in judgment_rows:
        if not bool(judgment.get("judge_passed", False)):
            continue
        judge_task_id = str(judgment.get("judge_task_id", "")).strip()
        task_id = judge_task_id.removesuffix("--judge")
        validated_row = validated_by_task_id.get(task_id)
        if validated_row is None:
            continue
        form_name = str(validated_row["form_name"])
        seed_rows_by_form[form_name] = _seed_row_from_validated_row(
            validated_row,
            source_id_suffix="surface_judge_pass",
            source_dataset="full_surface_judge_accept",
        )

    for retry_row in retry_curated_rows:
        task_id = str(retry_row["task_id"])
        validated_row = validated_by_task_id.get(task_id)
        if validated_row is None:
            continue
        replacement_row = dict(validated_row)
        replacement_row["response"] = str(retry_row["response"])
        form_name = str(validated_row["form_name"])
        seed_rows_by_form[form_name] = _seed_row_from_validated_row(
            replacement_row,
            source_id_suffix="retry_curated",
            source_dataset="retry_curated_poemlike",
        )

    if manual_seed_rows:
        _merge_manual_seed_rows(seed_rows_by_form, manual_seed_rows)

    return sorted(seed_rows_by_form.values(), key=lambda row: str(row["form_name"]))


def summarize_seed_rows(
    seed_rows: list[dict[str, Any]], all_validated_forms: set[str]
) -> dict[str, Any]:
    form_counts = Counter(str(row["form_name"]) for row in seed_rows)
    dataset_counts = Counter(str(row.get("source_dataset", "unknown")) for row in seed_rows)
    covered_forms = set(form_counts)
    missing_forms = sorted(all_validated_forms - covered_forms)
    return {
        "num_seed_rows": len(seed_rows),
        "form_counts": dict(sorted(form_counts.items())),
        "source_dataset_counts": dict(sorted(dataset_counts.items())),
        "covered_form_count": len(covered_forms),
        "missing_form_count": len(missing_forms),
        "missing_forms": missing_forms,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated", default=DEFAULT_VALIDATED)
    parser.add_argument("--judgments", default=DEFAULT_JUDGMENTS)
    parser.add_argument("--retry-curated", default=DEFAULT_RETRY_CURATED)
    parser.add_argument("--manual-seeds", help="Optional JSONL of manual override seed rows.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    validated_rows = _read_jsonl_rows(args.validated)
    judgment_rows = _read_jsonl_rows(args.judgments)
    retry_curated_rows = _read_jsonl_rows(args.retry_curated)
    manual_seed_rows = _read_jsonl_rows(args.manual_seeds) if args.manual_seeds else []

    seed_rows = build_full_surface_supplemental_seed_rows(
        validated_rows=validated_rows,
        judgment_rows=judgment_rows,
        retry_curated_rows=retry_curated_rows,
        manual_seed_rows=manual_seed_rows,
    )
    output_path = _write_jsonl_rows(seed_rows, args.output)
    summary = summarize_seed_rows(
        seed_rows,
        all_validated_forms={str(row["form_name"]) for row in validated_rows},
    )
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote seed rows to {output_path}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
