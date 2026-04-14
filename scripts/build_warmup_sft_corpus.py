#!/usr/bin/env python3
"""Assemble a curated warmup SFT corpus from stable Abide seed sources."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abide.training.synthetic_sft import (
    SeedPoem,
    SourceKind,
    build_synthetic_sft_records,
    build_synthetic_sft_records_from_seed_poems,
    summarize_synthetic_sft_records,
    write_synthetic_sft_jsonl,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_OUTPUT = "data/sft/abide_warmup_curated.jsonl"
DEFAULT_HARD_FORM_VALIDATED = "data/sft/codex_spark_hard_form_validated.jsonl"


def _read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    return [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]


def _load_supplemental_seed_poems(path: str | Path) -> list[SeedPoem]:
    rows = _read_jsonl_rows(path)
    seed_poems: list[SeedPoem] = []
    for row in rows:
        source_kind = str(row.get("source_kind", "synthetic"))
        if source_kind not in {"synthetic", "public_domain"}:
            raise ValueError(f"Unsupported source_kind in supplemental seed row: {source_kind}")
        seed_poems.append(
            SeedPoem(
                form_name=str(row["form_name"]),
                source_id=str(row["source_id"]),
                poem=str(row["poem"]),
                topic=str(row["topic"]),
                tone=str(row["tone"]),
                source_kind=cast("SourceKind", source_kind),
            )
        )
    return seed_poems


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_keys: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        key = (
            str(record.get("form_name", "")),
            str(record.get("prompt", "")),
            str(record.get("response", "")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(record)
    return deduped


def _write_sharded_jsonl(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    records_per_shard: int,
) -> list[Path]:
    if records_per_shard < 1:
        raise ValueError("records_per_shard must be at least 1.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.stem
    suffix = output.suffix or ".jsonl"
    shard_paths: list[Path] = []

    for start in range(0, len(records), records_per_shard):
        shard_records = records[start : start + records_per_shard]
        shard_index = start // records_per_shard + 1
        shard_path = output.with_name(f"{stem}-{shard_index:04d}{suffix}")
        write_synthetic_sft_jsonl(shard_records, shard_path)
        shard_paths.append(shard_path)

    return shard_paths


def _summarize_assembled_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    base_summary = summarize_synthetic_sft_records(records)
    source_dataset_counts = Counter(
        str(record.get("source_dataset", "unknown")) for record in records
    )
    return {
        **base_summary,
        "source_dataset_counts": dict(sorted(source_dataset_counts.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary")
    parser.add_argument(
        "--records-per-shard",
        type=int,
        help="Optional maximum records per emitted JSONL shard.",
    )
    parser.add_argument(
        "--rl-default-prompt-variants",
        type=int,
        default=8,
        help="Prompt variants per built-in rl_default seed poem.",
    )
    parser.add_argument(
        "--abecedarian-prompt-variants",
        type=int,
        default=6,
        help="Prompt variants per curated Abecedarian seed poem.",
    )
    parser.add_argument(
        "--hard-form-validated",
        default=DEFAULT_HARD_FORM_VALIDATED,
        help="Path to the validated hard-form codex shard.",
    )
    parser.add_argument(
        "--supplemental-seeds",
        help="Optional JSONL of curated supplemental seed poems.",
    )
    parser.add_argument(
        "--supplemental-prompt-variants",
        type=int,
        default=8,
        help="Prompt variants per supplemental seed poem.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    assembled_records: list[dict[str, Any]] = []

    rl_default_records = build_synthetic_sft_records(
        form_set="rl_default",
        prompt_variants_per_seed=args.rl_default_prompt_variants,
        seed=42,
    )
    for record in rl_default_records:
        record["source_dataset"] = "rl_default_seed_variants"
    assembled_records.extend(rl_default_records)

    abecedarian_records = build_synthetic_sft_records(
        form_names="Abecedarian",
        prompt_variants_per_seed=args.abecedarian_prompt_variants,
        seed=42,
        include_public_domain=False,
    )
    for record in abecedarian_records:
        record["source_dataset"] = "abecedarian_curated_seed_variants"
    assembled_records.extend(abecedarian_records)

    hard_form_rows = _read_jsonl_rows(args.hard_form_validated)
    for row in hard_form_rows:
        row.setdefault("source_kind", "synthetic")
        row["source_dataset"] = "codex_spark_hard_form_validated"
    assembled_records.extend(hard_form_rows)

    if args.supplemental_seeds:
        supplemental_seed_poems = _load_supplemental_seed_poems(args.supplemental_seeds)
        supplemental_records = build_synthetic_sft_records_from_seed_poems(
            seed_poems=supplemental_seed_poems,
            prompt_variants_per_seed=args.supplemental_prompt_variants,
            seed=42,
        )
        for record in supplemental_records:
            record["source_dataset"] = "supplemental_curated_seed_variants"
        assembled_records.extend(supplemental_records)

    deduped_records = _dedupe_records(assembled_records)
    summary = _summarize_assembled_records(deduped_records)
    if args.records_per_shard:
        shard_paths = _write_sharded_jsonl(
            deduped_records,
            args.output,
            records_per_shard=args.records_per_shard,
        )
        summary["shards"] = [str(path) for path in shard_paths]
        output_label = ", ".join(str(path) for path in shard_paths)
    else:
        output_path = write_synthetic_sft_jsonl(deduped_records, args.output)
        summary["shards"] = [str(output_path)]
        output_label = str(output_path)

    summary_path = (
        Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {summary['num_records']} records to {output_label}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
