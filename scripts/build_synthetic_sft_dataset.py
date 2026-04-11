#!/usr/bin/env python3
"""Export verifier-gated Abide warmup records for chat SFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abide.training.synthetic_sft import (
    build_synthetic_sft_records,
    summarize_synthetic_sft_records,
    write_synthetic_sft_jsonl,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_OUTPUT = "data/sft/abide_synthetic_rl_default.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form-set", default="rl_default", choices=("well_known", "rl_default"))
    parser.add_argument("--form-names", help="Optional comma-separated explicit form names.")
    parser.add_argument("--prompt-variants-per-seed", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--public-domain-only", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.synthetic_only and args.public_domain_only:
        parser.error("--synthetic-only and --public-domain-only are mutually exclusive.")

    records = build_synthetic_sft_records(
        form_set=args.form_set,
        form_names=args.form_names,
        prompt_variants_per_seed=args.prompt_variants_per_seed,
        seed=args.seed,
        include_synthetic=not args.public_domain_only,
        include_public_domain=not args.synthetic_only,
        require_passed=not args.allow_failed,
        min_score=args.min_score,
    )
    summary = summarize_synthetic_sft_records(records)
    output_path = write_synthetic_sft_jsonl(records, args.output)

    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {summary['num_records']} records to {output_path}")
    print(f"Summary written to {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
