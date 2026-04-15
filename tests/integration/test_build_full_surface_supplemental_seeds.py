"""Integration tests for building full-surface supplemental seed rows."""

from __future__ import annotations

import json

from scripts import build_full_surface_supplemental_seeds


def test_build_full_surface_supplemental_seed_rows_prefers_retries_and_manual_rows() -> None:
    validated_rows = [
        {
            "task_id": "haiku-001",
            "form_name": "Haiku",
            "response": "Cold rain taps the steps\nA blue cup cools beside the door\nDawn gathers in steam",
            "structural_brief": "Haiku: 3 lines with 5-7-5 syllable pattern",
        },
        {
            "task_id": "rondeau-001",
            "form_name": "Rondeau",
            "response": "Old raw response",
            "structural_brief": "Rondeau: refrain form",
        },
    ]
    judgment_rows = [
        {"judge_task_id": "haiku-001--judge", "judge_passed": True},
        {"judge_task_id": "rondeau-001--judge", "judge_passed": False},
    ]
    retry_curated_rows = [{"task_id": "rondeau-001", "response": "New retry response"}]
    manual_seed_rows = [
        {
            "form_name": "Haiku",
            "source_id": "haiku_manual_seed",
            "poem": "Manual seed poem",
            "topic": "manual topic",
            "tone": "manual tone",
            "source_kind": "synthetic",
            "source_dataset": "manual_override",
        }
    ]

    seed_rows = build_full_surface_supplemental_seeds.build_full_surface_supplemental_seed_rows(
        validated_rows=validated_rows,
        judgment_rows=judgment_rows,
        retry_curated_rows=retry_curated_rows,
        manual_seed_rows=manual_seed_rows,
    )

    assert {row["form_name"] for row in seed_rows} == {"Haiku", "Rondeau"}
    haiku_row = next(row for row in seed_rows if row["form_name"] == "Haiku")
    rondeau_row = next(row for row in seed_rows if row["form_name"] == "Rondeau")
    assert haiku_row["source_dataset"] == "manual_override"
    assert rondeau_row["source_dataset"] == "retry_curated_poemlike"
    assert rondeau_row["poem"] == "New retry response"


def test_build_full_surface_supplemental_seeds_cli_writes_summary(tmp_path) -> None:
    validated_path = tmp_path / "validated.jsonl"
    judgments_path = tmp_path / "judgments.jsonl"
    retry_path = tmp_path / "retry.jsonl"
    output_path = tmp_path / "seeds.jsonl"
    summary_path = tmp_path / "seeds.summary.json"

    validated_path.write_text(
        json.dumps(
            {
                "task_id": "haiku-001",
                "form_name": "Haiku",
                "response": "Cold rain taps the steps\nA blue cup cools beside the door\nDawn gathers in steam",
                "structural_brief": "Haiku: 3 lines with 5-7-5 syllable pattern",
            }
        )
        + "\n"
    )
    judgments_path.write_text(
        json.dumps({"judge_task_id": "haiku-001--judge", "judge_passed": True}) + "\n"
    )
    retry_path.write_text("")

    exit_code = build_full_surface_supplemental_seeds.main(
        [
            "--validated",
            str(validated_path),
            "--judgments",
            str(judgments_path),
            "--retry-curated",
            str(retry_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text())
    assert len(rows) == 1
    assert rows[0]["form_name"] == "Haiku"
    assert summary["covered_form_count"] == 1
    assert summary["missing_form_count"] == 0
