"""Integration tests for the curated warmup SFT corpus assembler."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import build_warmup_sft_corpus


def test_build_warmup_sft_corpus_cli_writes_summary(tmp_path) -> None:
    hard_form_validated = tmp_path / "hard_form_validated.jsonl"
    hard_form_validated.write_text(
        json.dumps(
            {
                "form_name": "Ghazal",
                "messages": [
                    {"role": "user", "content": "Write a ghazal."},
                    {"role": "assistant", "content": "Night gathers slow\nStreetlamps answer low"},
                ],
                "prompt": "Write a ghazal.",
                "response": "Night gathers slow\nStreetlamps answer low",
                "source_kind": "synthetic",
                "structural_brief": "Ghazal: 5-15 couplets with radif and qafiya",
                "verifier_passed": True,
                "verifier_score": 1.0,
            }
        )
        + "\n"
    )

    supplemental_path = tmp_path / "supplemental.jsonl"
    supplemental_path.write_text(
        json.dumps(
            {
                "form_name": "Haiku",
                "source_id": "haiku_harbor_watch_synth",
                "poem": "Harbor lamps awake\nRain beads on the mooring rope\nNight leans on the pier",
                "topic": "harbor lights after rain",
                "tone": "meditative",
                "source_kind": "synthetic",
            }
        )
        + "\n"
    )

    output_path = tmp_path / "warmup.jsonl"
    summary_path = tmp_path / "warmup.summary.json"

    exit_code = build_warmup_sft_corpus.main(
        [
            "--rl-default-prompt-variants",
            "1",
            "--abecedarian-prompt-variants",
            "1",
            "--hard-form-validated",
            str(hard_form_validated),
            "--supplemental-seeds",
            str(supplemental_path),
            "--supplemental-prompt-variants",
            "1",
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text())

    assert output_path.exists()
    assert summary_path.exists()
    assert any(row["source_dataset"] == "codex_spark_hard_form_validated" for row in rows)
    assert any(row["source_dataset"] == "supplemental_curated_seed_variants" for row in rows)
    assert "source_dataset_counts" in summary
    assert summary["source_dataset_counts"]["codex_spark_hard_form_validated"] == 1


def test_build_warmup_sft_corpus_can_write_shards(tmp_path) -> None:
    hard_form_validated = tmp_path / "hard_form_validated.jsonl"
    hard_form_validated.write_text("")

    output_path = tmp_path / "warmup.jsonl"
    summary_path = tmp_path / "warmup.summary.json"

    exit_code = build_warmup_sft_corpus.main(
        [
            "--rl-default-prompt-variants",
            "1",
            "--abecedarian-prompt-variants",
            "1",
            "--hard-form-validated",
            str(hard_form_validated),
            "--records-per-shard",
            "5",
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text())
    assert len(summary["shards"]) >= 3
    assert not output_path.exists()
    assert all(Path(path).exists() for path in summary["shards"])
