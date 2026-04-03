"""Integration tests for reward telemetry helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import reward_telemetry


def test_reward_telemetry_aggregates_and_logs_failure_summary() -> None:
    logged: list[dict[str, float]] = []
    telemetry = reward_telemetry.RewardTelemetry(
        label="unit",
        emit_every=2,
        use_wandb=False,
        log_fn=logged.append,
    )

    telemetry.record("Haiku", reward=1.0, passed=True)
    assert telemetry.emit() is None

    telemetry.record(
        "Haiku",
        reward=0.0,
        passed=False,
        failure_reason="missing syllables",
    )
    snapshot = telemetry.emit()

    assert snapshot is not None
    assert snapshot["window_count"] == 2
    assert snapshot["failure_reasons"] == [("missing syllables", 1)]
    assert snapshot["forms"]["Haiku"]["mean_reward"] == 0.5
    assert snapshot["forms"]["Haiku"]["pass_rate"] == 0.5
    assert snapshot["forms"]["Haiku"]["zero_rate"] == 0.5
    assert logged == [
        {
            "reward_telemetry/unit/window_count": 2.0,
            "reward_telemetry/unit/mean_reward": 0.5,
            "reward_telemetry/unit/pass_rate": 0.5,
            "reward_telemetry/unit/zero_rate": 0.5,
            "reward_telemetry/unit/failures/missing_syllables/count": 1.0,
            "reward_telemetry/unit/forms/Haiku/count": 2.0,
            "reward_telemetry/unit/forms/Haiku/mean_reward": 0.5,
            "reward_telemetry/unit/forms/Haiku/pass_rate": 0.5,
            "reward_telemetry/unit/forms/Haiku/zero_rate": 0.5,
        }
    ]


def test_extract_failure_reason_prefers_details_then_rubric() -> None:
    result = SimpleNamespace(
        passed=False,
        details={"issues": ["missing refrain"]},
        rubric=[SimpleNamespace(passed=False, criterion="refrain placement")],
    )

    assert reward_telemetry.extract_failure_reason(result) == "missing refrain"

    rubric_only = SimpleNamespace(
        passed=False,
        details={},
        rubric=[SimpleNamespace(passed=False, criterion="refrain placement")],
    )

    assert reward_telemetry.extract_failure_reason(rubric_only) == "refrain placement"


def test_extract_form_names_from_metadata_reads_form_name_and_info() -> None:
    assert reward_telemetry.extract_form_names_from_metadata(
        {"form_name": ["Villanelle"]},
        2,
    ) == ["Villanelle", None]
    assert reward_telemetry.extract_form_names_from_metadata(
        {"info": [{"form_name": "Ghazal"}]},
        2,
    ) == ["Ghazal", None]


def test_reward_telemetry_can_persist_jsonl_snapshots(tmp_path) -> None:
    path = tmp_path / "telemetry.jsonl"
    telemetry = reward_telemetry.RewardTelemetry(
        label="unit",
        emit_every=1,
        use_wandb=False,
        jsonl_path=path,
    )

    telemetry.record("Haiku", reward=0.25, passed=False, failure_reason="missing syllables")
    snapshot = telemetry.emit()

    assert snapshot is not None
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["label"] == "unit"
    assert payload["forms"]["Haiku"]["mean_reward"] == 0.25
