"""Tests for the Gemma 4 E2B prime-rl validation harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from scripts import prime_rl_validate

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_seeds_requires_multi_seed_validation() -> None:
    assert prime_rl_validate.parse_seeds("41,42,43") == (41, 42, 43)


def test_build_training_env_sets_seed_and_runtime(tmp_path: Path) -> None:
    run = prime_rl_validate.ValidationRun(
        seed=77,
        output_dir=tmp_path / "seed_77",
        port=8181,
        eval_port=8281,
    )

    env = prime_rl_validate.build_training_env(run, runtime_venv=".venv-prime-rl")

    assert env["ABIDE_RUN_PROFILE"] == "mixed-stable"
    assert env["ABIDE_FORM_SET"] == "well_known"
    assert env["ABIDE_OUTPUT_DIR"] == str((tmp_path / "seed_77").resolve())
    assert env["ABIDE_PORT"] == "8181"
    assert env["ABIDE_SEED"] == "77"
    assert env["ABIDE_PRIME_RL_VENV"] == ".venv-prime-rl"


def test_build_eval_model_dir_overlays_broadcast_weights(tmp_path: Path) -> None:
    base_model_dir = tmp_path / "base"
    base_model_dir.mkdir()
    (base_model_dir / "config.json").write_text("{}")
    (base_model_dir / "tokenizer.json").write_text("{}")
    (base_model_dir / "model.safetensors").write_text("base")

    broadcast_dir = tmp_path / "broadcast"
    broadcast_dir.mkdir()
    (broadcast_dir / "pytorch_model-00001-of-00002.bin").write_text("shard-a")
    (broadcast_dir / "pytorch_model.bin.index.json").write_text("{}")
    (broadcast_dir / "STABLE").write_text("")

    target_dir = tmp_path / "eval"
    prime_rl_validate.build_eval_model_dir(
        base_model_dir=base_model_dir,
        broadcast_dir=broadcast_dir,
        target_dir=target_dir,
    )

    assert (target_dir / "config.json").is_symlink()
    assert (target_dir / "tokenizer.json").is_symlink()
    assert not (target_dir / "model.safetensors").exists()
    assert (target_dir / "pytorch_model-00001-of-00002.bin").is_symlink()
    assert (target_dir / "pytorch_model.bin.index.json").is_symlink()


def test_summarize_holdout_samples_aggregates_per_form() -> None:
    samples = [
        {"form_name": "Haiku", "verification": {"score": 1.0, "passed": True}},
        {"form_name": "Haiku", "verification": {"score": 0.5, "passed": False}},
        {"form_name": "Villanelle", "verification": {"score": 0.25, "passed": False}},
    ]

    summary = prime_rl_validate.summarize_holdout_samples(samples)

    assert summary["sample_count"] == 3
    assert summary["mean_score"] == 0.5833333333333334
    assert summary["pass_rate"] == 1 / 3
    assert summary["per_form"]["Haiku"]["sample_count"] == 2
    assert summary["per_form"]["Haiku"]["mean_score"] == 0.75
    assert summary["per_form"]["Haiku"]["pass_rate"] == 0.5
    assert summary["per_form"]["Villanelle"]["mean_score"] == 0.25


def test_summarize_validation_runs_averages_completed_holdout_results() -> None:
    results = [
        {
            "seed": 41,
            "returncode": 0,
            "reward": 0.9,
            "holdout_summary": {
                "mean_score": 0.8,
                "pass_rate": 0.5,
                "per_form": {"Haiku": {"mean_score": 0.8, "pass_rate": 0.5}},
            },
        },
        {
            "seed": 42,
            "returncode": 0,
            "reward": 1.0,
            "holdout_summary": {
                "mean_score": 0.9,
                "pass_rate": 0.75,
                "per_form": {"Haiku": {"mean_score": 0.9, "pass_rate": 0.75}},
            },
        },
        {
            "seed": 43,
            "returncode": 1,
            "reward": None,
            "holdout_summary": None,
        },
    ]

    summary = prime_rl_validate.summarize_validation_runs(results)

    assert summary["completed_runs"] == 2
    assert summary["requested_runs"] == 3
    assert summary["mean_training_reward"] == pytest.approx(0.95)
    assert summary["mean_holdout_score"] == pytest.approx(0.85)
    assert summary["mean_holdout_pass_rate"] == pytest.approx(0.625)
    assert summary["per_form"]["Haiku"]["runs"] == 2
    assert summary["per_form"]["Haiku"]["mean_score"] == pytest.approx(0.85)


def test_load_prime_rl_run_config_reads_written_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "prime_rl" / "rl.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "[model]",
                'name = "/tmp/gemma"',
                "",
                "[orchestrator.sampling]",
                "max_tokens = 384",
                "",
                "[inference.model]",
                "max_model_len = 1280",
            ]
        )
    )

    config = prime_rl_validate.load_prime_rl_run_config(tmp_path)

    assert config["model"]["name"] == "/tmp/gemma"
    assert config["orchestrator"]["sampling"]["max_tokens"] == 384
    assert config["inference"]["model"]["max_model_len"] == 1280
