"""Tests for the Gemma 4 E2B prime-rl sweep harness."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts import prime_rl_sweep

if TYPE_CHECKING:
    from pathlib import Path


def test_default_sweep_experiments_target_mixed_prime_rl_runs() -> None:
    experiments = prime_rl_sweep.default_sweep_experiments()

    assert [experiment.name for experiment in experiments] == [
        "mixed_baseline_lr1e5",
        "mixed_lr5e6",
        "mixed_lr2e5",
        "mixed_seq1280_lr1e5",
        "mixed_b2r2_lr5e6",
    ]
    assert all(experiment.profile == "mixed-canary" for experiment in experiments)
    assert all(experiment.form_set == "well_known" for experiment in experiments)


def test_build_runner_env_sets_profile_output_and_overrides(tmp_path: Path) -> None:
    experiment = prime_rl_sweep.SweepExperiment(
        name="mixed_test",
        max_steps=6,
        num_prompts=40,
        batch_size=2,
        rollouts_per_example=2,
        learning_rate=5e-6,
    )

    env = prime_rl_sweep.build_runner_env(
        experiment,
        output_root=tmp_path,
        port=8171,
        runtime_venv=".venv-prime-rl",
    )

    assert env["ABIDE_RUN_PROFILE"] == "mixed-canary"
    assert env["ABIDE_FORM_SET"] == "well_known"
    assert env["ABIDE_OUTPUT_DIR"] == str((tmp_path / "mixed_test").resolve())
    assert env["ABIDE_PORT"] == "8171"
    assert env["ABIDE_PRIME_RL_VENV"] == ".venv-prime-rl"
    assert env["ABIDE_MAX_STEPS"] == "6"
    assert env["ABIDE_NUM_PROMPTS"] == "40"
    assert env["ABIDE_BATCH_SIZE"] == "2"
    assert env["ABIDE_ROLLOUTS_PER_EXAMPLE"] == "2"
    assert env["ABIDE_LEARNING_RATE"] == "5e-06"


def test_extract_summary_metrics_reads_reward_and_pass_fields(tmp_path: Path) -> None:
    summary = tmp_path / "final_summary.json"
    summary.write_text(
        json.dumps(
            {
                "metrics/abide-poetry-forms/abide_reward": 0.625,
                "metrics/abide-poetry-forms/abide_pass": 0.25,
                "stop_condition/all/generation_truncated": 0,
                "step": 7,
                "progress/total_samples": 32,
                "progress/total_tokens": 4096,
                "seq_len/all/mean": 212,
                "decode_len/all/mean": 118,
                "time/update_weights": 11.2,
                "time/wait_for_ckpt": 22.5,
            }
        )
    )

    metrics = prime_rl_sweep.extract_summary_metrics(summary)

    assert metrics == {
        "summary_path": str(summary),
        "reward": 0.625,
        "pass_rate": 0.25,
        "generation_truncated": 0,
        "step": 7,
        "total_samples": 32,
        "total_tokens": 4096,
        "seq_len_mean": 212.0,
        "decode_len_mean": 118.0,
        "update_weights_seconds": 11.2,
        "wait_for_ckpt_seconds": 22.5,
    }


def test_select_best_result_prefers_reward_and_pass_over_lower_objective() -> None:
    worse = {
        "name": "worse",
        "returncode": 0,
        "reward": 0.58,
        "pass_rate": 0.0,
        "generation_truncated": 0,
    }
    better = {
        "name": "better",
        "returncode": 0,
        "reward": 0.57,
        "pass_rate": 0.5,
        "generation_truncated": 0,
    }

    best = prime_rl_sweep.select_best_result([worse, better])

    assert best["name"] == "better"


def test_select_best_result_prefers_more_samples_when_objective_ties() -> None:
    smaller = {
        "name": "smaller",
        "returncode": 0,
        "reward": 1.0,
        "pass_rate": 1.0,
        "generation_truncated": 0,
        "total_samples": 4,
        "wait_for_ckpt_seconds": 40.0,
    }
    larger = {
        "name": "larger",
        "returncode": 0,
        "reward": 1.0,
        "pass_rate": 1.0,
        "generation_truncated": 0,
        "total_samples": 8,
        "wait_for_ckpt_seconds": 37.0,
    }

    best = prime_rl_sweep.select_best_result([smaller, larger])

    assert best["name"] == "larger"


def test_build_promoted_experiment_extends_best_candidate() -> None:
    experiment = prime_rl_sweep.SweepExperiment(
        name="mixed_lr5e6",
        learning_rate=5e-6,
        batch_size=2,
        rollouts_per_example=2,
        seq_len=1280,
    )

    promoted = prime_rl_sweep.build_promoted_experiment(experiment)

    assert promoted.name == "mixed_lr5e6_promoted"
    assert promoted.form_set == "well_known"
    assert promoted.max_steps == 8
    assert promoted.max_async_level == 0
    assert promoted.num_prompts == 64
    assert promoted.batch_size == 2
    assert promoted.rollouts_per_example == 2
    assert promoted.max_tokens == 384
    assert promoted.seq_len == 1280
    assert promoted.learning_rate == 5e-6
