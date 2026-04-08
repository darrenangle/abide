#!/usr/bin/env python3
"""Run repeatable Gemma 4 E2B prime-rl sweeps and rank mixed-form runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Add src and scripts to path for development.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from train_prime_rl import find_final_summary

DEFAULT_RESULTS_PATH = "/tmp/abide_prime_rl_gemma4_e2b_sweep_results.jsonl"
DEFAULT_OUTPUT_ROOT = "/tmp/abide_prime_rl_gemma4_e2b_sweeps"
DEFAULT_PORT_BASE = 8150


@dataclass(frozen=True)
class SweepExperiment:
    name: str
    profile: str = "mixed-canary"
    form_set: str = "well_known"
    max_steps: int | None = None
    max_async_level: int | None = None
    num_prompts: int | None = None
    batch_size: int | None = None
    rollouts_per_example: int | None = None
    max_tokens: int | None = None
    seq_len: int | None = None
    learning_rate: float | None = None


def default_sweep_experiments() -> list[SweepExperiment]:
    """Small mixed-form sweep that is realistic enough to rank candidates quickly."""
    return [
        SweepExperiment("mixed_baseline_lr1e5", learning_rate=1e-5),
        SweepExperiment("mixed_lr5e6", learning_rate=5e-6),
        SweepExperiment("mixed_lr2e5", learning_rate=2e-5),
        SweepExperiment("mixed_seq1280_lr1e5", learning_rate=1e-5, seq_len=1280),
        SweepExperiment(
            "mixed_b2r2_lr5e6",
            learning_rate=5e-6,
            num_prompts=32,
            batch_size=2,
            rollouts_per_example=2,
            seq_len=1280,
        ),
    ]


def build_promoted_experiment(best: SweepExperiment) -> SweepExperiment:
    """Promote the best short sweep candidate into a longer real run."""
    return SweepExperiment(
        name=f"{best.name}_promoted",
        profile="mixed-canary",
        form_set="well_known",
        max_steps=8,
        max_async_level=0,
        num_prompts=64 if (best.batch_size or 1) > 1 else 48,
        batch_size=best.batch_size,
        rollouts_per_example=best.rollouts_per_example,
        max_tokens=max(best.max_tokens or 384, 384),
        seq_len=max(best.seq_len or 1024, 1280),
        learning_rate=best.learning_rate,
    )


def build_runner_env(
    experiment: SweepExperiment,
    *,
    output_root: Path,
    port: int,
    runtime_venv: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["ABIDE_RUN_PROFILE"] = experiment.profile
    env["ABIDE_FORM_SET"] = experiment.form_set
    env["ABIDE_OUTPUT_DIR"] = str((output_root / experiment.name).resolve())
    env["ABIDE_PORT"] = str(port)
    if runtime_venv is not None:
        env["ABIDE_PRIME_RL_VENV"] = runtime_venv

    numeric_overrides: tuple[tuple[str, int | float | None], ...] = (
        ("ABIDE_MAX_STEPS", experiment.max_steps),
        ("ABIDE_MAX_ASYNC_LEVEL", experiment.max_async_level),
        ("ABIDE_NUM_PROMPTS", experiment.num_prompts),
        ("ABIDE_BATCH_SIZE", experiment.batch_size),
        ("ABIDE_ROLLOUTS_PER_EXAMPLE", experiment.rollouts_per_example),
        ("ABIDE_MAX_TOKENS", experiment.max_tokens),
        ("ABIDE_SEQ_LEN", experiment.seq_len),
        ("ABIDE_LEARNING_RATE", experiment.learning_rate),
    )
    for key, value in numeric_overrides:
        if value is not None:
            env[key] = str(value)

    return env


def extract_summary_metrics(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text())
    reward = payload.get("metrics/abide-poetry-forms/abide_reward", payload.get("reward/all/mean"))
    pass_rate = payload.get("metrics/abide-poetry-forms/abide_pass", 0.0)
    truncated = payload.get("stop_condition/all/generation_truncated", 0)
    return {
        "summary_path": str(summary_path),
        "reward": float(reward) if reward is not None else None,
        "pass_rate": float(pass_rate) if pass_rate is not None else None,
        "generation_truncated": int(truncated),
        "step": int(payload.get("step", -1)),
        "total_samples": int(payload.get("progress/total_samples", 0)),
        "total_tokens": int(payload.get("progress/total_tokens", 0)),
        "seq_len_mean": float(payload.get("seq_len/all/mean", 0.0)),
        "decode_len_mean": float(payload.get("decode_len/all/mean", 0.0)),
        "update_weights_seconds": float(payload.get("time/update_weights", 0.0)),
        "wait_for_ckpt_seconds": float(payload.get("time/wait_for_ckpt", 0.0)),
    }


def result_objective(result: dict[str, Any]) -> float:
    if result.get("returncode") != 0 or result.get("reward") is None:
        return -1e9
    reward = float(result["reward"])
    pass_rate = float(result.get("pass_rate") or 0.0)
    truncated = int(result.get("generation_truncated") or 0)
    return reward + (0.25 * pass_rate) - (0.1 * truncated)


def result_rank(result: dict[str, Any]) -> tuple[float, float, float, int, int, float]:
    """Rank results by objective first, then prefer stronger evidence and lower overhead."""
    return (
        result_objective(result),
        float(result.get("pass_rate") or 0.0),
        float(result.get("reward") or -1e9),
        int(result.get("total_samples") or 0),
        -int(result.get("generation_truncated") or 0),
        -float(result.get("wait_for_ckpt_seconds") or 0.0),
    )


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        result
        for result in results
        if result.get("returncode") == 0 and result.get("reward") is not None
    ]
    if not completed:
        raise RuntimeError("No completed sweep runs produced summary metrics.")
    return max(completed, key=result_rank)


def append_result(results_path: Path, payload: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_experiment(
    experiment: SweepExperiment,
    *,
    output_root: Path,
    port: int,
    runtime_venv: str | None,
) -> dict[str, Any]:
    output_dir = (output_root / experiment.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "runner.log"
    env = build_runner_env(
        experiment, output_root=output_root, port=port, runtime_venv=runtime_venv
    )
    command = ["bash", "scripts/run_prime_rl_gemma4_e2b.sh", "--no-prepare-runtime", "--no-wandb"]

    print(f"\n=== {experiment.name} ===")
    print(f"profile={experiment.profile} form_set={experiment.form_set} port={port}")

    start = time.monotonic()
    with log_path.open("w") as log_handle:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - start

    summary_path = find_final_summary(output_dir)
    payload: dict[str, Any] = {
        "name": experiment.name,
        "port": port,
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "log_path": str(log_path),
        "experiment": asdict(experiment),
    }
    if summary_path is not None:
        payload.update(extract_summary_metrics(summary_path))

    reward_display = payload.get("reward")
    pass_display = payload.get("pass_rate")
    print(
        f"returncode={completed.returncode} "
        f"reward={reward_display} pass={pass_display} "
        f"summary={payload.get('summary_path', 'missing')}"
    )
    return payload


def run_sweep(
    experiments: list[SweepExperiment],
    *,
    output_root: Path,
    results_path: Path,
    port_base: int,
    runtime_venv: str | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments):
        result = run_experiment(
            experiment,
            output_root=output_root,
            port=port_base + index,
            runtime_venv=runtime_venv,
        )
        append_result(results_path, result)
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument("--runtime-venv")
    parser.add_argument("--skip-promoted", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    results_path = Path(args.results)

    sweep_results = run_sweep(
        default_sweep_experiments(),
        output_root=output_root,
        results_path=results_path,
        port_base=args.port_base,
        runtime_venv=args.runtime_venv,
    )
    best_result = select_best_result(sweep_results)
    best_experiment = SweepExperiment(**best_result["experiment"])

    print("\n=== best_sweep ===")
    print(json.dumps(best_result, indent=2, sort_keys=True))

    if args.skip_promoted:
        return 0

    promoted = build_promoted_experiment(best_experiment)
    promoted_result = run_experiment(
        promoted,
        output_root=output_root,
        port=args.port_base + len(sweep_results),
        runtime_venv=args.runtime_venv,
    )
    append_result(results_path, promoted_result)

    print("\n=== promoted_run ===")
    print(json.dumps(promoted_result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
