#!/usr/bin/env python3
"""Run multi-seed Gemma 4 E2B validation with deterministic holdout scoring."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:
    import tomli as tomllib

import httpx

# Add src and scripts to path for development.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from prime_rl_sweep import extract_summary_metrics
from train_prime_rl import build_prime_rl_env, find_final_summary, prepare_runtime

from abide.training.prime_rl_env import (
    build_prime_rl_prompt_records,
    normalize_generated_poem,
    resolve_prime_rl_form_instances,
)

DEFAULT_OUTPUT_ROOT = "models/prime_rl_gemma4_e2b_validation"
DEFAULT_RESULTS_PATH = f"{DEFAULT_OUTPUT_ROOT}/results.jsonl"
DEFAULT_SUMMARY_PATH = f"{DEFAULT_OUTPUT_ROOT}/summary.json"
DEFAULT_RUNTIME_VENV = ".venv-prime-rl"
DEFAULT_PORT_BASE = 8180
DEFAULT_HOLDOUT_SEED = 424242
DEFAULT_HOLDOUT_PER_FORM = 2
DEFAULT_EVAL_TIMEOUT_SECONDS = 300.0
DEFAULT_EVAL_GPU_MEMORY_UTILIZATION = 0.65
_WEIGHT_FILE_PATTERNS = (
    "model*.safetensors",
    "pytorch_model*.bin",
    "*.safetensors.index.json",
    "*.bin.index.json",
)


@dataclass(frozen=True)
class ValidationRun:
    seed: int
    output_dir: Path
    port: int
    eval_port: int
    profile: str = "mixed-stable"
    form_set: str = "well_known"


def parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())
    if len(seeds) < 2:
        raise ValueError("Validation requires at least two seeds.")
    return seeds


def _matches_weight_pattern(name: str) -> bool:
    path = Path(name)
    return any(path.match(pattern) for pattern in _WEIGHT_FILE_PATTERNS)


def append_result(results_path: Path, payload: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def build_training_env(
    run: ValidationRun,
    *,
    runtime_venv: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["ABIDE_RUN_PROFILE"] = run.profile
    env["ABIDE_FORM_SET"] = run.form_set
    env["ABIDE_OUTPUT_DIR"] = str(run.output_dir.resolve())
    env["ABIDE_PORT"] = str(run.port)
    env["ABIDE_SEED"] = str(run.seed)
    if runtime_venv is not None:
        env["ABIDE_PRIME_RL_VENV"] = runtime_venv
    return env


def find_latest_broadcast_dir(output_dir: Path) -> Path:
    broadcast_root = output_dir / "run_default" / "broadcasts"
    candidates = sorted(
        broadcast_root.glob("step_*"),
        key=lambda path: int(path.name.split("_", 1)[1]),
    )
    stable_candidates = [candidate for candidate in candidates if (candidate / "STABLE").exists()]
    if stable_candidates:
        return stable_candidates[-1]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No broadcast checkpoints found under {broadcast_root}")


def load_prime_rl_run_config(output_dir: Path) -> dict[str, Any]:
    config_path = output_dir / "prime_rl" / "rl.toml"
    return cast("dict[str, Any]", tomllib.loads(config_path.read_text()))


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_eval_model_dir(
    *,
    base_model_dir: Path,
    broadcast_dir: Path,
    target_dir: Path,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)

    for child in base_model_dir.iterdir():
        if _matches_weight_pattern(child.name):
            continue
        target = target_dir / child.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.symlink_to(child.resolve())

    for child in broadcast_dir.iterdir():
        if child.name in {"NCCL_READY", "STABLE"}:
            continue
        target = target_dir / child.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.symlink_to(child.resolve())

    return target_dir


def pick_eval_gpu() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, int]] = []
    for line in completed.stdout.splitlines():
        index_text, free_text = line.strip().split(", ")
        rows.append((int(index_text), int(free_text)))
    rows.sort(key=lambda item: item[1], reverse=True)
    if not rows:
        raise RuntimeError("No visible GPUs available for evaluation.")
    return str(rows[0][0])


def tail_text(path: Path, *, lines: int = 40) -> str:
    if not path.exists():
        return ""
    content = path.read_text().splitlines()
    return "\n".join(content[-lines:])


def wait_for_server_ready(
    proc: subprocess.Popen[str],
    *,
    port: int,
    log_path: Path,
    timeout_seconds: float = DEFAULT_EVAL_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    base_url = f"http://127.0.0.1:{port}"

    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log_tail = tail_text(log_path)
                raise RuntimeError(
                    f"Evaluation server exited early with code {proc.returncode}.\n{log_tail}"
                )
            try:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)

    log_tail = tail_text(log_path)
    raise TimeoutError(f"Timed out waiting for {base_url}/health.\n{log_tail}")


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=20.0)


def launch_eval_server(
    *,
    model_dir: Path,
    port: int,
    runtime_venv: str,
    max_model_len: int,
    log_path: Path,
) -> tuple[subprocess.Popen[str], Any]:
    env = build_prime_rl_env(runtime_venv, use_local_model=True)
    env["CUDA_VISIBLE_DEVICES"] = pick_eval_gpu()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    command = [
        str(Path(runtime_venv) / "bin" / "vllm"),
        "serve",
        str(model_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "abide-eval",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(DEFAULT_EVAL_GPU_MEMORY_UTILIZATION),
        "--trust-remote-code",
        "--enforce-eager",
        "--disable-log-stats",
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w")
    try:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    return proc, log_handle


def score_holdout_samples(
    *,
    output_dir: Path,
    form_set: str,
    runtime_venv: str,
    holdout_seed: int,
    holdout_per_form: int,
    port: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_config = load_prime_rl_run_config(output_dir)
    model_name = run_config["model"]["name"]
    max_tokens = int(run_config["orchestrator"]["sampling"]["max_tokens"])
    max_model_len = int(run_config["inference"]["model"]["max_model_len"])
    forms = resolve_prime_rl_form_instances(form_set=form_set)
    prompt_records = build_prime_rl_prompt_records(
        num_prompts=holdout_per_form * len(forms),
        seed=holdout_seed,
        form_set=form_set,
    )

    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    samples_path = validation_dir / "holdout_samples.jsonl"
    broadcast_dir = find_latest_broadcast_dir(output_dir)

    with tempfile.TemporaryDirectory(prefix="eval_model_", dir=validation_dir) as temp_dir:
        eval_model_dir = build_eval_model_dir(
            base_model_dir=Path(model_name),
            broadcast_dir=broadcast_dir,
            target_dir=Path(temp_dir),
        )
        log_path = validation_dir / "eval_server.log"
        proc, log_handle = launch_eval_server(
            model_dir=eval_model_dir,
            port=port,
            runtime_venv=runtime_venv,
            max_model_len=max_model_len,
            log_path=log_path,
        )
        try:
            wait_for_server_ready(proc, port=port, log_path=log_path)
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120.0)
            try:
                samples: list[dict[str, Any]] = []
                with samples_path.open("w") as handle:
                    for prompt_index, record in enumerate(prompt_records):
                        form_name = str(record["info"]["form_name"])
                        response = client.post(
                            "/v1/chat/completions",
                            json={
                                "model": "abide-eval",
                                "messages": record["prompt"],
                                "temperature": 0.0,
                                "max_tokens": max_tokens,
                            },
                        )
                        response.raise_for_status()
                        payload = response.json()
                        message = payload["choices"][0]["message"]["content"]
                        generated_text = message if isinstance(message, str) else str(message)
                        poem = normalize_generated_poem(generated_text)
                        verification = forms[form_name].verify(poem)
                        sample = {
                            "prompt_index": prompt_index,
                            "form_name": form_name,
                            "prompt": record["prompt"],
                            "generated_text": generated_text,
                            "normalized_poem": poem,
                            "finish_reason": payload["choices"][0].get("finish_reason"),
                            "usage": payload.get("usage", {}),
                            "verification": verification.to_dict(),
                        }
                        handle.write(json.dumps(sample, sort_keys=True) + "\n")
                        samples.append(sample)
            finally:
                client.close()
        finally:
            stop_server(proc)
            log_handle.close()

    summary = summarize_holdout_samples(samples)
    summary["samples_path"] = str(samples_path)
    summary["broadcast_dir"] = str(broadcast_dir)
    return samples, summary


def summarize_holdout_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0, "mean_score": 0.0, "pass_rate": 0.0, "per_form": {}}

    per_form: dict[str, dict[str, float | int]] = {}
    total_score = 0.0
    total_passes = 0
    for sample in samples:
        form_name = str(sample["form_name"])
        verification = sample["verification"]
        score = float(verification["score"])
        passed = bool(verification["passed"])
        total_score += score
        total_passes += 1 if passed else 0
        bucket = per_form.setdefault(
            form_name,
            {"sample_count": 0, "mean_score": 0.0, "pass_rate": 0.0},
        )
        bucket["sample_count"] = int(bucket["sample_count"]) + 1
        bucket["mean_score"] = float(bucket["mean_score"]) + score
        bucket["pass_rate"] = float(bucket["pass_rate"]) + (1.0 if passed else 0.0)

    for metrics in per_form.values():
        sample_count = int(metrics["sample_count"])
        metrics["mean_score"] = float(metrics["mean_score"]) / sample_count
        metrics["pass_rate"] = float(metrics["pass_rate"]) / sample_count

    sample_count = len(samples)
    return {
        "sample_count": sample_count,
        "mean_score": total_score / sample_count,
        "pass_rate": total_passes / sample_count,
        "per_form": per_form,
    }


def summarize_validation_runs(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        result
        for result in results
        if result.get("returncode") == 0 and result.get("holdout_summary") is not None
    ]
    if not completed:
        return {
            "completed_runs": 0,
            "requested_runs": len(results),
            "mean_training_reward": 0.0,
            "mean_holdout_score": 0.0,
            "mean_holdout_pass_rate": 0.0,
            "per_form": {},
        }

    mean_training_reward = sum(float(result["reward"]) for result in completed) / len(completed)
    mean_holdout_score = sum(
        float(result["holdout_summary"]["mean_score"]) for result in completed
    ) / len(completed)
    mean_holdout_pass_rate = sum(
        float(result["holdout_summary"]["pass_rate"]) for result in completed
    ) / len(completed)

    per_form: dict[str, dict[str, Any]] = {}
    for result in completed:
        holdout_summary = result["holdout_summary"]
        for form_name, metrics in holdout_summary["per_form"].items():
            bucket = per_form.setdefault(
                form_name,
                {"runs": 0, "mean_score": 0.0, "mean_pass_rate": 0.0},
            )
            bucket["runs"] = int(bucket["runs"]) + 1
            bucket["mean_score"] = float(bucket["mean_score"]) + float(metrics["mean_score"])
            bucket["mean_pass_rate"] = float(bucket["mean_pass_rate"]) + float(metrics["pass_rate"])

    for metrics in per_form.values():
        runs = int(metrics["runs"])
        metrics["mean_score"] = float(metrics["mean_score"]) / runs
        metrics["mean_pass_rate"] = float(metrics["mean_pass_rate"]) / runs

    return {
        "completed_runs": len(completed),
        "requested_runs": len(results),
        "mean_training_reward": mean_training_reward,
        "mean_holdout_score": mean_holdout_score,
        "mean_holdout_pass_rate": mean_holdout_pass_rate,
        "per_form": per_form,
    }


def run_validation_seed(
    run: ValidationRun,
    *,
    runtime_venv: str,
    holdout_seed: int,
    holdout_per_form: int,
    no_prepare_runtime: bool,
    no_wandb: bool,
    evaluate_only: bool,
) -> dict[str, Any]:
    log_path = run.output_dir / "runner.log"
    if evaluate_only:
        completed_returncode = 0
        elapsed_seconds = 0.0
    else:
        reset_dir(run.output_dir)
        command = ["bash", "scripts/run_prime_rl_gemma4_e2b.sh"]
        if no_prepare_runtime:
            command.append("--no-prepare-runtime")
        if no_wandb:
            command.append("--no-wandb")

        env = build_training_env(run, runtime_venv=runtime_venv)
        start = time.monotonic()
        with log_path.open("w") as handle:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parent.parent,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        completed_returncode = completed.returncode
        elapsed_seconds = round(time.monotonic() - start, 2)

    payload: dict[str, Any] = {
        "seed": run.seed,
        "output_dir": str(run.output_dir),
        "returncode": completed_returncode,
        "elapsed_seconds": elapsed_seconds,
        "log_path": str(log_path),
        "profile": run.profile,
        "form_set": run.form_set,
    }

    summary_path = find_final_summary(run.output_dir)
    if summary_path is not None:
        payload.update(extract_summary_metrics(summary_path))

    if completed_returncode != 0:
        payload["error"] = tail_text(log_path)
        return payload

    try:
        _samples, holdout_summary = score_holdout_samples(
            output_dir=run.output_dir,
            form_set=run.form_set,
            runtime_venv=runtime_venv,
            holdout_seed=holdout_seed,
            holdout_per_form=holdout_per_form,
            port=run.eval_port,
        )
    except Exception as exc:
        payload["holdout_error"] = str(exc)
        return payload

    holdout_summary_path = run.output_dir / "validation" / "holdout_summary.json"
    holdout_summary_path.write_text(json.dumps(holdout_summary, indent=2, sort_keys=True) + "\n")
    payload["holdout_summary"] = holdout_summary
    payload["holdout_summary_path"] = str(holdout_summary_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--profile", default="mixed-stable")
    parser.add_argument("--form-set", default="well_known")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--runtime-venv", default=DEFAULT_RUNTIME_VENV)
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument("--holdout-seed", type=int, default=DEFAULT_HOLDOUT_SEED)
    parser.add_argument("--holdout-per-form", type=int, default=DEFAULT_HOLDOUT_PER_FORM)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--no-prepare-runtime", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.no_prepare_runtime:
        prepare_runtime(args.runtime_venv)

    results_path = Path(args.results)
    if results_path.exists():
        results_path.unlink()

    results: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        run = ValidationRun(
            seed=seed,
            output_dir=output_root / f"seed_{seed}",
            port=args.port_base + index,
            eval_port=args.port_base + 100 + index,
            profile=args.profile,
            form_set=args.form_set,
        )
        print(f"\n=== seed {seed} ===")
        result = run_validation_seed(
            run,
            runtime_venv=args.runtime_venv,
            holdout_seed=args.holdout_seed,
            holdout_per_form=args.holdout_per_form,
            no_prepare_runtime=True,
            no_wandb=args.no_wandb,
            evaluate_only=args.evaluate_only,
        )
        append_result(results_path, result)
        results.append(result)
        print(
            "returncode="
            f"{result['returncode']} reward={result.get('reward')} "
            f"holdout={result.get('holdout_summary', {}).get('mean_score')}"
        )

    summary = summarize_validation_runs(results)
    summary["seeds"] = list(seeds)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote validation summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
