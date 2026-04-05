#!/usr/bin/env python3
"""Modern prime-rl entrypoint for Abide poetry training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src and scripts to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from model_profiles import resolve_model_profile

from abide.training.prime_rl_env import DEFAULT_ENV_ID, PRIME_RL_DEFAULT_MODEL

DEFAULT_OUTPUT_DIR = "models/prime_rl_gemma4_e2b_well_known"
DEFAULT_RUNTIME_VENV = ".venv-prime-rl"
_WEIGHT_FILE_PATTERNS = (
    "model*.safetensors",
    "pytorch_model*.bin",
    "*.safetensors.index.json",
    "*.bin.index.json",
)


@dataclass(frozen=True)
class PrimeRLTrainingConfig:
    model_name: str = PRIME_RL_DEFAULT_MODEL
    model_path: str | None = None
    form_set: str = "well_known"
    single_form: str | None = None
    form_names: str | None = None
    num_prompts: int = 2048
    seed: int = 42
    max_steps: int = 8
    max_async_level: int = 4
    batch_size: int = 64
    rollouts_per_example: int = 4
    max_tokens: int = 384
    seq_len: int = 1280
    learning_rate: float = 1e-5
    output_dir: str = DEFAULT_OUTPUT_DIR
    port: int = 8000
    use_wandb: bool = False
    wandb_project: str = "abide-prime-rl"
    wandb_name: str | None = None
    runtime_venv: str = DEFAULT_RUNTIME_VENV
    prepare_runtime: bool = True
    run_training: bool = True
    dry_run: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PRIME_RL_DEFAULT_MODEL)
    parser.add_argument("--model-path", help="Optional local model artifact or snapshot path.")
    parser.add_argument(
        "--form-set", default="well_known", choices=("all", "rl_default", "well_known")
    )
    parser.add_argument("--single-form")
    parser.add_argument("--form-names", help="Comma-separated explicit form names.")
    parser.add_argument("--num-prompts", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-async-level", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollouts-per-example", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--seq-len", type=int, default=1280)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wandb-project", default="abide-prime-rl")
    parser.add_argument("--wandb-name")
    parser.add_argument("--runtime-venv", default=DEFAULT_RUNTIME_VENV)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-prepare-runtime", action="store_true")
    parser.add_argument("--write-config-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> PrimeRLTrainingConfig:
    return PrimeRLTrainingConfig(
        model_name=args.model,
        model_path=args.model_path,
        form_set=args.form_set,
        single_form=args.single_form,
        form_names=args.form_names,
        num_prompts=args.num_prompts,
        seed=args.seed,
        max_steps=args.max_steps,
        max_async_level=args.max_async_level,
        batch_size=args.batch_size,
        rollouts_per_example=args.rollouts_per_example,
        max_tokens=args.max_tokens,
        seq_len=args.seq_len,
        learning_rate=args.learning_rate,
        output_dir=args.output,
        port=args.port,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        runtime_venv=args.runtime_venv,
        prepare_runtime=not args.no_prepare_runtime,
        run_training=not args.write_config_only,
        dry_run=args.dry_run,
    )


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise ValueError("None is not a valid inline TOML literal")
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{key} = {_toml_literal(val)}" for key, val in value.items())
        return "{ " + items + " }"
    raise TypeError(f"Unsupported TOML literal: {type(value)!r}")


def _model_artifact_has_weights(model_dir: Path) -> bool:
    return any(any(model_dir.glob(pattern)) for pattern in _WEIGHT_FILE_PATTERNS)


def resolve_model_target(
    config: PrimeRLTrainingConfig, *, allow_download: bool = True
) -> tuple[str, bool]:
    if config.model_path:
        explicit_model_path = Path(config.model_path).expanduser().resolve()
        return str(explicit_model_path), True

    explicit_path = Path(config.model_name).expanduser()
    if explicit_path.exists():
        return str(explicit_path.resolve()), True

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return config.model_name, False

    try:
        snapshot = snapshot_download(repo_id=config.model_name, local_files_only=True)
    except Exception:
        snapshot = None

    if snapshot is not None:
        snapshot_path = Path(snapshot).resolve()
        if _model_artifact_has_weights(snapshot_path):
            return str(snapshot_path), True

    if not allow_download:
        return config.model_name, False

    downloaded_snapshot = snapshot_download(repo_id=config.model_name)
    downloaded_snapshot_path = Path(downloaded_snapshot).resolve()
    if _model_artifact_has_weights(downloaded_snapshot_path):
        return str(downloaded_snapshot_path), True

    return config.model_name, False


def build_prime_rl_toml(config: PrimeRLTrainingConfig, *, model_target: str | None = None) -> str:
    profile = resolve_model_profile(config.model_name)
    output_dir = Path(config.output_dir)
    model_lower = config.model_name.lower()
    resolved_model_target = model_target or config.model_name
    env_args: dict[str, Any] = {
        "form_set": config.form_set,
        "num_prompts": config.num_prompts,
        "seed": config.seed,
    }
    if config.single_form:
        env_args["form_name"] = config.single_form
    if config.form_names:
        env_args["form_names"] = config.form_names

    target_modules = profile.lora_target_modules
    if isinstance(target_modules, str):
        target_modules = (target_modules,)
    use_lora = profile.prime_rl_use_lora

    lines = [
        f"max_steps = {config.max_steps}",
        f"max_async_level = {config.max_async_level}",
        f"seq_len = {config.seq_len}",
        f"dry_run = {_toml_literal(config.dry_run)}",
        "",
        f"output_dir = {_toml_literal(output_dir.as_posix())}",
        "",
        "[ckpt]",
        "",
        "[model]",
        f"name = {_toml_literal(resolved_model_target)}",
        "",
        "[trainer.model]",
        'impl = "hf"',
        f"attn = {_toml_literal(profile.attn_implementation or 'sdpa')}",
        f"trust_remote_code = {_toml_literal(profile.trust_remote_code)}",
        'optimization_dtype = "bfloat16"',
        'reduce_dtype = "bfloat16"',
        "optim_cpu_offload = true",
        'fused_lm_head_token_chunk_size = "auto"',
        "",
        "[trainer.tokenizer]",
        f"name = {_toml_literal(resolved_model_target)}",
        f"trust_remote_code = {_toml_literal(profile.trust_remote_code)}",
        "",
        "[trainer.model.ac]",
        "freq = 1",
        "",
        "[trainer.optim]",
        f"lr = {config.learning_rate}",
        "",
        "[trainer.ckpt.weights]",
        "",
        "[orchestrator]",
        f"batch_size = {config.batch_size}",
        f"rollouts_per_example = {config.rollouts_per_example}",
        "",
        "[orchestrator.sampling]",
        f"max_tokens = {config.max_tokens}",
        "temperature = 0.8",
        "repetition_penalty = 1.1",
        f"seed = {config.seed}",
        "",
        "[[orchestrator.env]]",
        f"id = {_toml_literal(DEFAULT_ENV_ID)}",
        'name = "abide-poetry-forms"',
        f"args = {_toml_literal(env_args)}",
        "",
        "[inference]",
        f"enable_lora = {_toml_literal(use_lora)}",
        f"gpu_memory_utilization = {profile.vllm_gpu_memory_utilization}",
        "",
        "[inference.model]",
        f"name = {_toml_literal(resolved_model_target)}",
        'dtype = "bfloat16"',
        f"max_model_len = {min(config.seq_len, profile.vllm_max_model_len)}",
        f"enforce_eager = {_toml_literal(profile.vllm_enforce_eager)}",
        f"trust_remote_code = {_toml_literal(profile.trust_remote_code)}",
        "",
        "[inference.server]",
        f"port = {config.port}",
        "",
        "[deployment]",
        'type = "single_node"',
        "num_train_gpus = 1",
        "num_infer_gpus = 1",
    ]

    if use_lora:
        lines[lines.index("[trainer.optim]") : lines.index("[trainer.optim]")] = [
            "[trainer.model.lora]",
            f"rank = {profile.default_lora_r}",
            f"alpha = {profile.default_lora_alpha}",
            f"dropout = {profile.default_lora_dropout}",
            f"target_modules = {_toml_literal(list(target_modules))}",
            "",
        ]
        trainer_ckpt_idx = lines.index("[trainer.ckpt.weights]") + 1
        lines[trainer_ckpt_idx:trainer_ckpt_idx] = ["save_adapter_separately = true"]
        orchestrator_sampling_idx = lines.index("[orchestrator.sampling]")
        lines[orchestrator_sampling_idx:orchestrator_sampling_idx] = [
            "[orchestrator.model.lora]",
            'name = "abide-e2b-lora"',
            "",
        ]
        inference_model_idx = lines.index("[inference.model]")
        lines[inference_model_idx:inference_model_idx] = [
            f"max_lora_rank = {profile.default_lora_r}",
            "",
        ]

    if "gemma-4" in model_lower:
        lines.extend(
            [
                "",
                "[model.vlm]",
                'vision_encoder_attr = "model.vision_tower"',
                'language_model_attr = "model.language_model"',
            ]
        )

    if config.use_wandb:
        lines.extend(
            [
                "",
                "[wandb]",
                f"project = {_toml_literal(config.wandb_project)}",
            ]
        )
        if config.wandb_name:
            lines.append(f"name = {_toml_literal(config.wandb_name)}")
        else:
            lines.append(f"name = {_toml_literal(output_dir.name)}")
    else:
        lines.extend(
            [
                "",
                "[wandb]",
                "offline = true",
                "shared = false",
            ]
        )

    return "\n".join(lines) + "\n"


def write_prime_rl_config(
    config: PrimeRLTrainingConfig, *, model_target: str | None = None
) -> Path:
    output_dir = Path(config.output_dir)
    config_dir = output_dir / "prime_rl"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "rl.toml"
    config_path.write_text(build_prime_rl_toml(config, model_target=model_target))
    return config_path


def prepare_runtime(runtime_venv: str) -> None:
    env = os.environ.copy()
    env["ABIDE_PRIME_RL_VENV"] = runtime_venv
    subprocess.run(
        ["bash", "scripts/prepare_prime_rl_runtime.sh"],
        check=True,
        env=env,
    )


def build_prime_rl_command(config_path: Path, runtime_venv: str) -> list[str]:
    return [str(Path(runtime_venv) / "bin" / "rl"), "@", str(config_path)]


def build_prime_rl_env(runtime_venv: str, *, use_local_model: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = str(Path(runtime_venv) / "bin")
    repo_root = Path(__file__).resolve().parent.parent
    src_path = str(repo_root / "src")
    current_path = env.get("PATH", "")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PATH"] = f"{venv_bin}:{current_path}" if current_path else venv_bin
    env["PYTHONPATH"] = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path
    env["VIRTUAL_ENV"] = str(Path(runtime_venv).resolve())
    if use_local_model:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def main() -> int:
    parser = build_parser()
    config = config_from_args(parser.parse_args())
    model_target, use_local_model = resolve_model_target(
        config,
        allow_download=config.run_training and not config.dry_run,
    )
    if use_local_model and model_target != config.model_name:
        print(f"Using cached local model snapshot: {model_target}")
    config_path = write_prime_rl_config(config, model_target=model_target)
    print(f"Wrote prime-rl config to {config_path}")

    if config.prepare_runtime:
        prepare_runtime(config.runtime_venv)

    if not config.run_training:
        return 0

    command = build_prime_rl_command(config_path, config.runtime_venv)
    print("Running:", " ".join(command))
    subprocess.run(
        command,
        check=True,
        env=build_prime_rl_env(config.runtime_venv, use_local_model=use_local_model),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
