"""Regression tests for training-environment pyproject wiring."""

from __future__ import annotations

from pathlib import Path

import tomllib


def test_flash_attn_build_isolation_is_configured() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    uv_config = pyproject["tool"]["uv"]

    assert uv_config["extra-build-dependencies"]["flash-attn"] == [
        {"requirement": "torch", "match-runtime": True}
    ]
    assert uv_config["extra-build-variables"]["flash-attn"] == {
        "FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"
    }


def test_training_extra_declares_explicit_runtime_stack() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    training = pyproject["project"]["optional-dependencies"]["training"]

    assert "accelerate>=1.13.0" in training
    assert "vllm>=0.18.1" in training
    assert "verifiers>=0.1.11" in training
    assert all("verifiers[rl]" not in dep for dep in training)
    assert all("verifiers-rl" not in dep for dep in training)


def test_prepare_verifiers_runtime_uses_isolated_legacy_stack() -> None:
    script = Path("scripts/prepare_verifiers_runtime.sh").read_text()

    assert 'VERIFIERS_TAG="${ABIDE_VERIFIERS_TAG:-v0.1.11}"' in script
    assert "packages/verifiers-rl" in script
    assert "UV_PROJECT_ENVIRONMENT" in script
    assert "uv sync --no-dev --extra evals" in script
    assert "66e86f1dbd565292a253e7d2d6851f65dc4f14ba" in script
    assert "edaac7db98e34208209fd67d8c66781b8c2e4a53" in script
    assert 'uv pip install --python "${VENV_DIR}/bin/python" bitsandbytes' in script
    assert 'uv pip uninstall --python "${VENV_DIR}/bin/python" flash-attn' in script


def test_prepare_prime_rl_runtime_uses_pinned_gemma4_overlay() -> None:
    script = Path("scripts/prepare_prime_rl_runtime.sh").read_text()

    assert "66e86f1dbd565292a253e7d2d6851f65dc4f14ba" in script
    assert "edaac7db98e34208209fd67d8c66781b8c2e4a53" in script
    assert (
        'uv pip install --python "${VENV_DIR}/bin/python" --reinstall "${GEMMA4_VLLM_WHEEL_URL}"'
        in script
    )
    assert 'type(config).__name__ != "Gemma4Config"' in script


def test_prime_rl_runner_uses_subset_aware_profiles_and_env_overrides() -> None:
    script = Path("scripts/run_prime_rl_gemma4_e2b.sh").read_text()

    assert 'FORM_SET_OVERRIDE="${ABIDE_FORM_SET:-}"' in script
    assert 'OUTPUT_DIR_OVERRIDE="${ABIDE_OUTPUT_DIR:-}"' in script
    assert 'MAX_STEPS_OVERRIDE="${ABIDE_MAX_STEPS:-}"' in script
    assert 'NUM_PROMPTS_OVERRIDE="${ABIDE_NUM_PROMPTS:-}"' in script
    assert 'SEED_OVERRIDE="${ABIDE_SEED:-}"' in script
    assert 'BATCH_SIZE_OVERRIDE="${ABIDE_BATCH_SIZE:-}"' in script
    assert 'MAX_TOKENS_OVERRIDE="${ABIDE_MAX_TOKENS:-}"' in script
    assert 'SEQ_LEN_OVERRIDE="${ABIDE_SEQ_LEN:-}"' in script
    assert "smoke|short-canary)" in script
    assert "long-canary)" in script
    assert "mixed-canary|mixed-stable|mixed-soak)" in script
    assert (
        'echo "--max-steps 1 --max-async-level 0 --num-prompts 4 --batch-size 1 --rollouts-per-example 1 --max-tokens 128 --seq-len 512"'
        in script
    )
    assert (
        'echo "--max-steps 4 --max-async-level 0 --num-prompts 24 --batch-size 2 --rollouts-per-example 2 --max-tokens 128 --seq-len 512"'
        in script
    )
    assert (
        'echo "--max-steps 4 --max-async-level 0 --num-prompts 16 --batch-size 1 --rollouts-per-example 1 --max-tokens 384 --seq-len 1024"'
        in script
    )
    assert (
        'echo "--max-steps 8 --max-async-level 0 --num-prompts 48 --batch-size 1 --rollouts-per-example 1 --max-tokens 384 --seq-len 1280 --learning-rate 1e-05"'
        in script
    )
    assert 'EXTRA_PROFILE_ARGS+=("--num-prompts" "$NUM_PROMPTS_OVERRIDE")' in script
    assert 'EXTRA_PROFILE_ARGS+=("--seed" "$SEED_OVERRIDE")' in script
    assert 'EXTRA_PROFILE_ARGS+=("--batch-size" "$BATCH_SIZE_OVERRIDE")' in script
    assert 'EXTRA_PROFILE_ARGS+=("--seq-len" "$SEQ_LEN_OVERRIDE")' in script


def test_verifiers_vllm_server_matches_current_vllm_api() -> None:
    server = Path("src/abide/verifiers_vllm_server.py").read_text()

    assert "supported_tasks = await engine.get_supported_tasks()" in server
    assert 'cast("Any", build_app)(args, supported_tasks, engine.model_config)' in server
    assert "await init_app_state(engine, app.state, args, supported_tasks)" in server
    assert "get_vllm_config" not in server


def test_legacy_verifiers_runner_defaults_to_gemma4_and_local_server() -> None:
    runner = Path("scripts/run_grpo.sh").read_text()

    assert 'MODEL="${ABIDE_MODEL:-google/gemma-4-E4B-it}"' in runner
    assert (
        'OUTPUT_DIR="${ABIDE_OUTPUT_DIR:-models/abide_verifiers_gemma4_e4b_well_known}"' in runner
    )
    assert "-m abide.verifiers_vllm_server" in runner
    assert 'TRAIN_GPU="${ABIDE_TRAIN_GPU:-}"' in runner
    assert 'VLLM_GPU="${ABIDE_VLLM_GPU:-}"' in runner
    assert 'TRAIN_MIN_FREE_MIB="${ABIDE_TRAIN_MIN_FREE_MIB:-8192}"' in runner
