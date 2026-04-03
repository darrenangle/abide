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
