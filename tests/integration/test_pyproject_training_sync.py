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
