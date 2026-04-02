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
