"""Integration tests for the modern prime-rl training entrypoint."""

from __future__ import annotations

import importlib
from pathlib import Path

from scripts import train_prime_rl


def test_prime_rl_defaults_target_gemma4_e2b_and_well_known_subset() -> None:
    config = train_prime_rl.PrimeRLTrainingConfig()

    assert config.model_name == "google/gemma-4-E2B-it"
    assert config.model_path is None
    assert config.form_set == "well_known"
    assert config.use_wandb is False


def test_build_prime_rl_toml_uses_installable_abide_env() -> None:
    config = train_prime_rl.PrimeRLTrainingConfig(
        output_dir="models/test_prime_rl",
        single_form="Haiku",
    )

    toml_text = train_prime_rl.build_prime_rl_toml(config)

    assert 'id = "abide-poetry-forms"' in toml_text
    assert 'form_name = "Haiku"' in toml_text
    assert 'name = "google/gemma-4-E2B-it"' in toml_text
    assert "[model.vlm]" in toml_text
    assert 'language_model_attr = "model.language_model"' in toml_text
    assert "[trainer.tokenizer]" in toml_text
    assert "trust_remote_code = true" in toml_text
    assert "[trainer.model.lora]" not in toml_text
    assert "enable_lora = false" in toml_text
    assert "offline = true" in toml_text


def test_build_prime_rl_toml_keeps_lora_for_non_gemma4_models() -> None:
    config = train_prime_rl.PrimeRLTrainingConfig(
        model_name="google/gemma-3-4b-it",
        output_dir="models/test_prime_rl_gemma3",
    )

    toml_text = train_prime_rl.build_prime_rl_toml(config)

    assert "[trainer.model.lora]" in toml_text
    assert "save_adapter_separately = true" in toml_text
    assert "enable_lora = true" in toml_text


def test_write_prime_rl_config_writes_under_output_dir(tmp_path: Path) -> None:
    config = train_prime_rl.PrimeRLTrainingConfig(output_dir=str(tmp_path / "run"))

    config_path = train_prime_rl.write_prime_rl_config(config)

    assert config_path == tmp_path / "run" / "prime_rl" / "rl.toml"
    assert config_path.exists()


def test_build_prime_rl_command_uses_runtime_venv_bin() -> None:
    config_path = Path("/tmp/abide/prime_rl/rl.toml")

    command = train_prime_rl.build_prime_rl_command(config_path, ".venv-prime-rl")

    assert command == [".venv-prime-rl/bin/rl", "@", "/tmp/abide/prime_rl/rl.toml"]


def test_resolve_model_target_prefers_explicit_model_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "gemma-local"
    model_dir.mkdir()
    config = train_prime_rl.PrimeRLTrainingConfig(model_path=str(model_dir))

    target, use_local = train_prime_rl.resolve_model_target(config)

    assert target == str(model_dir.resolve())
    assert use_local is True


def test_resolve_model_target_uses_cached_snapshot(monkeypatch, tmp_path: Path) -> None:
    fake_snapshot = tmp_path / "fake-gemma-snapshot"
    fake_snapshot.mkdir()
    (fake_snapshot / "model.safetensors").write_text("weights")

    import sys
    import types

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda **_kwargs: str(fake_snapshot)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    config = train_prime_rl.PrimeRLTrainingConfig()

    target, use_local = train_prime_rl.resolve_model_target(config)

    assert target == str(fake_snapshot.resolve())
    assert use_local is True


def test_resolve_model_target_ignores_incomplete_snapshot_without_download(
    monkeypatch,
    tmp_path: Path,
) -> None:
    incomplete_snapshot = tmp_path / "snapshot"
    incomplete_snapshot.mkdir()

    import sys
    import types

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda **_kwargs: str(incomplete_snapshot)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    config = train_prime_rl.PrimeRLTrainingConfig()

    target, use_local = train_prime_rl.resolve_model_target(config, allow_download=False)

    assert target == "google/gemma-4-E2B-it"
    assert use_local is False


def test_resolve_model_target_downloads_when_snapshot_is_incomplete(
    monkeypatch,
    tmp_path: Path,
) -> None:
    incomplete_snapshot = tmp_path / "snapshot-incomplete"
    incomplete_snapshot.mkdir()
    complete_snapshot = tmp_path / "snapshot-complete"
    complete_snapshot.mkdir()
    (complete_snapshot / "model-00001-of-00002.safetensors").write_text("weights")

    import sys
    import types

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            return str(incomplete_snapshot)
        return str(complete_snapshot)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    config = train_prime_rl.PrimeRLTrainingConfig()

    target, use_local = train_prime_rl.resolve_model_target(config, allow_download=True)

    assert target == str(complete_snapshot.resolve())
    assert use_local is True
    assert calls[0]["local_files_only"] is True
    assert "local_files_only" not in calls[1]


def test_build_prime_rl_env_prefixes_runtime_bin(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")

    env = train_prime_rl.build_prime_rl_env(".venv-prime-rl")

    assert env["PATH"].startswith(".venv-prime-rl/bin:")
    assert "/src" in env["PYTHONPATH"]
    assert env["VIRTUAL_ENV"].endswith("/.venv-prime-rl")


def test_build_prime_rl_env_sets_offline_flags_for_local_models(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")

    env = train_prime_rl.build_prime_rl_env(".venv-prime-rl", use_local_model=True)

    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_sitecustomize_backfills_prime_rl_vllm_serve_symbol() -> None:
    import sitecustomize

    importlib.reload(sitecustomize)

    serve = importlib.import_module("vllm.entrypoints.cli.serve")

    assert hasattr(serve, "run_api_server_worker_proc")
