"""Integration tests for the modern prime-rl training entrypoint."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import train_prime_rl


def test_prime_rl_defaults_target_gemma4_e2b_and_well_known_subset() -> None:
    config = train_prime_rl.PrimeRLTrainingConfig()

    assert config.model_name == "google/gemma-4-E2B-it"
    assert config.model_path is None
    assert config.form_set == "well_known"
    assert config.use_wandb is False


def test_prime_rl_parser_accepts_short_and_long_well_known_form_sets() -> None:
    parser = train_prime_rl.build_parser()

    short_args = parser.parse_args(["--form-set", "well_known_short", "--no-wandb"])
    long_args = parser.parse_args(["--form-set", "well_known_long", "--no-wandb"])

    assert train_prime_rl.config_from_args(short_args).form_set == "well_known_short"
    assert train_prime_rl.config_from_args(long_args).form_set == "well_known_long"


def test_build_prime_rl_toml_uses_installable_abide_env() -> None:
    config = train_prime_rl.PrimeRLTrainingConfig(
        output_dir="models/test_prime_rl",
        single_form="Haiku",
        port=8134,
    )

    toml_text = train_prime_rl.build_prime_rl_toml(config)

    assert 'id = "abide-poetry-forms"' in toml_text
    assert 'form_name = "Haiku"' in toml_text
    assert 'name = "google/gemma-4-E2B-it"' in toml_text
    assert 'base_url = ["http://localhost:8134/v1"]' in toml_text
    assert "[model.vlm]" in toml_text
    assert 'language_model_attr = "model.language_model"' in toml_text
    assert "[trainer.tokenizer]" in toml_text
    assert "trust_remote_code = true" in toml_text
    assert "fsdp_cpu_offload = true" in toml_text
    assert "[trainer.model.ac_offloading]" in toml_text
    assert "max_inflight_activations = 2" in toml_text
    assert "optim_cpu_offload = true" not in toml_text
    assert "[trainer.model.lora]" not in toml_text
    assert "enable_lora = false" in toml_text
    assert "[trainer.weight_broadcast]" in toml_text
    assert "[inference.weight_broadcast]" not in toml_text
    assert toml_text.count('save_format = "torch"') >= 2
    assert "offline = true" in toml_text


def test_build_prime_rl_toml_preserves_long_form_set_in_env_args() -> None:
    config = train_prime_rl.PrimeRLTrainingConfig(
        output_dir="models/test_prime_rl_long",
        form_set="well_known_long",
    )

    toml_text = train_prime_rl.build_prime_rl_toml(config)

    assert 'form_set = "well_known_long"' in toml_text


def test_validate_length_budget_rejects_tiny_mixed_form_budget() -> None:
    config = train_prime_rl.PrimeRLTrainingConfig(
        form_set="well_known",
        max_tokens=128,
        seq_len=512,
    )

    with pytest.raises(ValueError, match="Long-form or mixed well-known runs need at least"):
        train_prime_rl.validate_length_budget(config)


def test_validate_length_budget_allows_short_subset_smoke_budget() -> None:
    config = train_prime_rl.PrimeRLTrainingConfig(
        form_set="well_known_short",
        max_tokens=128,
        seq_len=512,
    )

    train_prime_rl.validate_length_budget(config)


def test_validate_rollout_shape_rejects_indivisible_batch() -> None:
    config = train_prime_rl.PrimeRLTrainingConfig(batch_size=1, rollouts_per_example=2)

    with pytest.raises(ValueError, match="batch_size must be divisible"):
        train_prime_rl.validate_rollout_shape(config)


def test_build_prime_rl_toml_keeps_lora_for_non_gemma4_models() -> None:
    config = train_prime_rl.PrimeRLTrainingConfig(
        model_name="google/gemma-3-4b-it",
        output_dir="models/test_prime_rl_gemma3",
    )

    toml_text = train_prime_rl.build_prime_rl_toml(config)

    assert "[trainer.model.lora]" in toml_text
    assert "optim_cpu_offload = true" in toml_text
    assert "fsdp_cpu_offload = true" not in toml_text
    assert "save_adapter_separately = true" in toml_text
    assert "enable_lora = true" in toml_text


def test_write_prime_rl_config_writes_under_output_dir(tmp_path: Path) -> None:
    config = train_prime_rl.PrimeRLTrainingConfig(output_dir=str(tmp_path / "run"))

    config_path = train_prime_rl.write_prime_rl_config(config)

    assert config_path == tmp_path / "run" / "prime_rl" / "rl.toml"
    assert config_path.exists()


def test_find_final_summary_prefers_latest_run_summary(tmp_path: Path) -> None:
    first = tmp_path / "run_default" / "run-a" / "final_summary.json"
    second = tmp_path / "run_default" / "run-b" / "final_summary.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}")
    second.write_text("{}")

    assert train_prime_rl.find_final_summary(tmp_path) == second


def test_trainer_log_has_fatal_error_detects_marker(tmp_path: Path) -> None:
    trainer_log = tmp_path / "logs" / "trainer.log"
    trainer_log.parent.mkdir(parents=True)
    trainer_log.write_text("Fatal error in train\n")

    assert train_prime_rl.trainer_log_has_fatal_error(tmp_path) is True


def test_wait_for_prime_rl_completion_reaps_stuck_process_after_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary = tmp_path / "run_default" / "run-a" / "final_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}")
    trainer_log = tmp_path / "logs" / "trainer.log"
    trainer_log.parent.mkdir(parents=True)
    trainer_log.write_text("all good\n")

    class FakeProc:
        pid = 4242

        def __init__(self) -> None:
            self.wait_calls = 0

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is None or timeout >= 0.0
            self.wait_calls += 1
            return 0

    fake_proc = FakeProc()
    kill_calls: list[tuple[int, int]] = []
    sleeps: list[float] = []

    monkeypatch.setattr(train_prime_rl.os, "killpg", lambda pid, sig: kill_calls.append((pid, sig)))
    monotonic_values = iter((0.0, 0.0, 25.0))
    monkeypatch.setattr(train_prime_rl.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(train_prime_rl.time, "sleep", lambda seconds: sleeps.append(seconds))

    return_code = train_prime_rl.wait_for_prime_rl_completion(
        fake_proc,  # type: ignore[arg-type]
        output_dir=tmp_path,
        grace_period_seconds=20.0,
        poll_interval_seconds=2.0,
    )

    assert return_code == 0
    assert kill_calls == [(4242, train_prime_rl.signal.SIGTERM)]
    assert fake_proc.wait_calls == 1
    assert sleeps == [2.0]


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

    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_build_prime_rl_env_preserves_existing_allocator_setting(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    env = train_prime_rl.build_prime_rl_env(".venv-prime-rl")

    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_sitecustomize_backfills_prime_rl_vllm_serve_symbol() -> None:
    import sitecustomize

    importlib.reload(sitecustomize)

    serve = importlib.import_module("vllm.entrypoints.cli.serve")

    assert hasattr(serve, "run_api_server_worker_proc")


def test_sitecustomize_resolves_chat_kwargs_for_current_vllm_signature() -> None:
    import sitecustomize

    def fake_chat_init(
        self,
        engine_client,
        models,
        response_role,
        *,
        openai_serving_render,
        request_logger,
        chat_template,
        chat_template_content_format,
        trust_request_chat_template=False,
        return_tokens_as_token_ids=False,
        reasoning_parser="",
        enable_auto_tools=False,
        exclude_tools_when_tool_choice_none=False,
        tool_parser=None,
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
        enable_log_outputs=False,
        enable_log_deltas=True,
        default_chat_template_kwargs=None,
    ):
        return None

    state = SimpleNamespace(openai_serving_render="render")
    args = SimpleNamespace(
        chat_template_content_format="openai",
        trust_request_chat_template=True,
        return_tokens_as_token_ids=True,
        structured_outputs_config=SimpleNamespace(reasoning_parser="parser"),
        enable_auto_tool_choice=True,
        exclude_tools_when_tool_choice_none=True,
        tool_call_parser="hermes",
        enable_prompt_tokens_details=True,
        enable_force_include_usage=True,
        enable_log_outputs=True,
        enable_log_deltas=False,
        default_chat_template_kwargs={"foo": "bar"},
        log_error_stack=True,
    )

    kwargs = sitecustomize._resolve_openai_serving_chat_kwargs(
        fake_chat_init,
        state,
        args,
        request_logger="logger",
        resolved_chat_template="template",
    )

    assert kwargs["openai_serving_render"] == "render"
    assert kwargs["request_logger"] == "logger"
    assert kwargs["chat_template"] == "template"
    assert kwargs["reasoning_parser"] == "parser"
    assert kwargs["default_chat_template_kwargs"] == {"foo": "bar"}
    assert "log_error_stack" not in kwargs


def test_sitecustomize_backfills_prime_rl_perf_counter_for_none_experts(monkeypatch) -> None:
    import sitecustomize

    fake_prime_rl = types.ModuleType("prime_rl")
    fake_trainer = types.ModuleType("prime_rl.trainer")
    fake_perf = types.ModuleType("prime_rl.trainer.perf")

    class FakePerfCounter:
        @staticmethod
        def get_active_mm_params(config):
            if hasattr(config, "text_config"):
                config = config.text_config
            return float(config.num_experts * config.hidden_size)

    fake_perf.PerfCounter = FakePerfCounter

    monkeypatch.setitem(sys.modules, "prime_rl", fake_prime_rl)
    monkeypatch.setitem(sys.modules, "prime_rl.trainer", fake_trainer)
    monkeypatch.setitem(sys.modules, "prime_rl.trainer.perf", fake_perf)

    importlib.reload(sitecustomize)

    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=None, hidden_size=4))

    assert fake_perf.PerfCounter.get_active_mm_params(config) == 0.0
