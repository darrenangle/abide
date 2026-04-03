"""Integration tests for shared training model profiles."""

from __future__ import annotations

from pathlib import Path

from scripts import model_profiles, train_grpo_trl


def test_resolve_gemma4_e4b_profile_has_canary_defaults() -> None:
    profile = model_profiles.resolve_model_profile("google/gemma-4-E4B-it")

    assert profile.family == "gemma4_e4b"
    assert profile.stop_tokens == ("<end_of_turn>", "<eos>")
    assert profile.canary_batch_size == 8
    assert profile.canary_num_generations == 8
    assert profile.canary_beta == 0.02
    assert profile.canary_output_dir == "models/grpo_trl_gemma4_e4b"
    assert profile.vllm_gpu_memory_utilization == 0.8
    assert profile.vllm_max_model_len == 1024
    assert profile.vllm_enforce_eager is True
    assert profile.startup_timeout_seconds == 600
    assert profile.canary_use_vllm is True
    assert profile.default_lora_r == 16
    assert profile.default_lora_alpha == 32
    assert profile.causal_lm_load_kwargs() == {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }


def test_resolve_existing_profiles_preserves_other_families() -> None:
    assert model_profiles.resolve_model_profile("/tmp/baguettotron").family == "baguettotron"
    assert model_profiles.resolve_model_profile("Qwen/Qwen3-4B").family == "qwen_like"
    assert model_profiles.resolve_model_profile("google/gemma-3-4b-it").family == "gemma"


def test_train_grpo_trl_defaults_align_with_gemma4_e4b_profile() -> None:
    defaults = train_grpo_trl.TrainingArgs()
    profile = model_profiles.resolve_model_profile(defaults.model_name)

    assert defaults.model_name == "google/gemma-4-E4B-it"
    assert defaults.batch_size == profile.canary_batch_size
    assert defaults.num_generations == profile.canary_num_generations
    assert defaults.learning_rate == profile.canary_learning_rate
    assert defaults.beta == profile.canary_beta
    assert defaults.output_dir == profile.canary_output_dir
    assert defaults.use_vllm is profile.canary_use_vllm
    assert defaults.lora_r == profile.default_lora_r
    assert defaults.lora_alpha == profile.default_lora_alpha
    assert defaults.lora_dropout == profile.default_lora_dropout


def test_prepare_gemma4_runtime_uses_pinned_verified_overlay() -> None:
    script = Path("scripts/prepare_gemma4_runtime.sh").read_text()

    assert "66e86f1dbd565292a253e7d2d6851f65dc4f14ba" in script
    assert "edaac7db98e34208209fd67d8c66781b8c2e4a53" in script
    assert "uv pip install --reinstall vllm --pre" not in script


def test_gemma4_runner_exposes_named_profiles_and_auto_resume() -> None:
    script = Path("scripts/run_grpo_gemma4_e4b.sh").read_text()

    assert 'RUN_PROFILE="${ABIDE_RUN_PROFILE:-canary}"' in script
    assert 'AUTO_RESUME="${ABIDE_AUTO_RESUME:-1}"' in script
    assert "snapshot_download(repo_id=model, local_files_only=True)" in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "--auto-resume" in script
    assert "--telemetry-jsonl" in script
