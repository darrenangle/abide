"""Integration tests for shared training model profiles."""

from __future__ import annotations

from scripts import model_profiles, train_grpo_trl


def test_resolve_gemma4_e4b_profile_has_canary_defaults() -> None:
    profile = model_profiles.resolve_model_profile("google/gemma-4-E4B-it")

    assert profile.family == "gemma4_e4b"
    assert profile.stop_tokens == ("<end_of_turn>", "<eos>")
    assert profile.canary_batch_size == 8
    assert profile.canary_num_generations == 8
    assert profile.canary_beta == 0.02
    assert profile.canary_output_dir == "models/grpo_trl_gemma4_e4b"
    assert profile.startup_timeout_seconds == 600
    assert profile.canary_use_vllm is False
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
