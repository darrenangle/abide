from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    """Shared model-specific loading and canary defaults."""

    family: str
    stop_tokens: tuple[str, ...] = ()
    trust_remote_code: bool = True
    attn_implementation: str | None = "flash_attention_2"
    canary_num_prompts: int = 2000
    canary_batch_size: int = 8
    canary_num_generations: int = 8
    canary_beta: float = 0.02
    canary_learning_rate: float = 3e-5
    canary_output_dir: str = "models/grpo_trl_canary"
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int = 4096
    startup_timeout_seconds: int = 300

    def causal_lm_load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation
        return kwargs


DEFAULT_PROFILE = ModelProfile(family="default")

BAGUETTOTRON_PROFILE = ModelProfile(
    family="baguettotron",
    stop_tokens=("<|im_end|>",),
    attn_implementation=None,
    canary_output_dir="models/grpo_trl_baguettotron",
)

QWEN_LIKE_PROFILE = ModelProfile(
    family="qwen_like",
    stop_tokens=("<|im_end|>", "<|endoftext|>"),
)

GEMMA_PROFILE = ModelProfile(
    family="gemma",
    stop_tokens=("<end_of_turn>", "<eos>"),
    canary_output_dir="models/grpo_trl_gemma",
)

GEMMA_4_E2B_PROFILE = ModelProfile(
    family="gemma4_e2b",
    stop_tokens=("<end_of_turn>", "<eos>"),
    canary_num_prompts=2000,
    canary_batch_size=10,
    canary_num_generations=8,
    canary_beta=0.02,
    canary_learning_rate=3e-5,
    canary_output_dir="models/grpo_trl_gemma4_e2b",
    vllm_gpu_memory_utilization=0.9,
    vllm_max_model_len=4096,
    startup_timeout_seconds=600,
)

GEMMA_4_E4B_PROFILE = ModelProfile(
    family="gemma4_e4b",
    stop_tokens=("<end_of_turn>", "<eos>"),
    canary_num_prompts=2000,
    canary_batch_size=8,
    canary_num_generations=8,
    canary_beta=0.02,
    canary_learning_rate=3e-5,
    canary_output_dir="models/grpo_trl_gemma4_e4b",
    vllm_gpu_memory_utilization=0.88,
    vllm_max_model_len=4096,
    startup_timeout_seconds=600,
)


def resolve_model_profile(model_name: str) -> ModelProfile:
    """Resolve shared model settings from a model identifier."""
    model_lower = model_name.lower()

    if "baguettotron" in model_lower:
        return BAGUETTOTRON_PROFILE
    if "gemma-4-e4b" in model_lower:
        return GEMMA_4_E4B_PROFILE
    if "gemma-4-e2b" in model_lower:
        return GEMMA_4_E2B_PROFILE
    if "gemma" in model_lower:
        return GEMMA_PROFILE
    if any(token in model_lower for token in ("qwen", "deepseek", "olmo")):
        return QWEN_LIKE_PROFILE
    return DEFAULT_PROFILE


__all__ = [
    "BAGUETTOTRON_PROFILE",
    "DEFAULT_PROFILE",
    "GEMMA_4_E2B_PROFILE",
    "GEMMA_4_E4B_PROFILE",
    "GEMMA_PROFILE",
    "QWEN_LIKE_PROFILE",
    "ModelProfile",
    "resolve_model_profile",
]
