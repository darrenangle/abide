"""Integration tests for the legacy verifiers GRPO training helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from scripts import train_grpo


class _StubForm:
    def __init__(self, score: float) -> None:
        self._score = score

    def verify(self, _poem: str) -> SimpleNamespace:
        return SimpleNamespace(score=self._score, passed=self._score >= 0.8, rubric=[])


class _StubParser:
    def parse_answer(self, completion) -> str:
        del completion
        return "This completion is long enough to score."


def test_create_dataset_defaults_to_well_known_form_set(monkeypatch) -> None:
    fake_prompt_generator = ModuleType("prompt_generator")
    calls: list[str] = []

    def fake_generate_well_known_verifiers_dataset(*, num_prompts: int, seed: int) -> list[dict]:
        del num_prompts, seed
        calls.append("well_known")
        return [
            {
                "prompt": [{"role": "user", "content": "Write a Haiku poem about rain."}],
                "info": {"form_name": "Haiku"},
            }
        ]

    def unexpected(*args, **kwargs):
        raise AssertionError("unexpected dataset builder")

    fake_prompt_generator.generate_well_known_verifiers_dataset = (
        fake_generate_well_known_verifiers_dataset
    )
    fake_prompt_generator.generate_single_form_verifiers_dataset = unexpected
    fake_prompt_generator.generate_learnable_forms_verifiers_dataset = unexpected
    fake_prompt_generator.generate_traditional_verifiers_dataset = unexpected
    fake_prompt_generator.generate_verifiers_dataset = unexpected
    fake_prompt_generator.generate_rl_default_verifiers_dataset = unexpected
    monkeypatch.setitem(sys.modules, "prompt_generator", fake_prompt_generator)

    dataset, eval_dataset = train_grpo.create_dataset(train_grpo.TrainingConfig(num_prompts=1))

    assert calls == ["well_known"]
    assert eval_dataset is None
    assert dataset[0]["info"]["form_name"] == "Haiku"


def test_resolve_default_form_set_preserves_legacy_env_flags(monkeypatch) -> None:
    monkeypatch.delenv("ABIDE_FORM_SET", raising=False)
    monkeypatch.setenv("ABIDE_TRADITIONAL", "1")
    monkeypatch.delenv("ABIDE_LEARNABLE", raising=False)

    assert train_grpo.resolve_default_form_set() == "traditional"


def test_create_reward_function_uses_exact_form_metadata() -> None:
    forms = {
        "Ode": _StubForm(0.1),
        "Villanelle": _StubForm(0.9),
    }
    reward_fn = train_grpo.create_reward_function(
        forms,
        telemetry_every=999999,
        use_wandb=False,
    )

    reward = reward_fn(
        completion=[{"role": "assistant", "content": "ignored"}],
        info={"form_name": "Villanelle"},
        parser=_StubParser(),
    )

    assert reward == 0.9


def test_normalize_generated_poem_strips_tags_and_code_fences() -> None:
    raw = "<think>reasoning</think>```poem\nline one\nline two\n```<end_of_turn>"

    assert train_grpo.normalize_generated_poem(raw) == "line one\nline two"


def test_build_rl_config_uses_legacy_field_names(monkeypatch) -> None:
    class FakeRLConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(train_grpo, "import_verifiers_rl_trainer", lambda: (FakeRLConfig, object))

    rl_config = train_grpo.build_rl_config(
        train_grpo.TrainingConfig(use_wandb=False),
        max_steps=4,
    )

    assert "lora_rank" in rl_config.kwargs
    assert "lora_r" not in rl_config.kwargs
    assert rl_config.kwargs["report_to"] == []
    assert rl_config.kwargs["zero_truncated_completions"] is True


def test_train_grpo_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "scripts/train_grpo.py", "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--form-set" in result.stdout
    assert "--single-form" in result.stdout
    assert "--eval-prompts" in result.stdout
    assert "prime-rl" in result.stdout
