"""Integration tests for the TRL GRPO training helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from scripts import train_grpo_trl


class _StubForm:
    def __init__(self, score: float) -> None:
        self._score = score

    def verify(self, _poem: str) -> SimpleNamespace:
        return SimpleNamespace(score=self._score)


def test_create_reward_function_uses_exact_form_metadata() -> None:
    forms = {
        "Ode": _StubForm(0.1),
        "PindaricOde": _StubForm(0.9),
    }
    reward_fn = train_grpo_trl.create_reward_function(forms)

    rewards = reward_fn(
        completions=["This completion is long enough to score."],
        prompts=[[{"role": "user", "content": "Write a PindaricOde poem about winter."}]],
        form_name=["PindaricOde"],
    )

    assert rewards == [0.9]


def test_create_dataset_preserves_form_name_metadata(monkeypatch) -> None:
    fake_prompt_generator = ModuleType("prompt_generator")

    def fake_generate_rl_default_verifiers_dataset(*, num_prompts: int, seed: int) -> list[dict]:
        del num_prompts, seed
        return [
            {
                "prompt": [{"role": "user", "content": "Write a PindaricOde poem about winter."}],
                "info": {"form_name": "PindaricOde"},
            }
        ]

    fake_prompt_generator.generate_rl_default_verifiers_dataset = (
        fake_generate_rl_default_verifiers_dataset
    )
    fake_prompt_generator.generate_learnable_forms_verifiers_dataset = (
        fake_generate_rl_default_verifiers_dataset
    )
    fake_prompt_generator.generate_traditional_verifiers_dataset = (
        fake_generate_rl_default_verifiers_dataset
    )
    fake_prompt_generator.generate_verifiers_dataset = fake_generate_rl_default_verifiers_dataset
    fake_prompt_generator.resolve_form_selection_mode = lambda: "rl_default"
    monkeypatch.setitem(sys.modules, "prompt_generator", fake_prompt_generator)

    dataset = train_grpo_trl.create_dataset(train_grpo_trl.TrainingArgs(num_prompts=1), {})

    row = dataset[0]
    assert row["form_name"] == "PindaricOde"
    assert row["prompt"][0]["content"][0]["text"] == "Write a PindaricOde poem about winter."


def test_create_dataset_defaults_to_rl_default_mode(monkeypatch) -> None:
    fake_prompt_generator = ModuleType("prompt_generator")
    calls: list[str] = []

    def fake_generate_rl_default_verifiers_dataset(*, num_prompts: int, seed: int) -> list[dict]:
        del num_prompts, seed
        calls.append("rl_default")
        return [
            {
                "prompt": [{"role": "user", "content": "Write a Haiku poem about rain."}],
                "info": {"form_name": "Haiku"},
            }
        ]

    def unexpected(*args, **kwargs):
        raise AssertionError("unexpected dataset builder")

    fake_prompt_generator.generate_rl_default_verifiers_dataset = (
        fake_generate_rl_default_verifiers_dataset
    )
    fake_prompt_generator.generate_learnable_forms_verifiers_dataset = unexpected
    fake_prompt_generator.generate_traditional_verifiers_dataset = unexpected
    fake_prompt_generator.generate_verifiers_dataset = unexpected
    fake_prompt_generator.resolve_form_selection_mode = lambda: "rl_default"
    monkeypatch.setitem(sys.modules, "prompt_generator", fake_prompt_generator)

    dataset = train_grpo_trl.create_dataset(train_grpo_trl.TrainingArgs(num_prompts=1), {})

    assert calls == ["rl_default"]
    assert dataset[0]["form_name"] == "Haiku"


def test_train_grpo_trl_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "scripts/train_grpo_trl.py", "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--telemetry-every" in result.stdout
