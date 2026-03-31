"""Integration tests for the TRL GRPO training helpers."""

from __future__ import annotations

import sys
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

    def fake_generate_learnable_forms_verifiers_dataset(
        *, num_prompts: int, seed: int
    ) -> list[dict]:
        del num_prompts, seed
        return [
            {
                "prompt": [{"role": "user", "content": "Write a PindaricOde poem about winter."}],
                "info": {"form_name": "PindaricOde"},
            }
        ]

    fake_prompt_generator.generate_learnable_forms_verifiers_dataset = (
        fake_generate_learnable_forms_verifiers_dataset
    )
    monkeypatch.setitem(sys.modules, "prompt_generator", fake_prompt_generator)

    dataset = train_grpo_trl.create_dataset(train_grpo_trl.TrainingArgs(num_prompts=1), {})

    row = dataset[0]
    assert row["form_name"] == "PindaricOde"
    assert row["prompt"][0]["content"][0]["text"] == "Write a PindaricOde poem about winter."
