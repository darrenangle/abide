"""Integration tests for the legacy verifiers GRPO training helpers."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from scripts import model_profiles, train_grpo


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


def test_train_grpo_defaults_align_with_gemma4_e4b_profile() -> None:
    defaults = train_grpo.TrainingConfig()
    profile = model_profiles.resolve_model_profile(defaults.model_name)

    assert defaults.model_name == "google/gemma-4-E4B-it"
    assert profile.family == "gemma4_e4b"
    assert defaults.batch_size == profile.canary_batch_size
    assert defaults.output_dir == "models/abide_verifiers_gemma4_e4b_well_known"


def test_install_verifiers_client_compat_wraps_async_openai(monkeypatch) -> None:
    import openai
    import verifiers.clients as vf_clients
    import verifiers.envs.environment as vf_environment

    class WrappedClient:
        def __init__(self, client) -> None:
            self.client = client

    monkeypatch.setattr(vf_clients, "OpenAIChatCompletionsClient", WrappedClient)
    monkeypatch.setattr(
        vf_clients,
        "resolve_client",
        lambda client_or_config: ("base", client_or_config),
    )
    monkeypatch.setattr(
        vf_environment,
        "resolve_client",
        lambda client_or_config: ("env", client_or_config),
    )

    train_grpo.install_verifiers_client_compat()

    client = openai.AsyncOpenAI(api_key="EMPTY", base_url="http://127.0.0.1:1/v1")
    wrapped = vf_clients.resolve_client(client)

    assert isinstance(wrapped, WrappedClient)
    assert vf_environment.resolve_client is vf_clients.resolve_client
    assert vf_clients.resolve_client("config") == ("base", "config")


def test_install_verifiers_rl_kbit_compat_patches_prepare(monkeypatch) -> None:
    from types import ModuleType, SimpleNamespace

    marker = object()

    def prepare_model_for_kbit_training(model, *, use_gradient_checkpointing):
        assert use_gradient_checkpointing is False
        return marker

    def prepare_peft_model(model, peft_config, args):
        return ("prepared", model, peft_config, args)

    trainer_module = ModuleType("verifiers_rl.rl.trainer.trainer")
    trainer_module.prepare_peft_model = prepare_peft_model
    trainer_pkg = ModuleType("verifiers_rl.rl.trainer")
    trainer_pkg.trainer = trainer_module
    rl_pkg = ModuleType("verifiers_rl.rl")
    rl_pkg.trainer = trainer_pkg
    root_pkg = ModuleType("verifiers_rl")
    root_pkg.rl = rl_pkg

    peft_module = ModuleType("peft")
    peft_module.prepare_model_for_kbit_training = prepare_model_for_kbit_training

    monkeypatch.setitem(sys.modules, "verifiers_rl", root_pkg)
    monkeypatch.setitem(sys.modules, "verifiers_rl.rl", rl_pkg)
    monkeypatch.setitem(sys.modules, "verifiers_rl.rl.trainer", trainer_pkg)
    monkeypatch.setitem(sys.modules, "verifiers_rl.rl.trainer.trainer", trainer_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    train_grpo.install_verifiers_rl_kbit_compat()

    quantized_model = SimpleNamespace(quantization_method="bitsandbytes")
    result = trainer_module.prepare_peft_model(quantized_model, "cfg", "args")
    assert result == ("prepared", marker, "cfg", "args")

    plain_model = SimpleNamespace(quantization_method=None)
    result = trainer_module.prepare_peft_model(plain_model, "cfg", "args")
    assert result == ("prepared", plain_model, "cfg", "args")


def test_install_verifiers_rl_vllm_sync_compat_waits_for_background_tasks(monkeypatch) -> None:
    client_module = ModuleType("verifiers_rl.rl.inference.client")
    inference_pkg = ModuleType("verifiers_rl.rl.inference")
    rl_pkg = ModuleType("verifiers_rl.rl")
    root_pkg = ModuleType("verifiers_rl")

    class Logger:
        def error(self, *_args, **_kwargs) -> None:
            return None

    class VLLMClient:
        def __init__(self) -> None:
            self.server_url = "http://127.0.0.1:8000"
            self.rank = 1
            self.background_counts = [1, 0]
            self.post_calls = []
            self.broadcast_calls = []
            self.barrier_calls = 0
            self.session = SimpleNamespace(post=self._post)
            self.pynccl_comm = SimpleNamespace(
                broadcast=self._broadcast,
                group=SimpleNamespace(barrier=self._barrier),
            )

        def update_named_param(self, name, weights) -> None:
            del name, weights

        def _post(self, url, json, timeout):
            self.post_calls.append((url, json, timeout))
            return SimpleNamespace(status_code=200, text="ok")

        def _broadcast(self, weights, *, src):
            self.broadcast_calls.append((tuple(weights.shape), src))

        def _barrier(self) -> None:
            self.barrier_calls += 1

        def get_num_background_tasks(self) -> int:
            return self.background_counts.pop(0)

    client_module.VLLMClient = VLLMClient
    client_module.logger = Logger()
    inference_pkg.client = client_module
    rl_pkg.inference = inference_pkg
    root_pkg.rl = rl_pkg

    monkeypatch.setitem(sys.modules, "verifiers_rl", root_pkg)
    monkeypatch.setitem(sys.modules, "verifiers_rl.rl", rl_pkg)
    monkeypatch.setitem(sys.modules, "verifiers_rl.rl.inference", inference_pkg)
    monkeypatch.setitem(sys.modules, "verifiers_rl.rl.inference.client", client_module)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    train_grpo.install_verifiers_rl_vllm_sync_compat()

    client = VLLMClient()
    client_module.VLLMClient.update_named_param(client, "weight", torch.zeros(2))

    assert client.post_calls == [
        (
            "http://127.0.0.1:8000/update_named_param",
            {"name": "weight", "dtype": "torch.float32", "shape": (2,)},
            300.0,
        )
    ]
    assert client.broadcast_calls == [((2,), 1)]
    assert client.barrier_calls == 1
    assert client.background_counts == []


def test_install_transformers_allocator_warmup_compat_skips_when_enabled(monkeypatch) -> None:
    modeling_utils = ModuleType("transformers.modeling_utils")
    calls: list[str] = []

    def caching_allocator_warmup(model, expanded_device_map, hf_quantizer):
        del model, expanded_device_map, hf_quantizer
        calls.append("warmup")

    modeling_utils.caching_allocator_warmup = caching_allocator_warmup
    transformers_pkg = ModuleType("transformers")
    transformers_pkg.modeling_utils = modeling_utils

    monkeypatch.setitem(sys.modules, "transformers", transformers_pkg)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)
    monkeypatch.setenv("ABIDE_SKIP_TRANSFORMERS_ALLOCATOR_WARMUP", "1")

    train_grpo.install_transformers_allocator_warmup_compat()
    modeling_utils.caching_allocator_warmup(None, {}, None)

    assert calls == []


def test_enforce_runtime_preflight_rejects_overfull_visible_gpu(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info(_index):
            return (7 * 1024**3, 24 * 1024**3)

    monkeypatch.setattr(torch, "cuda", FakeCuda())
    monkeypatch.delenv("ABIDE_GEMMA4_E4B_MIN_FREE_MIB", raising=False)

    try:
        train_grpo.enforce_runtime_preflight("google/gemma-4-E4B-it")
    except RuntimeError as exc:
        assert "Insufficient free GPU memory" in str(exc)
        assert "visible device 0" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_apply_runtime_defaults_constrains_gemma4_e4b(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--max-prompt-len", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=768)

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    args = argparse.Namespace(
        model="google/gemma-4-E4B-it",
        rollouts=8,
        batch_size=8,
        max_seq_len=1536,
        max_prompt_len=384,
        max_tokens=768,
    )
    train_grpo.apply_runtime_defaults(args, parser)

    assert args.rollouts == 2
    assert args.batch_size == 2
    assert args.max_seq_len == 512
    assert args.max_prompt_len == 192
    assert args.max_tokens == 128
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert os.environ["ABIDE_SKIP_TRANSFORMERS_ALLOCATOR_WARMUP"] == "1"


def test_apply_runtime_defaults_preserves_explicit_overrides(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--max-prompt-len", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=768)

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    args = argparse.Namespace(
        model="google/gemma-4-E4B-it",
        rollouts=4,
        batch_size=3,
        max_seq_len=640,
        max_prompt_len=256,
        max_tokens=160,
    )
    train_grpo.apply_runtime_defaults(args, parser)

    assert args.rollouts == 4
    assert args.batch_size == 3
    assert args.max_seq_len == 640
    assert args.max_prompt_len == 256
    assert args.max_tokens == 160
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


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
