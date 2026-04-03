#!/usr/bin/env python3
"""
Train a Gemma model on well-known poetic forms with the legacy verifiers RL stack.

This script intentionally targets the smaller, local `vf.RLTrainer` path. Prime
Intellect's current upstream guidance recommends `prime-rl` for production
training and treats `vf.RLTrainer` as a lightweight legacy trainer for
educational or experimental runs:
https://docs.primeintellect.ai/verifiers/training

The Abide-side defaults here reflect that recommendation:
- Gemma is the default model family
- a focused well-known-form subset is the default dataset
- verifiers `MaybeThinkParser` is used to normalize completions
- reward telemetry remains enabled so form-specific failures are visible
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import re
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add src and scripts to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from model_profiles import resolve_model_profile
from reward_telemetry import (
    RewardTelemetry,
    bind_reward_telemetry,
    extract_failure_reason,
    flush_reward_telemetry,
)

from abide.forms.catalog import (
    RL_DEFAULT_FORM_NAMES,
    WELL_KNOWN_FORM_NAMES,
    instantiate_form,
    load_form_instances,
    load_rl_default_form_instances,
    load_well_known_form_instances,
)

if TYPE_CHECKING:
    from collections.abc import Callable


DEFAULT_GEMMA_MODEL = "google/gemma-4-E4B-it"
SUPPORTED_FORM_SETS = ("all", "learnable", "rl_default", "traditional", "well_known")


@dataclass
class TrainingConfig:
    """Training configuration for the local verifiers RL trainer."""

    model_name: str = DEFAULT_GEMMA_MODEL
    form_set: str = "well_known"
    single_form: str | None = None

    num_prompts: int = 16000
    eval_prompts: int = 0
    seed: int = 42

    num_train_epochs: int = 1
    rollouts_per_example: int = 8
    batch_size: int = 8
    micro_batch_size: int = 1
    learning_rate: float = 2e-5
    max_seq_len: int = 1536
    max_prompt_len: int = 384
    max_tokens: int = 768
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.1

    output_dir: str = "models/abide_verifiers_gemma4_e4b_well_known"
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 5

    use_wandb: bool = True
    wandb_project: str = "abide-verifiers-grpo"

    vllm_host: str = "0.0.0.0"
    vllm_port: int = 8000
    vllm_timeout: float = 300.0

    max_retries: int = 2
    retry_delay: float = 5.0
    resume_from: str | None = None
    telemetry_every: int = 256


def resolve_default_form_set() -> str:
    """Resolve a form-set default while keeping older env knobs working."""
    explicit = os.environ.get("ABIDE_FORM_SET", "").strip().lower()
    if explicit:
        return explicit
    if os.environ.get("ABIDE_LEARNABLE", "").lower() in {"1", "true", "yes"}:
        return "learnable"
    if os.environ.get("ABIDE_TRADITIONAL", "").lower() in {"1", "true", "yes"}:
        return "traditional"
    return "well_known"


def normalize_generated_poem(text: str) -> str:
    """Normalize raw model output into poem text before verification."""
    cleaned = text

    if "```" in cleaned:
        code_block_match = re.search(r"```(?:\w*\n)?(.*?)```", cleaned, re.DOTALL)
        if code_block_match:
            cleaned = code_block_match.group(1)

    for token in (
        "<think>",
        "</think>",
        "<|thinking|>",
        "<|/thinking|>",
        "<|im_end|>",
        "<|im_start|>",
        "<|end_of_text|>",
        "<|begin_of_text|>",
        "<|endoftext|>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<bos>",
        "<eos>",
    ):
        cleaned = cleaned.replace(token, "")

    preambles = (
        "No explanations.",
        "Here is the poem:",
        "Here's the poem:",
        "The poem:",
    )
    stripped = cleaned.strip()
    for preamble in preambles:
        if stripped.startswith(preamble):
            stripped = stripped[len(preamble) :].strip()
            break

    return stripped


def resolve_forms(config: TrainingConfig) -> dict[str, object]:
    """Load form instances that match the requested training subset."""
    if config.single_form:
        training_profile = config.single_form in set(RL_DEFAULT_FORM_NAMES) | set(
            WELL_KNOWN_FORM_NAMES
        )
        return {
            config.single_form: instantiate_form(
                config.single_form,
                training_profile=training_profile,
            )
        }

    if config.form_set == "well_known":
        return load_well_known_form_instances()
    if config.form_set == "rl_default":
        return load_rl_default_form_instances()
    return load_form_instances()


def resolve_dataset_builder(
    form_set: str,
    *,
    single_form: str | None = None,
) -> Callable[..., Any]:
    """Resolve the prompt-generator dataset builder for a form selection mode."""
    from prompt_generator import (
        generate_learnable_forms_verifiers_dataset,
        generate_rl_default_verifiers_dataset,
        generate_single_form_verifiers_dataset,
        generate_traditional_verifiers_dataset,
        generate_verifiers_dataset,
        generate_well_known_verifiers_dataset,
    )

    if single_form:
        return generate_single_form_verifiers_dataset
    if form_set == "well_known":
        return generate_well_known_verifiers_dataset
    if form_set == "learnable":
        return generate_learnable_forms_verifiers_dataset
    if form_set == "traditional":
        return generate_traditional_verifiers_dataset
    if form_set == "all":
        return generate_verifiers_dataset
    if form_set == "rl_default":
        return generate_rl_default_verifiers_dataset
    valid = ", ".join(SUPPORTED_FORM_SETS)
    raise ValueError(f"Unsupported form set {form_set!r}. Expected one of: {valid}")


def create_dataset(config: TrainingConfig) -> tuple[Any, Any | None]:
    """Create train and optional eval datasets in verifiers-compatible format."""
    builder = resolve_dataset_builder(config.form_set, single_form=config.single_form)
    builder_kwargs = {"num_prompts": config.num_prompts, "seed": config.seed}
    if config.single_form:
        builder_kwargs["form_name"] = config.single_form

    dataset = builder(**builder_kwargs)

    eval_dataset = None
    if config.eval_prompts > 0:
        eval_kwargs = dict(builder_kwargs)
        eval_kwargs["num_prompts"] = config.eval_prompts
        eval_kwargs["seed"] = config.seed + 1
        eval_dataset = builder(**eval_kwargs)

    return dataset, eval_dataset


def create_reward_function(
    forms: dict[str, object],
    *,
    telemetry_every: int = 256,
    use_wandb: bool = True,
):
    """Create a metadata-driven reward function over abide form instances."""
    telemetry = RewardTelemetry(
        label="verifiers",
        emit_every=telemetry_every,
        use_wandb=use_wandb,
    )

    def abide_reward(completion, info=None, parser=None, **kwargs) -> float:
        form_name: str | None = None
        try:
            if parser is not None:
                poem = parser.parse_answer(completion) or ""
            else:
                poem = normalize_generated_poem(str(completion))

            if len(poem.strip()) < 10:
                telemetry.record(
                    None,
                    reward=0.0,
                    passed=False,
                    failure_reason="short completion",
                )
                telemetry.emit()
                return 0.0

            info_dict = info if isinstance(info, dict) else {}
            form_name = info_dict.get("form_name")
            if not form_name:
                telemetry.record(
                    None,
                    reward=0.0,
                    passed=False,
                    failure_reason="missing form_name metadata",
                )
                telemetry.emit()
                return 0.0

            form_instance = forms.get(form_name)
            if form_instance is None:
                telemetry.record(
                    form_name,
                    reward=0.0,
                    passed=False,
                    failure_reason="unknown form",
                )
                telemetry.emit()
                return 0.0

            result = form_instance.verify(poem)
            reward = float(result.score)
            telemetry.record(
                form_name,
                reward=reward,
                passed=bool(getattr(result, "passed", False)),
                failure_reason=extract_failure_reason(result),
            )
            telemetry.emit()
            return reward
        except Exception as exc:
            print(f"[reward error: {exc}]")
            telemetry.record(
                form_name,
                reward=0.0,
                passed=False,
                failure_reason=f"reward error: {type(exc).__name__}",
            )
            telemetry.emit()
            return 0.0

    return bind_reward_telemetry(abide_reward, telemetry)


def create_environment(forms: dict[str, object], config: TrainingConfig):
    """Create a verifiers environment using parser-aware reward plumbing."""
    import verifiers as vf

    install_verifiers_client_compat()
    parser = vf.MaybeThinkParser(extract_fn=normalize_generated_poem)
    dataset, eval_dataset = create_dataset(config)
    reward_fn = create_reward_function(
        forms,
        telemetry_every=config.telemetry_every,
        use_wandb=config.use_wandb,
    )

    rubric = vf.Rubric(funcs=[reward_fn], weights=[1.0], parser=parser)

    def parsed_nonempty_metric(completion, parser, **kwargs) -> float:
        del kwargs
        return float(bool((parser.parse_answer(completion) or "").strip()))

    def parsed_word_count_metric(completion, parser, **kwargs) -> float:
        del kwargs
        return float(len((parser.parse_answer(completion) or "").split()))

    rubric.add_metric(parsed_nonempty_metric)
    rubric.add_metric(parsed_word_count_metric)

    env_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "rubric": rubric,
        "parser": parser,
    }
    if eval_dataset is not None:
        env_kwargs["eval_dataset"] = eval_dataset

    env = vf.SingleTurnEnv(**env_kwargs)
    return env, reward_fn


def install_verifiers_client_compat() -> None:
    """Bridge the verifiers/verifiers-rl client API mismatch.

    `verifiers-rl` still passes a raw `openai.AsyncOpenAI` subclass to
    `Environment.generate`, while newer `verifiers` releases expect a
    `verifiers.clients.Client` wrapper or a `ClientConfig`. Wrap the raw client
    on the Abide side so the isolated legacy runtime remains usable.
    """
    import verifiers.clients as vf_clients
    import verifiers.envs.environment as vf_environment
    from openai import AsyncOpenAI
    from verifiers.clients import OpenAIChatCompletionsClient

    current_resolve = vf_clients.resolve_client
    if getattr(current_resolve, "__name__", "") == "resolve_client_compat":
        return

    def resolve_client_compat(client_or_config):
        if isinstance(client_or_config, AsyncOpenAI):
            return OpenAIChatCompletionsClient(client_or_config)
        return current_resolve(client_or_config)

    vf_clients.resolve_client = resolve_client_compat
    vf_environment.resolve_client = resolve_client_compat


def install_verifiers_rl_kbit_compat() -> None:
    """Teach the legacy trainer how to prepare k-bit PEFT models."""
    import verifiers_rl.rl.trainer.trainer as rl_trainer_module
    from peft import prepare_model_for_kbit_training

    current_prepare = rl_trainer_module.prepare_peft_model
    if getattr(current_prepare, "__name__", "") == "prepare_peft_model_compat":
        return

    def prepare_peft_model_compat(model, peft_config, args):
        if getattr(model, "quantization_method", None) is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=False,
            )
        return current_prepare(model, peft_config, args)

    rl_trainer_module.prepare_peft_model = prepare_peft_model_compat


def install_verifiers_rl_vllm_sync_compat() -> None:
    """Serialize legacy vLLM weight syncs for the current Gemma 4 runtime.

    The legacy `verifiers-rl` client assumes it can pipeline parameter-update
    requests aggressively and let the server process them in the background.
    With the current Gemma 4 + vLLM stack, that can deadlock inside the NCCL
    barrier path during multi-step runs. Keep the protocol conservative:
    broadcast one parameter, wait for the barrier, then wait for the server's
    background queue to drain before returning to the trainer loop.
    """
    import time

    import verifiers_rl.rl.inference.client as inference_client
    from requests.exceptions import Timeout

    current_update = inference_client.VLLMClient.update_named_param
    if getattr(current_update, "__name__", "") == "update_named_param_compat":
        return

    def update_named_param_compat(self, name, weights):
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.server_url}/update_named_param"

        try:
            response = self.session.post(
                url,
                json={"name": name, "dtype": dtype, "shape": shape},
                timeout=300.0,
            )
        except Timeout as exc:
            inference_client.logger.error(
                f"Timeout waiting for server response for {name} after 300s"
            )
            raise Exception(f"Request timeout for {name} after 300s") from exc
        except Exception as exc:
            inference_client.logger.error(f"Error sending request for {name}: {exc}")
            raise

        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

        while self.get_num_background_tasks() > 0:
            time.sleep(0.05)

    inference_client.VLLMClient.update_named_param = update_named_param_compat


def install_transformers_allocator_warmup_compat() -> None:
    """Allow the legacy Gemma 4 load path to skip fragile allocator warmup.

    Recent Transformers releases warm the CUDA caching allocator during
    `from_pretrained()`. On the Gemma 4 E4B QLoRA path, that extra reservation
    can OOM even when the quantized model would otherwise fit. Keep the default
    conservative for this script and let callers opt back in via
    `ABIDE_SKIP_TRANSFORMERS_ALLOCATOR_WARMUP=0`.
    """
    import torch
    import transformers.modeling_utils as modeling_utils

    current_warmup = modeling_utils.caching_allocator_warmup
    if getattr(current_warmup, "__name__", "") == "caching_allocator_warmup_compat":
        return

    def caching_allocator_warmup_compat(model, expanded_device_map, hf_quantizer):
        skip = os.environ.get("ABIDE_SKIP_TRANSFORMERS_ALLOCATOR_WARMUP", "").lower()
        if skip in {"1", "true", "yes"}:
            return None
        try:
            return current_warmup(model, expanded_device_map, hf_quantizer)
        except torch.OutOfMemoryError:
            print(
                "Warning: skipping Transformers allocator warmup after CUDA OOM; "
                "continuing with incremental parameter loading."
            )
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            return None

    modeling_utils.caching_allocator_warmup = caching_allocator_warmup_compat


def import_verifiers_rl_trainer():
    """Import the optional verifiers RL trainer with a repo-specific hint."""
    try:
        from verifiers.rl.trainer import RLConfig, RLTrainer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The legacy verifiers RL trainer is not installed. "
            "Run `bash scripts/prepare_verifiers_runtime.sh` and retry."
        ) from exc
    return RLConfig, RLTrainer


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the highest-step checkpoint under an output directory."""
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def apply_runtime_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Apply verified model-specific runtime defaults when the user did not override them."""
    model_profile = resolve_model_profile(args.model)
    if model_profile.family != "gemma4_e4b":
        return

    if args.rollouts == parser.get_default("rollouts"):
        args.rollouts = 2
    if args.batch_size == parser.get_default("batch_size"):
        args.batch_size = 2
    if args.max_seq_len == parser.get_default("max_seq_len"):
        args.max_seq_len = 512
    if args.max_prompt_len == parser.get_default("max_prompt_len"):
        args.max_prompt_len = 192
    if args.max_tokens == parser.get_default("max_tokens"):
        args.max_tokens = 128
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ABIDE_SKIP_TRANSFORMERS_ALLOCATOR_WARMUP", "1")


def enforce_runtime_preflight(model_name: str) -> None:
    """Fail early when the visible training GPU is obviously too full.

    The legacy Gemma 4 E4B path assumes a mostly dedicated 24 GiB-class GPU.
    If the visible training device is already heavily occupied, the user gets a
    long model-load traceback instead of an actionable message.
    """
    model_profile = resolve_model_profile(model_name)
    if model_profile.family != "gemma4_e4b":
        return

    import torch

    if not torch.cuda.is_available():
        return

    min_free_mib = int(os.environ.get("ABIDE_GEMMA4_E4B_MIN_FREE_MIB", "8192"))
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mib = free_bytes // (1024**2)
    total_mib = total_bytes // (1024**2)

    if free_mib < min_free_mib:
        raise RuntimeError(
            "Insufficient free GPU memory for the legacy Gemma 4 E4B trainer: "
            f"{free_mib} MiB free on visible device 0 out of {total_mib} MiB total. "
            "Free the training GPU, pick a different GPU, or lower "
            "`ABIDE_GEMMA4_E4B_MIN_FREE_MIB` if you intentionally want to try a "
            "more constrained run."
        )


def setup_signal_handlers(trainer) -> None:
    """Save an interrupt checkpoint on SIGINT or SIGTERM."""
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def handler(signum, frame):
        del signum, frame
        print("\nReceived interrupt signal. Saving checkpoint...")
        try:
            trainer.save_model(str(Path(trainer.args.output_dir) / "interrupt-checkpoint"))
        except Exception as exc:  # pragma: no cover - best effort shutdown path
            print(f"Failed to save interrupt checkpoint: {exc}")
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def init_wandb(config: TrainingConfig) -> bool:
    """Initialize Weights & Biases if enabled."""
    if not config.use_wandb:
        return False

    try:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=f"verifiers-{Path(config.model_name).name}",
            config={
                "model": config.model_name,
                "form_set": config.form_set,
                "single_form": config.single_form,
                "num_prompts": config.num_prompts,
                "rollouts_per_example": config.rollouts_per_example,
                "batch_size": config.batch_size,
                "micro_batch_size": config.micro_batch_size,
                "learning_rate": config.learning_rate,
            },
        )
        return True
    except Exception as exc:
        print(f"Warning: failed to initialize wandb: {exc}")
        return False


def finish_wandb(enabled: bool) -> None:
    """Close a wandb run if one was opened."""
    if not enabled:
        return
    with contextlib.suppress(Exception):
        import wandb

        wandb.finish()


def build_rl_config(config: TrainingConfig, *, max_steps: int):
    """Build the legacy verifiers RLConfig with Gemma-friendly defaults."""
    RLConfig, _ = import_verifiers_rl_trainer()
    model_profile = resolve_model_profile(config.model_name)

    target_modules: list[str] | str | None
    if isinstance(model_profile.lora_target_modules, tuple):
        target_modules = list(model_profile.lora_target_modules)
    else:
        target_modules = model_profile.lora_target_modules

    eval_strategy = "steps" if config.eval_prompts > 0 else "no"
    eval_steps = config.eval_steps if config.eval_prompts > 0 else None

    return RLConfig(
        output_dir=config.output_dir,
        run_name=f"abide-verifiers-{int(time.time())}",
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=config.micro_batch_size,
        batch_size=config.batch_size,
        micro_batch_size=config.micro_batch_size,
        rollouts_per_example=config.rollouts_per_example,
        max_seq_len=config.max_seq_len,
        max_prompt_len=config.max_prompt_len,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        zero_truncated_completions=True,
        vllm_server_host=config.vllm_host,
        vllm_server_port=config.vllm_port,
        vllm_server_timeout=config.vllm_timeout,
        bf16=True,
        gradient_checkpointing=True,
        save_steps=config.save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        logging_steps=config.logging_steps,
        report_to="wandb" if config.use_wandb else [],
        remove_unused_columns=False,
        use_lora=True,
        lora_rank=model_profile.default_lora_r,
        lora_alpha=model_profile.default_lora_alpha,
        lora_dropout=model_profile.default_lora_dropout,
        lora_target_modules=target_modules,
        use_liger=False,
    )


def train_with_retry(config: TrainingConfig) -> int:
    """Run verifiers training with simple retry-once checkpoint recovery."""
    _, RLTrainer = import_verifiers_rl_trainer()
    install_verifiers_rl_kbit_compat()
    install_verifiers_rl_vllm_sync_compat()

    print("=" * 60)
    print("Abide Verifiers GRPO Training")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Form set: {config.form_set}")
    if config.single_form:
        print(f"Single form: {config.single_form}")
    print(f"Prompts: {config.num_prompts}")
    print(f"Eval prompts: {config.eval_prompts}")
    print(f"Rollouts per example: {config.rollouts_per_example}")
    print(f"Output: {config.output_dir}")
    print("=" * 60)

    forms = resolve_forms(config)
    print(f"Forms: {len(forms)} ({', '.join(forms)})")

    env, reward_fn = create_environment(forms, config)
    max_steps = max(1, (config.num_prompts // max(config.batch_size, 1)) * config.num_train_epochs)
    rl_config = build_rl_config(config, max_steps=max_steps)
    wandb_enabled = init_wandb(config)

    for attempt in range(config.max_retries):
        try:
            print(f"\nTraining attempt {attempt + 1}/{config.max_retries}")

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = config.resume_from or config.model_name
            model_profile = resolve_model_profile(model_path)
            print(f"Loading model: {model_path}")
            if model_profile.family == "gemma4_e4b":
                enforce_runtime_preflight(model_path)
                install_transformers_allocator_warmup_compat()

            load_kwargs = dict(model_profile.causal_lm_load_kwargs())
            if model_profile.family == "gemma4_e4b":
                from transformers import BitsAndBytesConfig

                load_kwargs.update(
                    {
                        "device_map": "auto",
                        "max_memory": {0: "18GiB", "cpu": "64GiB"},
                        "low_cpu_mem_usage": True,
                        "quantization_config": BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        ),
                    }
                )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                **load_kwargs,
            )
            model.config.use_cache = False
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=model_profile.trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.bos_token_id = tokenizer.bos_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
            generation_config = getattr(model, "generation_config", None)
            if generation_config is not None:
                generation_config.pad_token_id = tokenizer.pad_token_id
                generation_config.bos_token_id = tokenizer.bos_token_id
                generation_config.eos_token_id = tokenizer.eos_token_id

            trainer = RLTrainer(
                model=model,
                env=env,
                args=rl_config,
                processing_class=tokenizer,
            )
            setup_signal_handlers(trainer)

            trainer.train(resume_from_checkpoint=config.resume_from)
            flush_reward_telemetry(reward_fn)

            final_path = Path(config.output_dir) / "final"
            trainer.save_model(str(final_path))
            print(f"Saved final model to {final_path}")

            finish_wandb(wandb_enabled)
            return 0
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            flush_reward_telemetry(reward_fn)
            finish_wandb(wandb_enabled)
            return 1
        except Exception as exc:
            print(f"\nError during training: {exc}")
            traceback.print_exc()
            flush_reward_telemetry(reward_fn)
            with contextlib.suppress(Exception):
                del model
            with contextlib.suppress(Exception):
                gc.collect()
                torch.cuda.empty_cache()

            if attempt >= config.max_retries - 1:
                finish_wandb(wandb_enabled)
                return 1

            latest_checkpoint = find_latest_checkpoint(Path(config.output_dir))
            if latest_checkpoint is not None:
                config.resume_from = str(latest_checkpoint)
                print(f"Retrying from checkpoint {config.resume_from} in {config.retry_delay}s...")
            else:
                print(f"Retrying from base model in {config.retry_delay}s...")
            time.sleep(config.retry_delay)

    finish_wandb(wandb_enabled)
    return 1


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train a Gemma poetry model with the legacy verifiers RL trainer.",
        epilog=(
            "This script is for local experimental runs. Upstream verifiers guidance now "
            "recommends prime-rl for production training."
        ),
    )
    parser.add_argument("--model", default=os.environ.get("ABIDE_MODEL", DEFAULT_GEMMA_MODEL))
    parser.add_argument(
        "--form-set",
        choices=SUPPORTED_FORM_SETS,
        default=resolve_default_form_set(),
        help="Curated prompt/form subset to train against.",
    )
    parser.add_argument(
        "--single-form",
        default=os.environ.get("ABIDE_SINGLE_FORM") or None,
        help="Override the form set and train on a single named form.",
    )
    parser.add_argument("--prompts", type=int, default=16000, help="Number of training prompts.")
    parser.add_argument(
        "--eval-prompts",
        type=int,
        default=0,
        help="Optional number of held-out prompts for periodic evaluation.",
    )
    parser.add_argument("--rollouts", type=int, default=8, help="Rollouts per example.")
    parser.add_argument("--batch-size", type=int, default=8, help="Total rollout batch size.")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Per-device micro batch size.",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--max-seq-len", type=int, default=1536, help="Max train sequence length.")
    parser.add_argument("--max-prompt-len", type=int, default=384, help="Max prompt length.")
    parser.add_argument("--max-tokens", type=int, default=768, help="Generation max tokens.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling value.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Sampling repetition penalty.",
    )
    parser.add_argument(
        "--output",
        default="models/abide_verifiers_gemma4_e4b_well_known",
        help="Output directory.",
    )
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint cadence.")
    parser.add_argument("--eval-steps", type=int, default=50, help="Eval cadence when enabled.")
    parser.add_argument("--resume", type=str, help="Resume from a checkpoint path.")
    parser.add_argument("--host", default="0.0.0.0", help="vf-vllm host.")
    parser.add_argument("--port", type=int, default=8000, help="vf-vllm port.")
    parser.add_argument(
        "--vllm-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for the vLLM server.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--telemetry-every",
        type=int,
        default=256,
        help="Emit aggregate reward telemetry every N scored samples.",
    )
    args = parser.parse_args()
    apply_runtime_defaults(args, parser)

    config = TrainingConfig(
        model_name=args.model,
        form_set=args.form_set,
        single_form=args.single_form,
        num_prompts=args.prompts,
        eval_prompts=args.eval_prompts,
        seed=args.seed,
        num_train_epochs=args.epochs,
        rollouts_per_example=args.rollouts,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        max_prompt_len=args.max_prompt_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_dir=args.output,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        use_wandb=not args.no_wandb,
        resume_from=args.resume,
        vllm_host=args.host,
        vllm_port=args.port,
        vllm_timeout=args.vllm_timeout,
        telemetry_every=args.telemetry_every,
    )
    return train_with_retry(config)


if __name__ == "__main__":
    sys.exit(main())
