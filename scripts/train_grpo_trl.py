#!/usr/bin/env python3
"""
GRPO training script using TRL's GRPOTrainer with proper KL regularization.

This script uses TRL's native GRPOTrainer instead of the verifiers library,
which allows us to add the beta (KL coefficient) parameter that prevents
policy collapse.

Key differences from verifiers-based training:
- Uses TRL's GRPOTrainer with GRPOConfig
- Includes beta parameter for KL regularization (default 0.04, DeepSeek uses 0.001)
- Uses TRL's vLLM server integration
- Logs KL divergence when beta > 0

Usage:
    # Start trl vllm server on GPU 1 first (in separate terminal):
    CUDA_VISIBLE_DEVICES=1 uv run trl vllm-serve --model google/gemma-4-E4B-it --port 8000

    # Then run training on GPU 0:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_grpo_trl.py

    # With custom beta:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_grpo_trl.py --beta 0.01
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from model_profiles import GEMMA_4_E4B_PROFILE, resolve_model_profile
from reward_telemetry import (
    RewardTelemetry,
    bind_reward_telemetry,
    extract_failure_reason,
    extract_form_names_from_metadata,
    flush_reward_telemetry,
)

from abide.forms.catalog import RL_DEFAULT_FORM_NAMES, load_form_instances

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass
class TrainingArgs:
    """Training configuration."""

    # Model
    model_name: str = "google/gemma-4-E4B-it"
    model_path: str | None = None

    # Dataset
    num_prompts: int = 50000
    seed: int = 42

    # Training hyperparameters
    batch_size: int = GEMMA_4_E4B_PROFILE.canary_batch_size
    num_generations: int = GEMMA_4_E4B_PROFILE.canary_num_generations
    learning_rate: float = GEMMA_4_E4B_PROFILE.canary_learning_rate
    max_completion_length: int = 1024
    max_prompt_length: int = 512

    # KL regularization - THE KEY ADDITION!
    beta: float = GEMMA_4_E4B_PROFILE.canary_beta

    # Clipping
    epsilon: float = 0.2  # PPO-style clipping

    # Output
    output_dir: str = GEMMA_4_E4B_PROFILE.canary_output_dir
    save_steps: int = 50
    logging_steps: int = 10

    # vLLM
    vllm_port: int = 8000
    vllm_host: str = "0.0.0.0"

    # LoRA
    use_lora: bool = True
    lora_r: int = GEMMA_4_E4B_PROFILE.default_lora_r
    lora_alpha: int = GEMMA_4_E4B_PROFILE.default_lora_alpha
    lora_dropout: float = GEMMA_4_E4B_PROFILE.default_lora_dropout

    # Wandb
    wandb_project: str = "abide-grpo"
    use_wandb: bool = True
    telemetry_every: int = 256
    telemetry_jsonl: str | None = None
    use_vllm: bool = GEMMA_4_E4B_PROFILE.canary_use_vllm
    resume_from_checkpoint: str | None = None
    auto_resume: bool = False


def get_forms() -> dict[str, object]:
    """Load all training forms from abide.forms."""
    return load_form_instances()


def extract_text_recursive(obj) -> str:
    """Recursively extract text from nested structures.

    Handles any depth of nesting for TRL multimodal formats.
    Always returns a string - never a list.
    """
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        # Could be {"type": "text", "text": "..."} or {"role": "user", "content": ...}
        if obj.get("type") == "text":
            text_val = obj.get("text", "")
            # text_val could itself be nested
            return extract_text_recursive(text_val)
        if "content" in obj:
            return extract_text_recursive(obj["content"])
        if "text" in obj:
            return extract_text_recursive(obj["text"])
        return ""

    if isinstance(obj, list):
        parts = []
        for item in obj:
            text = extract_text_recursive(item)
            # Ensure text is always a string (defensive)
            if isinstance(text, str) and text:
                parts.append(text)
            elif text:
                # If somehow not a string, convert it
                parts.append(str(text))
        return " ".join(parts)

    # Fallback: convert to string
    return str(obj)


def get_completion_text(completion) -> str:
    """Extract poem text from completion, stripping model tokens.

    TRL may pass completions as:
    - str: plain text
    - list: multimodal format, can be arbitrarily nested
    - dict: {"type": "text", "text": "..."}
    """
    # Use recursive extraction to handle any format
    text = extract_text_recursive(completion)

    # Strip ChatML tokens
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = text.replace("<|end_of_text|>", "").replace("<|begin_of_text|>", "")
    text = text.replace("<|endoftext|>", "")

    # Strip Gemma tokens
    text = text.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
    text = text.replace("<bos>", "").replace("<eos>", "")

    # Strip markdown code blocks
    code_block_match = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1)

    # Strip common preambles
    preambles = [
        "No explanations.",
        "Here is the poem:",
        "Here's the poem:",
        "The poem:",
    ]
    for preamble in preambles:
        if text.strip().startswith(preamble):
            text = text.strip()[len(preamble) :]

    return text.strip()


def extract_prompt_text(prompt) -> str:
    """Extract text from prompt, handling various TRL formats.

    TRL may pass prompts as:
    - str: plain text
    - list of dicts: chat format [{"role": "user", "content": ...}]
    - content can be str or list of {"type": "text", "text": "..."}
    - Can be arbitrarily nested
    """
    return extract_text_recursive(prompt)


def create_reward_function(
    forms: dict[str, object],
    *,
    telemetry_every: int = 256,
    use_wandb: bool = True,
    telemetry_jsonl: str | None = None,
):
    """Create reward function for TRL.

    TRL's reward functions receive:
    - completions: list of completion strings (or multimodal lists)
    - prompts: list of prompt strings (or chat format lists)
    - **kwargs: additional metadata
    """
    _call_count = [0]
    _debug_printed = [False]
    telemetry = RewardTelemetry(
        label="trl",
        emit_every=telemetry_every,
        use_wandb=use_wandb,
        jsonl_path=telemetry_jsonl,
    )

    def reward_fn(completions: list, prompts: list, **kwargs: Any) -> list[float]:
        """Score completions against their target forms."""
        rewards = []
        form_names = extract_form_names_from_metadata(kwargs, len(completions))

        # Debug: print structure once when it changes
        if not _debug_printed[0] and _call_count[0] > 100:
            _debug_printed[0] = True
            if completions:
                print(f"[DEBUG step {_call_count[0]}] completion type: {type(completions[0])}")
                if isinstance(completions[0], list) and completions[0]:
                    print(f"[DEBUG] completion[0][0] type: {type(completions[0][0])}")
                    print(f"[DEBUG] completion[0][0] value: {completions[0][0]}")
            if prompts:
                print(f"[DEBUG step {_call_count[0]}] prompt type: {type(prompts[0])}")
                if isinstance(prompts[0], list) and prompts[0]:
                    print(f"[DEBUG] prompt[0][0] type: {type(prompts[0][0])}")

        for _i, (completion, prompt, form_name) in enumerate(zip(completions, prompts, form_names)):
            _call_count[0] += 1
            try:
                poem = get_completion_text(completion)

                # Ensure these are strings
                if not isinstance(poem, str):
                    poem = str(poem) if poem else ""

                if not poem or len(poem.strip()) < 10:
                    rewards.append(0.0)
                    telemetry.record(
                        form_name,
                        reward=0.0,
                        passed=False,
                        failure_reason="short completion",
                    )
                    telemetry.emit()
                    continue

                if not form_name:
                    if _call_count[0] <= 3:
                        prompt_text = extract_prompt_text(prompt)
                        print(
                            f"[DEBUG] Missing form_name metadata for prompt: {prompt_text[:100]}..."
                        )
                    rewards.append(0.0)
                    telemetry.record(
                        None,
                        reward=0.0,
                        passed=False,
                        failure_reason="missing form_name metadata",
                    )
                    telemetry.emit()
                    continue

                form_instance = forms.get(form_name)
                if form_instance is None:
                    rewards.append(0.0)
                    telemetry.record(
                        form_name,
                        reward=0.0,
                        passed=False,
                        failure_reason="unknown form",
                    )
                    telemetry.emit()
                    continue

                result = form_instance.verify(poem)
                reward = float(result.score)
                rewards.append(reward)
                telemetry.record(
                    form_name,
                    reward=reward,
                    passed=bool(getattr(result, "passed", False)),
                    failure_reason=extract_failure_reason(result),
                )
                telemetry.emit()

                if _call_count[0] <= 3:
                    print(f"[DEBUG] {form_name}: score={result.score:.3f}")

            except Exception as e:
                import traceback

                print(f"[reward error: {e}]")
                if _call_count[0] < 200:
                    traceback.print_exc()
                rewards.append(0.0)
                telemetry.record(
                    form_name,
                    reward=0.0,
                    passed=False,
                    failure_reason=f"reward error: {type(e).__name__}",
                )
                telemetry.emit()

        return rewards

    return bind_reward_telemetry(reward_fn, telemetry)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the numerically latest checkpoint directory, if present."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints: list[tuple[int, Path]] = []
    for candidate in output_path.iterdir():
        if not candidate.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", candidate.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), candidate))

    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resolve_resume_checkpoint(
    output_dir: str | Path,
    *,
    explicit_path: str | None,
    auto_resume: bool,
) -> str | None:
    """Resolve the checkpoint path used to resume training."""
    if explicit_path:
        checkpoint_path = Path(explicit_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path)

    if not auto_resume:
        return None

    latest_checkpoint = find_latest_checkpoint(output_dir)
    if latest_checkpoint is None:
        return None
    return str(latest_checkpoint)


def create_dataset(args: TrainingArgs, forms: dict[str, object]) -> Dataset:
    """Create training dataset using prompt generator."""
    from datasets import Dataset
    from prompt_generator import (
        generate_learnable_forms_verifiers_dataset,
        generate_rl_default_verifiers_dataset,
        generate_traditional_verifiers_dataset,
        generate_verifiers_dataset,
        resolve_form_selection_mode,
    )

    form_mode = resolve_form_selection_mode()

    if form_mode == "learnable":
        print("Using LEARNABLE forms only (top 10 with highest GRPO signal)")
        raw_dataset = generate_learnable_forms_verifiers_dataset(
            num_prompts=args.num_prompts,
            seed=args.seed,
        )
    elif form_mode == "traditional":
        print("Using TRADITIONAL forms only")
        raw_dataset = generate_traditional_verifiers_dataset(
            num_prompts=args.num_prompts,
            seed=args.seed,
        )
    elif form_mode == "all":
        print("Using ALL instantiable forms")
        raw_dataset = generate_verifiers_dataset(
            num_prompts=args.num_prompts,
            seed=args.seed,
        )
    else:
        print("Defaulting to the curated RL-default forms for this TRL experiment")
        raw_dataset = generate_rl_default_verifiers_dataset(
            num_prompts=args.num_prompts,
            seed=args.seed,
        )

    # Convert to HuggingFace Dataset format expected by TRL
    # TRL expects a "prompt" column with chat messages
    # For vLLM multimodal, content must be list of {"type": "text", "text": "..."} dicts
    prompts = []
    form_names = []
    for item in raw_dataset:
        prompt_text = extract_prompt_text(item["prompt"])
        form_info = item.get("info", {})
        form_name = form_info.get("form_name") if isinstance(form_info, dict) else None
        if not isinstance(form_name, str) or not form_name:
            raise ValueError(f"Dataset item is missing info.form_name: {item}")

        prompts.append([{"role": "user", "content": [{"type": "text", "text": prompt_text}]}])
        form_names.append(form_name)

    dataset = Dataset.from_dict({"prompt": prompts, "form_name": form_names})
    print(f"Created dataset with {len(dataset)} prompts")

    return dataset


def create_grpo_config(training_args: TrainingArgs, *, max_steps: int):
    from trl import GRPOConfig

    config_kwargs: dict[str, Any] = {
        "output_dir": training_args.output_dir,
        "beta": training_args.beta,
        "epsilon": training_args.epsilon,
        "num_generations": training_args.num_generations,
        "max_completion_length": training_args.max_completion_length,
        "temperature": 0.7,
        "top_p": 0.95,
        "learning_rate": training_args.learning_rate,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": training_args.batch_size,
        "generation_batch_size": max(training_args.batch_size, training_args.num_generations),
        "max_steps": max_steps,
        "logging_steps": training_args.logging_steps,
        "save_steps": training_args.save_steps,
        "save_total_limit": 5,
        "report_to": ["wandb"] if training_args.use_wandb else [],
        "run_name": f"grpo-trl-beta{training_args.beta}",
        "gradient_checkpointing": True,
        "bf16": True,
        "loss_type": "dapo",
        "mask_truncated_completions": False,
        "seed": training_args.seed,
        "use_vllm": training_args.use_vllm,
    }
    if training_args.use_vllm:
        config_kwargs.update(
            {
                "vllm_mode": "server",
                "vllm_server_host": training_args.vllm_host,
                "vllm_server_port": training_args.vllm_port,
            }
        )
    return GRPOConfig(**config_kwargs)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GRPO training with TRL (includes KL regularization)"
    )
    parser.add_argument("--model", default=TrainingArgs.model_name, help="Model name")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local model artifact path used for loading weights/tokenizer",
    )
    parser.add_argument("--prompts", type=int, default=50000, help="Number of prompts")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TrainingArgs.batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=TrainingArgs.num_generations,
        help="Generations per prompt (rollouts)",
    )
    parser.add_argument(
        "--lr", type=float, default=TrainingArgs.learning_rate, help="Learning rate"
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=TrainingArgs.max_completion_length,
        help="Maximum completion length in tokens",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=TrainingArgs.max_prompt_length,
        help="Maximum prompt length in tokens",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=TrainingArgs.beta,
        help="KL coefficient (0.001-0.04 typical)",
    )
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping epsilon")
    parser.add_argument("--output", default=TrainingArgs.output_dir, help="Output directory")
    parser.add_argument("--save-steps", type=int, default=50, help="Save every N steps")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank override (defaults to the selected model profile)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha override (defaults to the selected model profile)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="LoRA dropout override (defaults to the selected model profile)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--use-vllm", action="store_true", help="Use a vLLM server for generation")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM and generate locally")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume from an explicit checkpoint path",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from the latest checkpoint in the output directory when present",
    )
    parser.add_argument(
        "--telemetry-every",
        type=int,
        default=256,
        help="Emit aggregate reward telemetry every N scored samples",
    )
    parser.add_argument(
        "--telemetry-jsonl",
        default=None,
        help="Optional JSONL file for persisted reward telemetry snapshots",
    )
    args = parser.parse_args()
    if args.use_vllm and args.no_vllm:
        parser.error("--use-vllm and --no-vllm are mutually exclusive")

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_SILENT"] = "true"

    # Check for env override
    if os.environ.get("ABIDE_MODEL"):
        args.model = os.environ["ABIDE_MODEL"]

    model_profile = resolve_model_profile(args.model)

    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOTrainer
    except ModuleNotFoundError as exc:
        missing = exc.name or "training dependency"
        raise SystemExit(
            "Missing optional training dependency "
            f"{missing!r}. Install training extras with `uv sync --extra training`."
        ) from exc

    use_vllm = TrainingArgs.use_vllm
    if args.use_vllm:
        use_vllm = True
    elif args.no_vllm:
        use_vllm = False

    training_args = TrainingArgs(
        model_name=args.model,
        model_path=args.model_path,
        num_prompts=args.prompts,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        epsilon=args.epsilon,
        output_dir=args.output,
        save_steps=args.save_steps,
        vllm_port=args.port,
        use_wandb=not args.no_wandb,
        seed=args.seed,
        telemetry_every=args.telemetry_every,
        telemetry_jsonl=args.telemetry_jsonl,
        use_vllm=use_vllm,
        lora_r=args.lora_r if args.lora_r is not None else model_profile.default_lora_r,
        lora_alpha=(
            args.lora_alpha if args.lora_alpha is not None else model_profile.default_lora_alpha
        ),
        lora_dropout=(
            args.lora_dropout
            if args.lora_dropout is not None
            else model_profile.default_lora_dropout
        ),
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume=args.auto_resume,
    )

    print("=" * 60)
    print("TRL GRPO Training (with KL regularization)")
    print("=" * 60)
    print(f"Model: {training_args.model_name}")
    if training_args.model_path is not None:
        print(f"Model artifacts: {training_args.model_path}")
    print(f"Prompts: {training_args.num_prompts}")
    print(f"Generations per prompt: {training_args.num_generations}")
    print(f"Batch size: {training_args.batch_size}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Beta (KL coef): {training_args.beta}")
    print(f"Epsilon (clip): {training_args.epsilon}")
    print(f"Generation backend: {'vLLM server' if training_args.use_vllm else 'local model'}")
    print(f"Output: {training_args.output_dir}")
    if training_args.telemetry_jsonl is not None:
        print(f"Telemetry JSONL: {training_args.telemetry_jsonl}")
    print("=" * 60)

    # Load forms
    from prompt_generator import resolve_form_selection_mode

    form_mode = resolve_form_selection_mode()
    if form_mode == "rl_default":
        forms = load_form_instances(list(RL_DEFAULT_FORM_NAMES), training_profile=True)
    else:
        forms = get_forms()
    print(f"Loaded {len(forms)} forms")

    # Create dataset
    dataset = create_dataset(training_args, forms)

    # Create reward function
    reward_fn = create_reward_function(
        forms,
        telemetry_every=training_args.telemetry_every,
        use_wandb=training_args.use_wandb,
        telemetry_jsonl=training_args.telemetry_jsonl,
    )

    # Compute max_steps
    max_steps = training_args.num_prompts // training_args.batch_size
    print(f"Max steps: {max_steps}")

    # Configure TRL GRPO
    grpo_config = create_grpo_config(training_args, max_steps=max_steps)
    model_load_target = training_args.model_path or training_args.model_name

    # Load model
    print(f"Loading model: {model_load_target}")
    model = AutoModelForCausalLM.from_pretrained(
        model_load_target,
        torch_dtype=torch.bfloat16,
        **model_profile.causal_lm_load_kwargs(),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_load_target,
        trust_remote_code=model_profile.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = None
    if training_args.use_lora:
        peft_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=model_profile.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"Using LoRA: r={training_args.lora_r}, alpha={training_args.lora_alpha}")

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )

    print("\nStarting TRL GRPO training...")
    print(f"KL regularization enabled with beta={training_args.beta}")
    print("=" * 60)

    resume_checkpoint = resolve_resume_checkpoint(
        training_args.output_dir,
        explicit_path=training_args.resume_from_checkpoint,
        auto_resume=training_args.auto_resume,
    )
    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")

    # Train
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    finally:
        flush_reward_telemetry(reward_fn)

    # Save final model
    final_path = Path(training_args.output_dir) / "final"
    trainer.save_model(final_path)
    print(f"\nSaved final model to {final_path}")

    print("\n" + "=" * 60)
    print("TRL GRPO Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
