#!/usr/bin/env python3
"""
GRPO training script for Baguettotron.

Baguettotron-specific:
- Uses <think>...</think> reasoning traces
- ChatML format: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>
- REQUIRES </think> in output - zero reward otherwise
- Tuned inference params: max_tokens=2048, rep_penalty=1.2

Usage:
    # Start vLLM server first:
    CUDA_VISIBLE_DEVICES=1 uv run trl vllm-serve \\
        --model /home/darren/10k-poems/models/baguettotron_sft/final \\
        --port 8000 --max_model_len 4096

    # Then run training:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_grpo_baguettotron.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# CRITICAL FIX: Monkey-patch TRL's GRPOTrainer to NOT skip special tokens
# TRL hardcodes skip_special_tokens=True which strips </think> from output
# We need </think> to be preserved for our reward function
_original_batch_decode = None


def _patched_batch_decode(self, token_ids, skip_special_tokens=True, **kwargs):
    """Patched batch_decode that preserves special tokens for reasoning models."""
    # ALWAYS preserve special tokens - we need </think>
    return _original_batch_decode(self, token_ids, skip_special_tokens=False, **kwargs)


# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from reward_telemetry import (  # noqa: E402
    RewardTelemetry,
    bind_reward_telemetry,
    extract_failure_reason,
    extract_form_names_from_metadata,
    flush_reward_telemetry,
)

from abide.forms.catalog import load_form_instances  # noqa: E402

if TYPE_CHECKING:
    from datasets import Dataset

# ============================================================
# BAGUETTOTRON CONFIGURATION
# ============================================================
MODEL_PATH = "/home/darren/10k-poems/models/baguettotron_sft/final"

# Inference params (tuned via param_search.py Dec 2025)
MAX_COMPLETION_LENGTH = 2048  # Keep at 2048 - 3072 causes OOM
TEMPERATURE = 0.6
TOP_P = 0.95
# Note: repetition_penalty handled by vLLM server, not here

# Training params
NUM_PROMPTS = 10000  # Reduced - Gemma converged in ~400 steps
BATCH_SIZE = 16
NUM_GENERATIONS = 16
LEARNING_RATE = 5e-5
BETA = 0.01  # KL coefficient - lowered from 0.04 to allow more divergence

# LoRA
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05


def get_forms(form_names: list[str] | None = None) -> dict[str, object]:
    """Load forms through the shared catalog."""
    return load_form_instances(form_names)


def extract_poem_from_completion(completion) -> tuple[str, bool]:
    """Extract poem from Baguettotron completion.

    Baguettotron format (reasoning model):
        <think>reasoning here</think>
        actual poem here
        <|im_end|>

    Returns:
        (poem_text, has_valid_format)

    If </think> is missing, returns ("", False) - model gets 0 reward.
    """
    # Handle TRL's various formats
    text = completion
    if isinstance(text, list):
        if len(text) > 0 and isinstance(text[0], dict):
            text = text[-1].get("content", "")
            if isinstance(text, list):
                text = " ".join(t.get("text", "") if isinstance(t, dict) else str(t) for t in text)
        elif len(text) > 0:
            text = "".join(str(x) for x in text)
    text = str(text)

    # REQUIRE </think> - this is a reasoning model!
    if "</think>" not in text:
        return "", False

    # Extract text AFTER </think>
    poem = text.split("</think>", 1)[-1]

    # Clean up end tokens and padding
    poem = poem.replace("<|im_end>", "").replace("<|im_end|>", "")
    poem = poem.replace("[PAD]", "")  # Strip padding tokens

    return poem.strip(), True


def extract_prompt_text(prompt) -> str:
    """Extract text from TRL prompt format (handles deeply nested structures)."""

    def extract_strings(obj) -> list[str]:
        """Recursively extract all strings from nested structure."""
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, dict):
            result = []
            for key in ["text", "content"]:
                if key in obj:
                    result.extend(extract_strings(obj[key]))
            return result
        if isinstance(obj, list):
            result = []
            for item in obj:
                result.extend(extract_strings(item))
            return result
        return [str(obj)]

    strings = extract_strings(prompt)
    return " ".join(s for s in strings if s)


def create_reward_function(
    forms: dict[str, object],
    *,
    telemetry_every: int = 256,
    use_wandb: bool = True,
):
    """Create reward function for Baguettotron.

    - Returns 0.0 if completion doesn't have </think> (bad format)
    - Returns 0.0 if poem is too short
    - Returns form.verify(poem).score otherwise
    """
    _call_count = [0]
    telemetry = RewardTelemetry(
        label="baguettotron",
        emit_every=telemetry_every,
        use_wandb=use_wandb,
    )

    def reward_fn(completions: list, prompts: list, **kwargs: Any) -> list[float]:
        rewards = []
        form_names = extract_form_names_from_metadata(kwargs, len(completions))

        for completion, _prompt, form_name in zip(completions, prompts, form_names):
            _call_count[0] += 1

            try:
                poem, valid_format = extract_poem_from_completion(completion)

                if not valid_format:
                    rewards.append(0.0)
                    telemetry.record(
                        form_name,
                        reward=0.0,
                        passed=False,
                        failure_reason="missing </think>",
                    )
                    telemetry.emit()
                    continue

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

                if _call_count[0] <= 5:
                    print(f"[DEBUG] {form_name}: score={result.score:.3f}", flush=True)

            except Exception as e:
                import traceback

                print(f"[reward error: {e}]")
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


def load_learnable_forms_from_results(results_file: str, top_n: int = 10) -> list[str]:
    """Load top N learnable forms from Baguettotron variance analysis.

    Uses grpo_signal from the search results which combines:
    - within_prompt_std (variance in rollout scores)
    - range_factor: 4 * mean * (1 - mean), peaks at 0.5

    High grpo_signal = high variance AND medium difficulty = best for learning.
    """
    import json

    with Path(results_file).open() as f:
        data = json.load(f)

    results = data.get("results", [])

    # Sort by grpo_signal (pre-computed in find_learnable_forms.py)
    sorted_results = sorted(results, key=lambda x: x.get("grpo_signal", 0), reverse=True)

    # Take top N forms with is_learnable=True
    learnable = []
    for r in sorted_results:
        if r.get("is_learnable", False):
            learnable.append(r["form_name"])
            if len(learnable) >= top_n:
                break

    return learnable


def create_dataset(
    num_prompts: int, seed: int = 42, learnable_forms: list[str] | None = None
) -> Dataset:
    """Create training dataset with learnable forms only."""
    from datasets import Dataset
    from prompt_generator import generate_learnable_forms_verifiers_dataset

    # Use custom learnable forms if provided
    prompt_generator = None
    if learnable_forms:
        # Temporarily override LEARNABLE_FORMS in prompt_generator
        import prompt_generator

        original_forms = prompt_generator.LEARNABLE_FORMS
        prompt_generator.LEARNABLE_FORMS = learnable_forms
        print(f"Using custom learnable forms: {learnable_forms}")

    try:
        raw_dataset = generate_learnable_forms_verifiers_dataset(num_prompts=num_prompts, seed=seed)
    finally:
        if learnable_forms and prompt_generator is not None:
            prompt_generator.LEARNABLE_FORMS = original_forms

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


def main():
    import argparse

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    parser = argparse.ArgumentParser(description="GRPO training for Baguettotron")
    parser.add_argument("--prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--output", default="models/grpo_baguettotron")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-file",
        default="experiments/baguettotron_form_variance.json",
        help="Load learnable forms from variance analysis results",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top learnable forms to use"
    )
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full model training)")
    parser.add_argument(
        "--telemetry-every",
        type=int,
        default=256,
        help="Emit aggregate reward telemetry every N scored samples",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Training: Baguettotron (Learnable Forms)")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Prompts: {args.prompts}")
    print(f"Batch size: {args.batch_size}")
    print(f"Generations: {args.num_generations}")
    print(f"Beta (KL): {args.beta}")
    print(f"Max completion: {MAX_COMPLETION_LENGTH}")

    # Load top learnable forms from variance analysis
    learnable_forms = None
    if Path(args.results_file).exists():
        learnable_forms = load_learnable_forms_from_results(args.results_file, args.top_n)
        print(f"Top {len(learnable_forms)} learnable forms from {args.results_file}:")
        for i, name in enumerate(learnable_forms, 1):
            print(f"  {i}. {name}")
    else:
        print(f"WARNING: Results file not found: {args.results_file}")
        print("Using default learnable forms from prompt_generator")

    print("=" * 60)

    # Load forms
    forms = get_forms(learnable_forms)
    print(f"Loaded {len(forms)} forms")

    # Create dataset with learnable forms
    dataset = create_dataset(args.prompts, args.seed, learnable_forms)

    # Create reward function
    reward_fn = create_reward_function(
        forms,
        telemetry_every=args.telemetry_every,
        use_wandb=not args.no_wandb,
    )

    # Compute max_steps
    max_steps = args.prompts // args.batch_size
    print(f"Max steps: {max_steps}")

    # TRL config
    grpo_config = GRPOConfig(
        output_dir=args.output,
        # KL regularization
        beta=args.beta,
        # Generation
        num_generations=args.num_generations,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_prompt_length=512,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        # CRITICAL: Stop at end token - include both variants for safety
        # Baguettotron uses '<|im_end>' (no closing |) but include both just in case
        # ALSO: skip_special_tokens=False to preserve </think> in output!
        generation_kwargs={
            "stop": ["<|im_end>", "<|im_end|>"],
            "skip_special_tokens": False,  # CRITICAL: preserve </think> tag
        },
        # Training
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.batch_size,
        max_steps=max_steps,
        # vLLM
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="0.0.0.0",
        vllm_server_port=args.port,
        # Logging
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=5,
        report_to="wandb" if not args.no_wandb else "none",
        run_name="grpo-baguettotron",
        # Memory
        gradient_checkpointing=True,
        bf16=True,
        # DAPO loss - prevents entropy collapse via:
        # 1. Dynamic KL coefficient that increases as entropy drops
        # 2. Clip-higher mechanism to encourage exploration
        loss_type="dapo",
        mask_truncated_completions=False,
        seed=args.seed,
    )

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CRITICAL: Monkey-patch tokenizer to preserve special tokens
    # TRL hardcodes skip_special_tokens=True in batch_decode, stripping </think>
    global _original_batch_decode
    _original_batch_decode = tokenizer.__class__.batch_decode
    tokenizer.__class__.batch_decode = _patched_batch_decode
    print("Applied monkey-patch to preserve </think> in batch_decode")

    # LoRA (optional - can be disabled with --no-lora)
    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"Using LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    else:
        print("LoRA disabled - full model training")

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )

    print("\nStarting training...")
    print("=" * 60)

    try:
        trainer.train()
    finally:
        flush_reward_telemetry(reward_fn)

    # Save
    final_path = Path(args.output) / "final"
    trainer.save_model(final_path)
    print(f"\nSaved to {final_path}")


if __name__ == "__main__":
    main()
