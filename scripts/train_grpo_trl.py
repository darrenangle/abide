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
    CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model google/gemma-3-4b-it --port 8000

    # Then run training on GPU 0:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_trl.py

    # With custom beta:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_trl.py --beta 0.01
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from trl import GRPOConfig, GRPOTrainer


@dataclass
class TrainingArgs:
    """Training configuration."""

    # Model
    model_name: str = "google/gemma-3-4b-it"

    # Dataset
    num_prompts: int = 50000
    seed: int = 42

    # Training hyperparameters
    batch_size: int = 16
    num_generations: int = 16  # TRL calls this num_generations (equivalent to rollouts)
    learning_rate: float = 5e-5
    max_completion_length: int = 1536
    max_prompt_length: int = 512

    # KL regularization - THE KEY ADDITION!
    beta: float = 0.04  # KL coefficient. DeepSeek uses 0.001, some papers use 0.04

    # Clipping
    epsilon: float = 0.2  # PPO-style clipping

    # Output
    output_dir: str = "models/grpo_trl"
    save_steps: int = 50
    logging_steps: int = 10

    # vLLM
    vllm_port: int = 8000
    vllm_host: str = "0.0.0.0"

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Wandb
    wandb_project: str = "abide-grpo"
    use_wandb: bool = True


def get_forms() -> dict[str, object]:
    """Load all training forms from abide.forms."""
    import abide.forms as forms_module

    all_forms = {}
    for name in forms_module.__all__:
        try:
            form_class = getattr(forms_module, name)
            try:
                all_forms[name] = form_class()
            except TypeError:
                # Handle forms that need specific params
                if name == "StaircasePoem" or name == "DescendingStaircasePoem":
                    all_forms[name] = form_class(num_words=7)
                elif name == "VowelBudgetPoem":
                    all_forms[name] = form_class(vowel_count=30)
                elif name == "PrecisionVerse":
                    all_forms[name] = form_class(chars_per_line=25)
                elif name == "ExactWordPoem":
                    all_forms[name] = form_class(word_count=20)
                elif name == "CharacterBudgetPoem":
                    all_forms[name] = form_class(character="e", count=10)
                elif name == "TotalCharacterPoem":
                    all_forms[name] = form_class(total_chars=100)
                elif name == "FibonacciVerse":
                    all_forms[name] = form_class(num_lines=5)
                elif name == "TriangularVerse":
                    all_forms[name] = form_class(num_lines=4)
                elif name == "PiKu":
                    all_forms[name] = form_class(num_lines=5)
                elif name == "PrecisionHaiku":
                    all_forms[name] = form_class(chars_per_line=17)
                elif name == "ArithmeticVerse":
                    all_forms[name] = form_class(start=2, diff=2, num_lines=5)
                elif name == "PositionalPoem":
                    all_forms[name] = form_class(positions=[1, 2, 3])
                elif name == "IsolatedCouplet":
                    all_forms[name] = form_class(position=3)
                elif name == "AlternatingIsolation":
                    all_forms[name] = form_class(num_lines=6)
                elif name == "DoubleAcrosticPoem":
                    all_forms[name] = form_class(word="POETRY")
                elif name == "CombinedChallenge":
                    all_forms[name] = form_class(num_lines=4)
                elif name == "Lipogram":
                    all_forms[name] = form_class(forbidden="e")
                elif name == "Univocalic":
                    all_forms[name] = form_class(vowel="a")
                elif name == "Mesostic":
                    all_forms[name] = form_class(spine="POEM")
                elif name == "Anaphora":
                    all_forms[name] = form_class(phrase="I am", num_lines=4)
                elif name == "ModularVerse":
                    all_forms[name] = form_class(modulus=3, num_lines=6)
                elif name == "CoprimeVerse":
                    all_forms[name] = form_class(base=6, num_lines=4)
                elif name == "SquareStanzas":
                    all_forms[name] = form_class(size=4)
                elif name == "SelfReferential":
                    all_forms[name] = form_class(num_lines=4)
                elif name == "GoldenRatioVerse":
                    all_forms[name] = form_class(num_lines=6)
                elif name == "PythagoreanTercet":
                    all_forms[name] = form_class(scale=2)
                else:
                    continue
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            continue

    return all_forms


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


def create_reward_function(forms: dict[str, object]):
    """Create reward function for TRL.

    TRL's reward functions receive:
    - completions: list of completion strings (or multimodal lists)
    - prompts: list of prompt strings (or chat format lists)
    - **kwargs: additional metadata
    """
    _call_count = [0]
    _debug_printed = [False]

    def reward_fn(completions: list, prompts: list, **kwargs) -> list[float]:
        """Score completions against their target forms."""
        rewards = []

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

        for _i, (completion, prompt) in enumerate(zip(completions, prompts)):
            _call_count[0] += 1
            try:
                poem = get_completion_text(completion)
                prompt_text = extract_prompt_text(prompt)

                # Ensure these are strings
                if not isinstance(poem, str):
                    poem = str(poem) if poem else ""
                if not isinstance(prompt_text, str):
                    prompt_text = str(prompt_text) if prompt_text else ""

                if not poem or len(poem.strip()) < 10:
                    rewards.append(0.0)
                    continue

                # Extract form name from prompt
                # Prompts look like: "Write a [FormName] about..."
                form_name = None
                for name in forms:
                    if name.lower() in prompt_text.lower() or name in prompt_text:
                        form_name = name
                        break

                if not form_name:
                    if _call_count[0] <= 3:
                        print(f"[DEBUG] Could not extract form from prompt: {prompt_text[:100]}...")
                    rewards.append(0.0)
                    continue

                form_instance = forms.get(form_name)
                if form_instance is None:
                    rewards.append(0.0)
                    continue

                result = form_instance.verify(poem)
                rewards.append(float(result.score))

                if _call_count[0] <= 3:
                    print(f"[DEBUG] {form_name}: score={result.score:.3f}")

            except Exception as e:
                import traceback

                print(f"[reward error: {e}]")
                if _call_count[0] < 200:
                    traceback.print_exc()
                rewards.append(0.0)

        return rewards

    return reward_fn


def create_dataset(args: TrainingArgs, forms: dict[str, object]) -> Dataset:
    """Create training dataset using prompt generator."""
    from prompt_generator import generate_learnable_forms_verifiers_dataset

    use_learnable = os.environ.get("ABIDE_LEARNABLE", "").lower() in ("1", "true", "yes")

    if use_learnable:
        print("Using LEARNABLE forms only (top 10 with highest GRPO signal)")
        raw_dataset = generate_learnable_forms_verifiers_dataset(
            num_prompts=args.num_prompts,
            seed=args.seed,
        )
    else:
        # Default to learnable forms for TRL experiments
        print("Defaulting to LEARNABLE forms for TRL experiment")
        raw_dataset = generate_learnable_forms_verifiers_dataset(
            num_prompts=args.num_prompts,
            seed=args.seed,
        )

    # Convert to HuggingFace Dataset format expected by TRL
    # TRL expects a "prompt" column with chat messages
    # For vLLM multimodal, content must be list of {"type": "text", "text": "..."} dicts
    prompts = []
    for item in raw_dataset:
        # TRL with vLLM expects content as list of typed parts
        prompts.append([{"role": "user", "content": [{"type": "text", "text": item["prompt"]}]}])

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"Created dataset with {len(dataset)} prompts")

    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GRPO training with TRL (includes KL regularization)"
    )
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Model name")
    parser.add_argument("--prompts", type=int, default=50000, help="Number of prompts")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num-generations", type=int, default=16, help="Generations per prompt (rollouts)"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--beta", type=float, default=0.04, help="KL coefficient (0.001-0.04 typical)"
    )
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping epsilon")
    parser.add_argument("--output", default="models/grpo_trl", help="Output directory")
    parser.add_argument("--save-steps", type=int, default=50, help="Save every N steps")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Check for env override
    if os.environ.get("ABIDE_MODEL"):
        args.model = os.environ["ABIDE_MODEL"]

    training_args = TrainingArgs(
        model_name=args.model,
        num_prompts=args.prompts,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        beta=args.beta,
        epsilon=args.epsilon,
        output_dir=args.output,
        save_steps=args.save_steps,
        vllm_port=args.port,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )

    print("=" * 60)
    print("TRL GRPO Training (with KL regularization)")
    print("=" * 60)
    print(f"Model: {training_args.model_name}")
    print(f"Prompts: {training_args.num_prompts}")
    print(f"Generations per prompt: {training_args.num_generations}")
    print(f"Batch size: {training_args.batch_size}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Beta (KL coef): {training_args.beta}")
    print(f"Epsilon (clip): {training_args.epsilon}")
    print(f"Output: {training_args.output_dir}")
    print("=" * 60)

    # Load forms
    forms = get_forms()
    print(f"Loaded {len(forms)} forms")

    # Create dataset
    dataset = create_dataset(training_args, forms)

    # Create reward function
    reward_fn = create_reward_function(forms)

    # Compute max_steps
    max_steps = training_args.num_prompts // training_args.batch_size
    print(f"Max steps: {max_steps}")

    # Configure TRL GRPO
    grpo_config = GRPOConfig(
        output_dir=training_args.output_dir,
        # KL regularization - THE KEY PARAMETER
        beta=training_args.beta,
        # Clipping
        epsilon=training_args.epsilon,
        # Generation
        num_generations=training_args.num_generations,
        max_completion_length=training_args.max_completion_length,
        max_prompt_length=training_args.max_prompt_length,
        temperature=0.7,
        top_p=0.95,
        # Training
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=1,  # Micro batch
        gradient_accumulation_steps=training_args.batch_size,
        max_steps=max_steps,
        # vLLM integration
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=training_args.vllm_host,
        vllm_server_port=training_args.vllm_port,
        # Logging & saving
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        save_total_limit=5,
        report_to="wandb" if training_args.use_wandb else "none",
        run_name=f"grpo-trl-beta{training_args.beta}",
        # Memory optimization
        gradient_checkpointing=True,
        bf16=True,
        # DAPO-style loss (recommended)
        loss_type="dapo",
        # WARNING: Do NOT mask truncated completions if most outputs hit max length!
        # With mask_truncated_completions=True and clipped_ratio=1.0, loss becomes 0.
        mask_truncated_completions=False,
        # Seed
        seed=training_args.seed,
    )

    # Load model
    print(f"Loading model: {training_args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name,
        trust_remote_code=True,
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

    # Train
    trainer.train()

    # Save final model
    final_path = Path(training_args.output_dir) / "final"
    trainer.save_model(final_path)
    print(f"\nSaved final model to {final_path}")

    print("\n" + "=" * 60)
    print("TRL GRPO Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
