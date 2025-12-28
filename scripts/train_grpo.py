#!/usr/bin/env python3
"""
GRPO training script for poetic form generation using verifiers RLTrainer.

Uses abide constraints as verifiable rewards to train models
to follow complex poetic form requirements.

Features:
- Recombinant prompt generation (62k+ combinations)
- Checkpointing every N steps
- Best model tracking based on reward
- Robust error handling with retries
- Resume from checkpoint support

Usage:
    # Start vf-vllm on GPU 0 first (in separate terminal):
    CUDA_VISIBLE_DEVICES=0 vf-vllm --model /home/darren/10k-poems/models/baguettotron_sft/final --port 8000 --enforce-eager

    # Then run training on GPU 1:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py

    # Resume from checkpoint:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py --resume models/abide_grpo/checkpoint-500

    # Or use a different model:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py --model google/gemma-3-270m-it
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import wandb

# Add src and scripts to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


import re

# Model paths
BAGUETTOTRON_PATH = "/home/darren/10k-poems/models/baguettotron_sft/final"
DEEPSEEK_R1_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OLMO_THINK_PATH = "allenai/OLMo-3-7B-Think-DPO"
OLMO_INSTRUCT_PATH = "allenai/OLMo-3-7B-Instruct"
GEMMA_3N_E4B_PATH = "google/gemma-3n-E4B-it"


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults for 2x4090."""

    # Model - OLMo-3-7B-Instruct (7B instruction-tuned model for poetry)
    model_name: str = OLMO_INSTRUCT_PATH  # Override with --model or ABIDE_MODEL env

    # Dataset
    num_prompts: int = 100000
    seed: int = 42

    # Training hyperparameters
    num_train_epochs: int = 1
    rollouts_per_example: int = 16
    batch_size: int = 32
    micro_batch_size: int = 1  # Keep at 1 for 7B models to avoid OOM
    learning_rate: float = 5e-5  # Aggressive LR for faster learning
    max_seq_len: int = 2048  # Room for reasoning traces + poem
    max_prompt_len: int = 384

    # Checkpointing
    output_dir: str = "models/abide_grpo"
    save_steps: int = 100
    eval_steps: int = 50
    keep_best_n: int = 3

    # Logging
    logging_steps: int = 10
    use_wandb: bool = True
    wandb_project: str = "abide-grpo"

    # vLLM
    vllm_port: int = 8000

    # Error handling
    max_retries: int = 3
    retry_delay: float = 5.0

    # Resume
    resume_from: str | None = None


class BestModelTracker:
    """Track and keep the N best models based on reward."""

    def __init__(self, output_dir: str, keep_n: int = 3):
        self.output_dir = Path(output_dir)
        self.keep_n = keep_n
        self.best_models: list[tuple[float, str]] = []  # (reward, path)
        self.tracker_file = self.output_dir / "best_models.json"
        self._load()

    def _load(self):
        """Load existing tracker state."""
        if self.tracker_file.exists():
            try:
                with self.tracker_file.open() as f:
                    data = json.load(f)
                    self.best_models = [(m["reward"], m["path"]) for m in data]
            except Exception:
                self.best_models = []

    def _save(self):
        """Save tracker state."""
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        with self.tracker_file.open("w") as f:
            json.dump(
                [{"reward": r, "path": p} for r, p in self.best_models],
                f,
                indent=2,
            )

    def maybe_save(self, reward: float, step: int, save_fn) -> bool:
        """Save model if it's among the best N."""
        # Check if this would be in top N
        if len(self.best_models) < self.keep_n or reward > self.best_models[-1][0]:
            # Save the model
            save_path = str(self.output_dir / f"best-step-{step}-reward-{reward:.4f}")
            save_fn(save_path)

            # Update tracker
            self.best_models.append((reward, save_path))
            self.best_models.sort(key=lambda x: -x[0])  # Sort by reward descending

            # Remove old models if we have too many
            while len(self.best_models) > self.keep_n:
                _, old_path = self.best_models.pop()
                if Path(old_path).exists():
                    shutil.rmtree(old_path)
                    print(f"  Removed old model: {old_path}")

            self._save()
            return True

        return False

    def get_best(self) -> str | None:
        """Get path to best model."""
        if self.best_models:
            return self.best_models[0][1]
        return None


def get_forms() -> dict[str, object]:
    """Load ALL training forms from abide.forms."""
    import abide.forms as forms_module

    all_forms = {}
    for name in forms_module.__all__:
        try:
            form_class = getattr(forms_module, name)
            # Try to instantiate with no args first
            try:
                all_forms[name] = form_class()
            except TypeError:
                # Some forms need specific params - use sensible defaults
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
                    # Skip forms we can't instantiate
                    print(f"  Skipping {name} (needs params)")
                    continue
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            continue

    return all_forms


def get_completion_text(completion, require_think_close: bool = True) -> str:
    """Extract poem text from completion, requiring proper </think> tag.

    For Baguettotron: The chat template already includes <think> in the prompt,
    so completions look like: "reasoning here</think>poem here"

    If require_think_close=True (default for Baguettotron):
      - MUST have </think> tag, otherwise returns "" (0 reward)
      - Returns only what comes AFTER </think>

    If require_think_close=False (for non-reasoning models):
      - Returns the text as-is (after stripping special tokens)
    """
    val = completion
    if isinstance(val, list):
        if len(val) > 0 and isinstance(val[0], dict):
            val = val[-1].get("content", "")
        elif len(val) > 0 and isinstance(val[0], str):
            val = "".join(val)
    if isinstance(val, list):
        val = "".join([str(x) for x in val])

    text = str(val)

    # Check for thinking close tags (different models use different formats)
    # Baguettotron: </think>
    # OLMo-Think: <|/thinking|>
    has_think_close = "</think>" in text or "<|/thinking|>" in text

    if require_think_close:
        # For thinking models: completion should be "reasoning[close_tag]poem"
        if not has_think_close:
            # No closing tag = incomplete reasoning = 0 reward
            return ""
        # Extract only what comes after the thinking section
        if "</think>" in text:
            text = text.split("</think>", 1)[-1]
        elif "<|/thinking|>" in text:
            text = text.split("<|/thinking|>", 1)[-1]

    # Strip any remaining thinking tags
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("<|thinking|>", "").replace("<|/thinking|>", "")

    # Strip ChatML tokens if present
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = text.replace("<|end_of_text|>", "").replace("<|begin_of_text|>", "")
    text = text.replace("<|endoftext|>", "")

    # Strip Gemma tokens if present
    text = text.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
    text = text.replace("<bos>", "").replace("<eos>", "")

    # Strip markdown code blocks (common in chat models)
    # Handle ```poem\n...\n``` or just ```\n...\n```
    code_block_match = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1)

    # Strip common preambles that chat models add
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


def create_reward_function(forms: dict[str, object], require_think_close: bool = True):
    """Create reward function closure over forms.

    Args:
        forms: Dict of form_name -> form_instance
        require_think_close: If True, completions must have </think> tag (for Baguettotron)
    """

    _call_count = [0]  # Mutable to track calls

    def abide_reward(completion, **kwargs) -> float:
        """Score poem against the target form. Returns 0-1 score."""
        _call_count[0] += 1
        try:
            poem = get_completion_text(completion, require_think_close=require_think_close)
            if not poem or len(poem.strip()) < 10:
                return 0.0

            # form_name is in the 'info' dict (set in dataset)
            info = kwargs.get("info", {})
            form_name = info.get("form_name") if isinstance(info, dict) else None

            if not form_name:
                if _call_count[0] <= 3:
                    print(f"[DEBUG] No form_name in info: {info}")
                return 0.0

            form_instance = forms.get(form_name)
            if form_instance is None:
                if _call_count[0] <= 3:
                    print(f"[DEBUG] Form not found: {form_name}")
                return 0.0

            result = form_instance.verify(poem)
            if _call_count[0] <= 3:
                # Show first 200 chars of extracted poem to debug extraction
                poem_preview = poem.replace("\n", "\\n")[:200]
                print(f"[DEBUG] {form_name}: score={result.score:.3f}")
                print(f"[DEBUG] poem: {poem_preview}...")
            return result.score
        except Exception as e:
            print(f"[reward error: {e}]")
            return 0.0

    return abide_reward


def create_environment(forms: dict[str, object], config: TrainingConfig):
    """Create verifiers environment with generated prompts."""
    from prompt_generator import (
        generate_learnable_forms_verifiers_dataset,
        generate_single_form_verifiers_dataset,
        generate_traditional_verifiers_dataset,
        generate_verifiers_dataset,
    )

    import verifiers as vf

    # Detect if using a thinking model (requires </think> tag in completions)
    # These models use explicit <think>...</think> tags in their chat templates
    model_lower = config.model_name.lower()
    # Only true thinking models (NOT olmo-instruct which has no think tags)
    is_thinking_model = any(
        x in model_lower
        for x in ["baguettotron", "qwq", "qwen3", "deepseek-r1", "r1-distill", "olmo-think"]
    )
    if is_thinking_model:
        print("Thinking model detected: requiring </think> tag in completions")
    else:
        print("Standard model: not requiring thinking tags")

    # Check for single-form ablation mode
    single_form = os.environ.get("ABIDE_SINGLE_FORM", "")

    # Check if traditional mode requested via env var or config
    use_traditional = os.environ.get("ABIDE_TRADITIONAL", "").lower() in ("1", "true", "yes")

    # Check if learnable forms mode requested (forms with high within-rollout variance)
    use_learnable = os.environ.get("ABIDE_LEARNABLE", "").lower() in ("1", "true", "yes")

    print(f"Generating {config.num_prompts} prompts...")
    if single_form:
        print(f"ABLATION MODE: Single form only ({single_form})")
        dataset = generate_single_form_verifiers_dataset(
            form_name=single_form,
            num_prompts=config.num_prompts,
            seed=config.seed,
        )
    elif use_learnable:
        print("Using LEARNABLE forms only (top 10 with highest GRPO signal)")
        dataset = generate_learnable_forms_verifiers_dataset(
            num_prompts=config.num_prompts,
            seed=config.seed,
        )
    elif use_traditional:
        print("Using TRADITIONAL forms only (weighted sampling)")
        dataset = generate_traditional_verifiers_dataset(
            num_prompts=config.num_prompts,
            seed=config.seed,
        )
    else:
        dataset = generate_verifiers_dataset(
            num_prompts=config.num_prompts,
            seed=config.seed,
        )
    print(f"Generated {len(dataset)} prompts")

    reward_fn = create_reward_function(forms, require_think_close=is_thinking_model)
    rubric = vf.Rubric(funcs=[reward_fn], weights=[1.0])
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)

    return env


def setup_signal_handlers(trainer):
    """Set up graceful shutdown on SIGINT/SIGTERM."""
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def handler(signum, frame):
        print("\n" + "=" * 60)
        print("Received interrupt signal. Saving checkpoint...")
        print("=" * 60)
        try:
            trainer.save_model(trainer.args.output_dir + "/interrupt-checkpoint")
            print(f"Saved to {trainer.args.output_dir}/interrupt-checkpoint")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        # Restore original handlers and re-raise
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def train_with_retry(config: TrainingConfig) -> int:
    """Run training with retry logic."""
    from verifiers.rl.trainer import RLConfig, RLTrainer

    print("=" * 60)
    print("Abide GRPO Training")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Prompts: {config.num_prompts}")
    print(f"Rollouts per example: {config.rollouts_per_example}")
    print(f"Total rollouts: ~{config.num_prompts * config.rollouts_per_example:,}")
    print(f"Output: {config.output_dir}")
    print("=" * 60)
    print()

    # Initialize wandb (wrapped in try-except to avoid crashing on sync issues)
    wandb_enabled = False
    if config.use_wandb:
        try:
            wandb.init(
                project=config.wandb_project,
                name=f"grpo-{config.model_name.split('/')[-1]}",
                config={
                    "model": config.model_name,
                    "num_prompts": config.num_prompts,
                    "rollouts_per_example": config.rollouts_per_example,
                    "batch_size": config.batch_size,
                    "micro_batch_size": config.micro_batch_size,
                    "learning_rate": config.learning_rate,
                    "max_seq_len": config.max_seq_len,
                },
            )
            wandb_enabled = True
            print("Wandb initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")

    # Load forms
    forms = get_forms()
    print(f"Forms: {len(forms)} ({', '.join(forms.keys())})")

    # Create environment
    env = create_environment(forms, config)

    # Best model tracker
    best_tracker = BestModelTracker(config.output_dir, keep_n=config.keep_best_n)

    # Training with retries
    for attempt in range(config.max_retries):
        try:
            print(f"\nTraining attempt {attempt + 1}/{config.max_retries}")

            # Configure training
            # Compute max_steps from num_prompts (verifiers defaults to 500!)
            max_steps = (config.num_prompts // config.batch_size) * config.num_train_epochs
            print(
                f"Computed max_steps: {max_steps} ({config.num_prompts} prompts / {config.batch_size} batch x {config.num_train_epochs} epochs)"
            )

            rl_config = RLConfig(
                output_dir=config.output_dir,
                run_name=f"abide-grpo-{int(time.time())}",
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_train_epochs,
                max_steps=max_steps,  # Override verifiers default of 500!
                per_device_train_batch_size=config.micro_batch_size,
                batch_size=config.batch_size,
                micro_batch_size=config.micro_batch_size,
                rollouts_per_example=config.rollouts_per_example,
                max_seq_len=config.max_seq_len,
                max_prompt_len=config.max_prompt_len,
                vllm_server_port=config.vllm_port,
                temperature=0.6,  # Nemotron starts at 0.6, increases later
                top_p=0.95,  # Nemotron uses 0.95 for diverse sampling
                repetition_penalty=1.2,  # Mild penalty to break loops
                max_tokens=2048,  # Poem output only (no thinking overhead)
                mask_truncated_completions=True,  # DAPO: exclude truncated from loss
                bf16=True,
                gradient_checkpointing=True,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                save_total_limit=5,  # Keep only last 5 checkpoints
                report_to="wandb" if config.use_wandb else "none",
                remove_unused_columns=False,
                use_lora=True,
            )

            # Inject stop tokens based on model type
            model_lower = config.model_name.lower()
            if "baguettotron" in model_lower:
                rl_config.sampling_args["stop"] = ["<|im_end|>"]
                print("Added stop token: <|im_end|>")
            elif "gemma" in model_lower:
                rl_config.sampling_args["stop"] = ["<end_of_turn>", "<eos>"]
                print("Added stop tokens: <end_of_turn>, <eos>")
            elif "qwen" in model_lower or "deepseek" in model_lower or "olmo" in model_lower:
                rl_config.sampling_args["stop"] = ["<|im_end|>", "<|endoftext|>"]
                print("Added stop tokens: <|im_end|>, <|endoftext|>")

            # Load model
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = config.resume_from or config.model_name
            print(f"Loading model: {model_path}")

            model_lower = model_path.lower()

            # Configure model loading based on architecture
            if "baguettotron" in model_lower:
                # Baguettotron is llama-based, no flash_attention needed
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
            elif "qwen" in model_lower or "deepseek" in model_lower:
                # Qwen/DeepSeek works with flash_attention_2
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            elif "olmo" in model_lower:
                # OLMo works with flash_attention_2
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            elif "gemma" in model_lower:
                # Gemma 3 - use flash_attention_2 for efficiency
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            else:
                # Default: use flash_attention_2
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # Ensure pad token is set (Baguettotron uses [PAD])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer: pad={tokenizer.pad_token}, eos={tokenizer.eos_token}")

            # Create trainer
            trainer = RLTrainer(
                model=model,
                env=env,
                args=rl_config,
                processing_class=tokenizer,
            )

            # Set up graceful shutdown
            setup_signal_handlers(trainer)

            print("\nStarting training...")
            print("=" * 60)

            trainer.train()

            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)

            # Save final model
            final_path = config.output_dir + "/final"
            trainer.save_model(final_path)
            print(f"Saved final model to {final_path}")

            # Print best model info
            best_path = best_tracker.get_best()
            if best_path:
                print(f"Best model: {best_path}")

            if wandb_enabled:
                with contextlib.suppress(Exception):
                    wandb.finish()

            return 0

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            if wandb_enabled:
                with contextlib.suppress(Exception):
                    wandb.finish()
            return 1

        except Exception as e:
            print(f"\nError during training: {e}")
            traceback.print_exc()

            if attempt < config.max_retries - 1:
                print(f"Retrying in {config.retry_delay}s...")
                time.sleep(config.retry_delay)

                # Try to resume from last checkpoint
                checkpoints = list(Path(config.output_dir).glob("checkpoint-*"))
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
                    config.resume_from = str(latest)
                    print(f"Will resume from {config.resume_from}")
            else:
                print("Max retries exceeded. Training failed.")
                if wandb_enabled:
                    with contextlib.suppress(Exception):
                        wandb.finish()
                return 1

    if wandb_enabled:
        with contextlib.suppress(Exception):
            wandb.finish()
    return 1


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GRPO training for abide poetry forms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default=None, help="Model name or path")
    parser.add_argument("--prompts", type=int, default=100000, help="Number of prompts")
    parser.add_argument("--rollouts", type=int, default=16, help="Rollouts per example")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output", default="models/abide_grpo", help="Output directory")
    parser.add_argument("--save-steps", type=int, default=100, help="Save every N steps")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = TrainingConfig(
        num_prompts=args.prompts,
        rollouts_per_example=args.rollouts,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
        save_steps=args.save_steps,
        resume_from=args.resume,
        vllm_port=args.port,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )

    if args.model:
        config.model_name = args.model
    elif os.environ.get("ABIDE_MODEL"):
        config.model_name = os.environ["ABIDE_MODEL"]

    return train_with_retry(config)


if __name__ == "__main__":
    sys.exit(main())
