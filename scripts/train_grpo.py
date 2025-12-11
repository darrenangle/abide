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
    CUDA_VISIBLE_DEVICES=0 vf-vllm --model google/gemma-3n-e2b-it --port 8000 --enforce-eager

    # Then run training on GPU 1:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py

    # Resume from checkpoint:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py --resume models/abide_grpo/checkpoint-500
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# Add src and scripts to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults for 2x4090."""

    # Model
    model_name: str = "google/gemma-3n-e2b-it"  # Override with --model or ABIDE_MODEL env

    # Dataset
    num_prompts: int = 10000
    seed: int = 42

    # Training hyperparameters
    num_train_epochs: int = 1
    rollouts_per_example: int = 8
    batch_size: int = 16
    micro_batch_size: int = 2
    learning_rate: float = 1e-6
    max_seq_len: int = 2048
    max_prompt_len: int = 512

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
    """Load all training forms."""
    from abide.forms import hard, mathematical, novel

    return {
        "StaircasePoem": hard.StaircasePoem(num_words=7),
        "VowelBudgetPoem": hard.VowelBudgetPoem(vowel_count=30),
        "PrecisionVerse": hard.PrecisionVerse(chars_per_line=25),
        "ExactWordPoem": hard.ExactWordPoem(word_count=20),
        "CharacterBudgetPoem": hard.CharacterBudgetPoem(character="e", count=10),
        "FibonacciVerse": mathematical.FibonacciVerse(num_lines=5),
        "TriangularVerse": mathematical.TriangularVerse(num_lines=4),
        "PiKu": mathematical.PiKu(num_lines=5),
        "HourglassVerse": novel.HourglassVerse(),
        "PrimeVerse": novel.PrimeVerse(),
        "GoldenRatio": novel.GoldenRatio(),
    }


def get_completion_text(completion) -> str:
    """Extract text from completion."""
    val = completion
    if isinstance(val, list):
        if len(val) > 0 and isinstance(val[0], dict):
            val = val[-1].get("content", "")
        elif len(val) > 0 and isinstance(val[0], str):
            val = "".join(val)
    if isinstance(val, list):
        val = "".join([str(x) for x in val])
    return str(val)


def create_reward_function(forms: dict[str, object]):
    """Create reward function closure over forms."""

    def abide_reward(completion, **kwargs) -> float:
        """Score poem against the target form. Returns 0-1 score."""
        try:
            poem = get_completion_text(completion)
            if not poem or len(poem.strip()) < 10:
                return 0.0

            form_name = kwargs.get("form_name")
            if not form_name:
                return 0.0

            form_instance = forms.get(form_name)
            if form_instance is None:
                return 0.0

            result = form_instance.verify(poem)
            return result.score
        except Exception as e:
            print(f"[reward error: {e}]")
            return 0.0

    return abide_reward


def create_environment(forms: dict[str, object], config: TrainingConfig):
    """Create verifiers environment with generated prompts."""
    from prompt_generator import generate_verifiers_dataset

    import verifiers as vf

    print(f"Generating {config.num_prompts} prompts...")
    dataset = generate_verifiers_dataset(
        num_prompts=config.num_prompts,
        seed=config.seed,
    )
    print(f"Generated {len(dataset)} prompts")

    reward_fn = create_reward_function(forms)
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

    # Set WandB to offline to prevent network errors from crashing
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = config.wandb_project

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
            rl_config = RLConfig(
                output_dir=config.output_dir,
                run_name=f"abide-grpo-{int(time.time())}",
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_train_epochs,
                per_device_train_batch_size=config.micro_batch_size,
                batch_size=config.batch_size,
                micro_batch_size=config.micro_batch_size,
                rollouts_per_example=config.rollouts_per_example,
                max_seq_len=config.max_seq_len,
                max_prompt_len=config.max_prompt_len,
                vllm_server_port=config.vllm_port,
                temperature=0.7,
                max_tokens=1500,
                bf16=True,
                gradient_checkpointing=True,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                report_to="wandb" if config.use_wandb else "none",
                remove_unused_columns=False,
                use_lora=True,
            )

            # Load model manually (Gemma 3n needs special handling - no Liger, no use_cache)
            import torch
            from transformers import AutoModelForCausalLM

            model_path = config.resume_from or config.model_name
            print(f"Loading model: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

            # Create trainer
            trainer = RLTrainer(
                model=model,
                env=env,
                args=rl_config,
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

            return 0

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
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
                return 1

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
    parser.add_argument("--prompts", type=int, default=10000, help="Number of prompts")
    parser.add_argument("--rollouts", type=int, default=8, help="Rollouts per example")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
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
