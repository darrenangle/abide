#!/usr/bin/env python3
"""
GRPO training script for poetic form generation using verifiers.

Uses abide constraints as verifiable rewards to train models
to follow complex poetic form requirements.

Usage:
    # Run baseline evaluation with local vLLM
    uv run python scripts/train_grpo.py --eval-only

    # Train using verifiers RLTrainer (requires prime-rl setup)
    uv run python scripts/train_grpo.py --train

    # Use OpenRouter for quick evaluation
    uv run python scripts/train_grpo.py --eval-only --openrouter

Prerequisites:
    uv sync --extra evals
    # For local inference: uv sync --extra vllm && python scripts/serve_local.py
    # For training: uv run vf-setup (sets up prime-rl)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Forms selected for training (challenging but achievable)
TRAINING_FORMS = [
    # Hard forms - main targets
    "StaircasePoem",
    "VowelBudgetPoem",
    "PrecisionVerse",
    "ExactWordPoem",
    "CharacterBudgetPoem",
    "ArithmeticVerse",
    # Mathematical forms
    "FibonacciVerse",
    "TriangularVerse",
    "CoprimeVerse",
    "PiKu",
    # Novel forms - medium difficulty
    "HourglassVerse",
    "PrimeVerse",
    "GoldenRatio",
    "DescendingStaircase",
]


def get_forms(form_names: list[str] | None = None) -> dict[str, object]:
    """Load form instances."""
    from abide.forms import hard, mathematical, novel

    # Map form names to modules and classes
    form_configs = {
        # Hard forms with easier parameters
        "StaircasePoem": (hard, "StaircasePoem", {"num_words": 7}),
        "VowelBudgetPoem": (hard, "VowelBudgetPoem", {"vowel_count": 30}),
        "PrecisionVerse": (hard, "PrecisionVerse", {"chars_per_line": 25}),
        "ExactWordPoem": (hard, "ExactWordPoem", {"word_count": 20}),
        "CharacterBudgetPoem": (hard, "CharacterBudgetPoem", {"character": "e", "count": 10}),
        "ArithmeticVerse": (hard, "ArithmeticVerse", {}),
        # Mathematical forms
        "FibonacciVerse": (mathematical, "FibonacciVerse", {"num_lines": 5}),
        "TriangularVerse": (mathematical, "TriangularVerse", {"num_lines": 4}),
        "CoprimeVerse": (mathematical, "CoprimeVerse", {"num_lines": 4}),
        "PiKu": (mathematical, "PiKu", {"num_lines": 5}),
        # Novel forms
        "HourglassVerse": (novel, "HourglassVerse", {}),
        "PrimeVerse": (novel, "PrimeVerse", {}),
        "GoldenRatio": (novel, "GoldenRatio", {}),
        "DescendingStaircase": (novel, "DescendingStaircase", {}),
    }

    if form_names is None:
        form_names = TRAINING_FORMS

    forms = {}
    for name in form_names:
        if name in form_configs:
            module, cls_name, kwargs = form_configs[name]
            form_class = getattr(module, cls_name)
            forms[name] = form_class(**kwargs)
        else:
            print(f"Warning: Unknown form {name}")

    return forms


def create_abide_env(forms: dict[str, object], num_samples: int = 50):
    """Create a verifiers SingleTurnEnv with abide rewards."""
    from datasets import Dataset

    import verifiers as vf

    topics = [
        "the passage of time",
        "love and loss",
        "nature and seasons",
        "memory and dreams",
        "hope and despair",
        "the ocean at night",
        "childhood memories",
        "autumn leaves falling",
        "a distant mountain",
        "rain on windows",
    ]

    # Create dataset
    data_rows = []
    for form_name, form_instance in forms.items():
        description = form_instance.describe()
        for i in range(num_samples):
            topic = topics[i % len(topics)]
            prompt = (
                f"Write a {form_name} poem about {topic}.\n"
                f"Requirements: {description}\n"
                f"Output ONLY the poem, nothing else."
            )
            data_rows.append(
                {
                    "question": prompt,
                    "answer": form_name,
                }
            )

    dataset = Dataset.from_list(data_rows)

    # Create reward function
    def abide_reward(completion: list[dict], answer: str, **kwargs) -> float:
        """Score poem against the target form."""
        try:
            poem = ""
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    content = msg.get("content")
                    if content:
                        poem = content
                    break

            if not poem:
                return 0.0

            form_instance = forms.get(answer)
            if form_instance is None:
                return 0.0

            result = form_instance.verify(poem)
            return result.score
        except Exception:
            return 0.0

    rubric = vf.Rubric(funcs=[abide_reward], weights=[1.0])
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)

    return env, dataset


def run_evaluation(
    env,
    client,
    model: str,
    num_examples: int,
    rollouts: int = 2,
) -> dict:
    """Run evaluation and print results."""
    import time

    print(f"\nEvaluating {num_examples} examples with {rollouts} rollouts each...")
    start = time.time()

    results = env.evaluate_sync(
        client=client,
        model=model,
        num_examples=num_examples,
        rollouts_per_example=rollouts,
    )

    elapsed = time.time() - start

    # Aggregate by form
    form_scores: dict[str, list[float]] = {}
    for i, answer in enumerate(results["answer"]):
        if answer not in form_scores:
            form_scores[answer] = []
        form_scores[answer].append(results["reward"][i])

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    sorted_forms = sorted(form_scores.items(), key=lambda x: -sum(x[1]) / len(x[1]))
    for form_name, scores in sorted_forms:
        mean = sum(scores) / len(scores)
        print(f"{form_name}: {mean:.1%} (n={len(scores)})")

    all_scores = [s for scores in form_scores.values() for s in scores]
    if all_scores:
        overall_mean = sum(all_scores) / len(all_scores)
        print(f"\nOverall: {overall_mean:.1%}")
        print(f"Time: {elapsed:.1f}s")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train/evaluate models on poetic forms using verifiers + abide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        default="google/gemma-3n-E4B-it",
        help="Model to use (default: google/gemma-3n-E4B-it)",
    )
    parser.add_argument(
        "--forms",
        type=str,
        default=None,
        help="Comma-separated list of forms (default: all training forms)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Samples per form (default: 10)",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=2,
        help="Rollouts per example (default: 2)",
    )

    # Inference backend
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)",
    )
    backend.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter API (requires OPENROUTER_API_KEY)",
    )

    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only",
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="Run GRPO training (requires prime-rl setup)",
    )

    args = parser.parse_args()

    # Get forms
    form_names = args.forms.split(",") if args.forms else None
    forms = get_forms(form_names)
    if not forms:
        print("Error: No valid forms specified")
        return 1

    print("=" * 60)
    print("Abide Poetry Training/Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Forms: {len(forms)} ({', '.join(forms.keys())})")
    print()

    # Create environment
    env, dataset = create_abide_env(forms, num_samples=args.samples)

    # Set up client
    from openai import OpenAI

    if args.openrouter:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OPENROUTER_API_KEY not set")
            return 1
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=120.0,
        )
        print("Using OpenRouter API")
    else:
        client = OpenAI(
            base_url=f"{args.vllm_url}/v1",
            api_key="local",
            timeout=120.0,
        )
        print(f"Using local vLLM at {args.vllm_url}")

    if args.eval_only:
        run_evaluation(
            env=env,
            client=client,
            model=args.model,
            num_examples=len(dataset),
            rollouts=args.rollouts,
        )
        return 0

    if args.train:
        print("\n" + "=" * 60)
        print("GRPO TRAINING")
        print("=" * 60)
        print()
        print("To train with verifiers, use the prime-rl setup:")
        print()
        print("  1. Set up prime-rl:")
        print("     uv run vf-setup")
        print()
        print("  2. Create a config file (configs/abide-poetry.toml):")
        print("     [model]")
        print(f'     name = "{args.model}"')
        print()
        print("     [env]")
        print('     name = "abide-poetry"')
        print()
        print("  3. Run training:")
        print("     uv run prime-rl @ configs/abide-poetry.toml")
        print()
        print("Alternatively, for a simpler setup, use TRL directly:")
        print("  See: https://github.com/willccbb/verifiers for examples")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
