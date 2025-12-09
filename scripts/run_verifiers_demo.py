#!/usr/bin/env python3
"""
Demo: Running abide poetry evaluation with the verifiers framework.

This script demonstrates proper integration with the verifiers library
(https://github.com/PrimeIntellect-ai/verifiers) for evaluating LLM
poetry generation capabilities.

Usage:
    # Run with default settings (all forms)
    uv run python scripts/run_verifiers_demo.py

    # Use a specific model
    uv run python scripts/run_verifiers_demo.py --model anthropic/claude-3-haiku

    # Run specific forms only
    uv run python scripts/run_verifiers_demo.py --forms Haiku,Limerick,Sonnet

Environment:
    OPENROUTER_API_KEY: Your OpenRouter API key (required)
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_all_forms() -> dict[str, object]:
    """Get all form classes from abide.forms."""
    from abide import forms

    all_forms = {}
    for name in dir(forms):
        obj = getattr(forms, name)
        if inspect.isclass(obj) and hasattr(obj, "describe") and hasattr(obj, "verify"):
            try:
                instance = obj()
                all_forms[name] = instance
            except Exception:
                pass  # Skip forms that can't be instantiated with defaults
    return all_forms


def create_prompt(form_name: str, form_instance: object, topic: str) -> str:
    """Create a generation prompt using the form's describe() method."""
    description = form_instance.describe()
    return (
        f"Write a {form_name} about {topic}.\n"
        f"Requirements: {description}\n"
        f"Output ONLY the poem, nothing else."
    )


def run_verifiers_eval(
    model: str,
    forms: dict[str, object],
    topic: str,
    rollouts: int = 2,
) -> None:
    """Run evaluation using the verifiers framework."""
    try:
        from datasets import Dataset
        from openai import OpenAI

        import verifiers as vf
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: uv sync --extra evals")
        sys.exit(1)

    # Create OpenAI client pointed at OpenRouter
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,  # 60 second timeout per request
    )

    print("Running verifiers evaluation")
    print(f"Model: {model}")
    print(f"Forms: {len(forms)}")
    print(f"Topic: {topic}")
    print(f"Rollouts per form: {rollouts}")
    print()

    # Create dataset - one example per form
    data_rows = []
    for form_name, form_instance in forms.items():
        prompt_text = create_prompt(form_name, form_instance, topic)
        data_rows.append(
            {
                "question": prompt_text,
                "answer": form_name,  # Used by reward func to get constraint
            }
        )

    dataset = Dataset.from_list(data_rows)
    print(f"Created dataset with {len(dataset)} examples")
    print()

    # Create a reward function that scores based on the form
    def abide_reward(completion: list[dict], answer: str, **kwargs: object) -> float:
        """Score poem against the target form."""
        try:
            # Extract the generated text from completion
            poem = ""
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    content = msg.get("content")
                    if content:
                        poem = content
                    break

            if not poem:
                return 0.0

            # Get constraint for this form
            form_instance = forms.get(answer)
            if form_instance is None:
                return 0.0

            result = form_instance.verify(poem)
            return result.score
        except Exception as e:
            print(f"  [reward error for {answer}: {e}]")
            return 0.0

    # Create rubric with single reward function
    rubric = vf.Rubric(
        funcs=[abide_reward],
        weights=[1.0],
    )

    # Create environment
    env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
    )

    print("Starting evaluation...")
    print("-" * 60)

    # Run evaluation
    try:
        results = env.evaluate_sync(
            client=client,
            model=model,
            num_examples=len(dataset),
            rollouts_per_example=rollouts,
        )
    except Exception as e:
        print(f"\nEvaluation error: {e}")
        print("Some models may not be compatible with this evaluation.")
        return

    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    # Aggregate by form
    form_scores: dict[str, list[float]] = {f: [] for f in forms}

    answers = results["answer"]
    rewards = results["reward"]

    for i in range(len(answers)):
        form_name = answers[i]
        score = rewards[i]
        if form_name in form_scores:
            form_scores[form_name].append(score)

    # Sort by score descending
    sorted_forms = sorted(
        form_scores.items(),
        key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0,
        reverse=True,
    )

    for form_name, scores in sorted_forms:
        if scores:
            mean_score = sum(scores) / len(scores)
            print(f"{form_name}: {mean_score:.1%} (n={len(scores)})")

    # Overall
    all_scores = [s for scores in form_scores.values() for s in scores]
    if all_scores:
        print()
        print(f"Overall mean: {sum(all_scores) / len(all_scores):.1%}")
        print(f"Total samples: {len(all_scores)}")

    # Show metadata
    metadata = results["metadata"]
    print(f"Time: {metadata['time_ms'] / 1000:.1f}s")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run abide poetry evaluation with verifiers framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        default="deepseek/deepseek-chat-v3-0324:free",
        help="Model ID to use (default: deepseek/deepseek-chat-v3-0324:free)",
    )
    parser.add_argument(
        "--forms",
        type=str,
        default=None,
        help="Comma-separated list of form names (default: all forms)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="nature",
        help="Topic for poems (default: nature)",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=2,
        help="Number of rollouts per form (default: 2)",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Get your API key at https://openrouter.ai/keys")
        return 1

    # Get all forms
    all_forms = get_all_forms()
    print(f"Found {len(all_forms)} forms in abide.forms")

    # Filter if specified
    if args.forms:
        form_names = [f.strip() for f in args.forms.split(",")]
        forms = {k: v for k, v in all_forms.items() if k in form_names}
        if not forms:
            print(f"Error: No matching forms found for: {args.forms}")
            print(f"Available: {', '.join(sorted(all_forms.keys()))}")
            return 1
    else:
        forms = all_forms

    print("=" * 60)
    print("abide + verifiers Demo Evaluation")
    print("=" * 60)

    try:
        run_verifiers_eval(
            model=args.model,
            forms=forms,
            topic=args.topic,
            rollouts=args.rollouts,
        )
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
