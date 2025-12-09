#!/usr/bin/env python3
"""
Run the novel forms evaluation with verifiers framework.

This script evaluates LLM ability to follow unusual, non-traditional
poetic constraints that test instruction-following rather than
memorized patterns.

Usage:
    # Run with default settings
    uv run python scripts/run_novel_forms_eval.py

    # Use a specific model
    uv run python scripts/run_novel_forms_eval.py --model anthropic/claude-3-haiku

    # Run specific forms only
    uv run python scripts/run_novel_forms_eval.py --forms HourglassVerse,PrimeVerse

Environment:
    OPENROUTER_API_KEY: Your OpenRouter API key (required)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# List of all novel form names
NOVEL_FORMS = [
    "AlphabeticTerminus",
    "BinaryBeat",
    "ColorSpectrum",
    "ConsonantCascade",
    "DescendingStaircase",
    "EchoEnd",
    "ElementalVerse",
    "ExclamationEcho",
    "GoldenRatio",
    "HourglassVerse",
    "MirrorFrame",
    "MonotoneMountain",
    "NumberWord",
    "NumericalEcho",
    "OddEvenDance",
    "PrimeVerse",
    "QuestionQuest",
    "SandwichSonnet",
    "TemporalVerse",
    "ThunderVerse",
    "UniqueUtterance",
    "VoidVerse",
    "VowelPilgrimage",
    "WhisperPoem",
]


def get_novel_forms() -> dict[str, object]:
    """Get all novel form classes from abide.forms.novel."""
    from abide.forms import novel

    forms = {}
    for name in NOVEL_FORMS:
        if hasattr(novel, name):
            form_class = getattr(novel, name)
            try:
                # Special case for AlphabeticTerminus - use shorter version
                if name == "AlphabeticTerminus":
                    instance = form_class(letters="ABCDEFGH")
                else:
                    instance = form_class()
                forms[name] = instance
            except Exception as e:
                print(f"Warning: Could not instantiate {name}: {e}")
    return forms


def create_prompt(form_name: str, form_instance: object, topic: str) -> str:
    """Create a generation prompt using the form's describe() method."""
    description = form_instance.describe()
    return (
        f"Write a {form_name} poem about {topic}.\n"
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
        timeout=120.0,  # 120 second timeout per request (novel forms may take longer)
    )

    print("Running novel forms evaluation")
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
    def novel_form_reward(completion: list[dict], answer: str, **kwargs: object) -> float:
        """Score poem against the target novel form."""
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
        funcs=[novel_form_reward],
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
    print("NOVEL FORMS RESULTS")
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

    # Categorize by difficulty
    easy = []
    medium = []
    hard = []

    for form_name, scores in sorted_forms:
        if scores:
            mean_score = sum(scores) / len(scores)
            entry = (form_name, mean_score, len(scores))
            if mean_score >= 0.8:
                easy.append(entry)
            elif mean_score >= 0.5:
                medium.append(entry)
            else:
                hard.append(entry)

    print("\n✓ EASY (80%+):")
    for name, score, n in easy:
        print(f"  {name}: {score:.1%} (n={n})")

    print("\n~ MEDIUM (50-80%):")
    for name, score, n in medium:
        print(f"  {name}: {score:.1%} (n={n})")

    print("\n✗ HARD (<50%):")
    for name, score, n in hard:
        print(f"  {name}: {score:.1%} (n={n})")

    # Overall
    all_scores = [s for scores in form_scores.values() for s in scores]
    if all_scores:
        print()
        print("-" * 60)
        print(f"Overall mean: {sum(all_scores) / len(all_scores):.1%}")
        print(f"Total samples: {len(all_scores)}")
        print(f"Forms tested: {len([f for f in form_scores.values() if f])}")

    # Show metadata
    metadata = results["metadata"]
    print(f"Time: {metadata['time_ms'] / 1000:.1f}s")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run novel forms evaluation with verifiers framework",
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
        help="Comma-separated list of form names (default: all novel forms)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="the passage of time",
        help="Topic for poems (default: 'the passage of time')",
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

    # Get all novel forms
    all_forms = get_novel_forms()
    print(f"Found {len(all_forms)} novel forms")

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
    print("Novel Forms Evaluation (Instruction-Following Test)")
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
