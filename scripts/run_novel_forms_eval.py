#!/usr/bin/env python3
"""
Run the novel forms evaluation using the verifiers framework.

This script evaluates LLM ability to follow unusual, non-traditional
poetic constraints that test instruction-following rather than
memorized patterns.

Patches verifiers' error handling to return dummy responses instead of
crashing, allowing evaluation to continue when some requests fail.

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
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Track errors for reporting
_api_errors: list[str] = []


def patch_verifiers_error_handling() -> None:
    """
    Patch verifiers to return dummy responses on ANY API error, not just
    context length errors. This allows evaluation to continue.
    """
    from verifiers.envs.environment import Environment
    from verifiers.utils.message_utils import get_overlong_prompt_dummy_response

    original_get_model_response = Environment.get_model_response

    async def resilient_get_model_response(self, client, model, prompt, **kwargs):
        try:
            return await original_get_model_response(self, client, model, prompt, **kwargs)
        except Exception as e:
            # Log and track the error
            error_msg = str(e)[:100]
            _api_errors.append(error_msg)
            self.logger.warning(f"API error (returning empty): {error_msg}")
            # Return dummy response - verifiers will score it as 0
            message_type = kwargs.get("message_type") or self.message_type
            return get_overlong_prompt_dummy_response(message_type)

    Environment.get_model_response = resilient_get_model_response


# Apply patch before importing verifiers components
patch_verifiers_error_handling()

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

# Hard forms designed to exploit LLM weaknesses
HARD_FORMS = [
    "AlternatingIsolation",
    "ArithmeticVerse",
    "CharacterBudgetPoem",
    "CharacterPalindromePoem",
    "CombinedChallenge",
    "DescendingStaircasePoem",
    "DoubleAcrosticPoem",
    "ExactWordPoem",
    "IsolatedCouplet",
    "PositionalPoem",
    "PrecisionHaiku",
    "PrecisionVerse",
    "StaircasePoem",
    "TotalCharacterPoem",
    "VowelBudgetPoem",
]


def get_all_forms() -> dict[str, object]:
    """Get all form classes (novel + hard)."""
    from abide.forms import hard, novel

    forms = {}

    # Load novel forms
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

    # Load hard forms
    for name in HARD_FORMS:
        if hasattr(hard, name):
            form_class = getattr(hard, name)
            try:
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

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=120.0,
    )

    print("Running novel forms verifiers evaluation")
    print(f"Model: {model}")
    print(f"Forms: {len(forms)}")
    print(f"Topic: {topic}")
    print(f"Rollouts per form: {rollouts}")
    print()

    # Create dataset - one example per form
    data_rows = []
    for form_name, form_instance in forms.items():
        prompt_text = create_prompt(form_name, form_instance, topic)
        data_rows.append({"question": prompt_text, "answer": form_name})

    dataset = Dataset.from_list(data_rows)
    print(f"Created dataset with {len(dataset)} examples")

    # Create reward function
    def abide_reward(completion: list[dict], answer: str, **kwargs: object) -> float:
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
        except Exception as e:
            print(f"  [reward error for {answer}: {e}]")
            return 0.0

    rubric = vf.Rubric(funcs=[abide_reward], weights=[1.0])
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)

    print("Starting evaluation...")
    print("-" * 60)

    start_time = time.time()
    _api_errors.clear()

    results = env.evaluate_sync(
        client=client,
        model=model,
        num_examples=len(dataset),
        rollouts_per_example=rollouts,
    )

    elapsed = time.time() - start_time

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

    # Categorize by difficulty
    easy = []
    medium = []
    hard = []

    for form_name, scores in form_scores.items():
        if scores:
            mean_score = sum(scores) / len(scores)
            entry = (form_name, mean_score, len(scores))
            if mean_score >= 0.8:
                easy.append(entry)
            elif mean_score >= 0.5:
                medium.append(entry)
            else:
                hard.append(entry)

    # Sort each category by score
    easy.sort(key=lambda x: -x[1])
    medium.sort(key=lambda x: -x[1])
    hard.sort(key=lambda x: -x[1])

    print("\n✓ EASY (80%+):")
    if easy:
        for name, score, n in easy:
            print(f"  {name}: {score:.1%} (n={n})")
    else:
        print("  (none)")

    print("\n~ MEDIUM (50-80%):")
    if medium:
        for name, score, n in medium:
            print(f"  {name}: {score:.1%} (n={n})")
    else:
        print("  (none)")

    print("\n✗ HARD (<50%):")
    if hard:
        for name, score, n in hard:
            print(f"  {name}: {score:.1%} (n={n})")
    else:
        print("  (none)")

    # Overall stats
    all_scores = [s for scores in form_scores.values() for s in scores]
    print()
    print("-" * 60)
    if all_scores:
        print(f"Overall mean: {sum(all_scores) / len(all_scores):.1%}")
        print(f"Total samples: {len(all_scores)}")
    print(f"Forms tested: {len([f for f in form_scores.values() if f])}")

    metadata = results["metadata"]
    print(f"Time: {metadata['time_ms'] / 1000:.1f}s (verifiers), {elapsed:.1f}s (total)")

    # Show API errors if any
    if _api_errors:
        print()
        print(f"API errors encountered: {len(_api_errors)}")
        for err in _api_errors[:5]:
            print(f"  - {err[:80]}")
        if len(_api_errors) > 5:
            print(f"  ... and {len(_api_errors) - 5} more")


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

    # Get all forms (novel + hard)
    all_forms = get_all_forms()
    print(f"Found {len(all_forms)} forms ({len(NOVEL_FORMS)} novel + {len(HARD_FORMS)} hard)")

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
    print("Forms Evaluation (Instruction-Following Test)")
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
