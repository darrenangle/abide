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
import time
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
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> None:
    """Run evaluation with robust error handling."""
    try:
        from openai import OpenAI
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: uv sync --extra evals")
        sys.exit(1)

    # Create OpenAI client pointed at OpenRouter
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=90.0,
    )

    print("Running verifiers evaluation (with retry/skip)")
    print(f"Model: {model}")
    print(f"Forms: {len(forms)}")
    print(f"Topic: {topic}")
    print(f"Rollouts per form: {rollouts}")
    print(f"Max retries per request: {max_retries}")
    print()

    # Track results
    form_scores: dict[str, list[float]] = {f: [] for f in forms}
    errors: list[str] = []
    skipped = 0
    total_requests = len(forms) * rollouts
    completed = 0
    start_time = time.time()

    print("Starting evaluation...")
    print("-" * 60)

    for form_name, form_instance in forms.items():
        prompt = create_prompt(form_name, form_instance, topic)

        for rollout_idx in range(rollouts):
            completed += 1
            progress = f"[{completed}/{total_requests}]"

            # Retry loop
            success = False
            last_error = None

            for attempt in range(max_retries):
                try:
                    # Make API request
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2048,
                        temperature=0.7,
                    )

                    # Extract poem from response
                    poem = ""
                    if response.choices and response.choices[0].message:
                        poem = response.choices[0].message.content or ""

                    if not poem.strip():
                        raise ValueError("Empty response from model")

                    # Score the poem
                    result = form_instance.verify(poem)
                    score = result.score
                    form_scores[form_name].append(score)

                    # Show progress
                    status = "✓" if result.passed else "~"
                    print(f"{progress} {form_name} r{rollout_idx + 1}: {status} {score:.0%}")

                    success = True
                    break

                except KeyboardInterrupt:
                    print("\n\nInterrupted by user. Showing partial results...\n")
                    _print_results(form_scores, forms, errors, skipped, start_time)
                    return

                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        # Retry after delay
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue

            if not success:
                # All retries failed - skip this one
                skipped += 1
                error_msg = f"{form_name} r{rollout_idx + 1}: {last_error}"
                errors.append(error_msg)
                print(
                    f"{progress} {form_name} r{rollout_idx + 1}: ✗ SKIPPED ({last_error[:50]}...)"
                )

    # Print final results
    print()
    _print_results(form_scores, forms, errors, skipped, start_time)


def _print_results(
    form_scores: dict[str, list[float]],
    forms: dict[str, object],
    errors: list[str],
    skipped: int,
    start_time: float,
) -> None:
    """Print evaluation results."""
    print("=" * 60)
    print("Results")
    print("=" * 60)

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

    # Overall stats
    all_scores = [s for scores in form_scores.values() for s in scores]
    elapsed = time.time() - start_time

    print()
    print("-" * 60)
    if all_scores:
        print(f"Overall mean: {sum(all_scores) / len(all_scores):.1%}")
        print(f"Total samples: {len(all_scores)}")
    print(f"Skipped: {skipped}")
    print(f"Time: {elapsed:.1f}s")

    if errors:
        print()
        print("Errors encountered:")
        for err in errors[:10]:  # Show first 10
            print(f"  - {err[:80]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


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
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries per failed request (default: 3)",
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
            max_retries=args.retries,
        )
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
