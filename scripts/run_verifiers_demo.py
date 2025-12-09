#!/usr/bin/env python3
"""
Demo: Running abide poetry evaluation with the verifiers framework.

This script demonstrates proper integration with the verifiers library
(https://github.com/PrimeIntellect-ai/verifiers) for evaluating LLM
poetry generation capabilities.

Usage:
    # Run with default settings
    uv run python scripts/run_verifiers_demo.py

    # Use a specific model
    uv run python scripts/run_verifiers_demo.py --model anthropic/claude-3-haiku

    # Run specific forms only
    uv run python scripts/run_verifiers_demo.py --forms haiku,limerick,sonnet

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


# The "obvious" forms for poetry evaluation - well-known, clear constraints
OBVIOUS_FORMS = [
    "haiku",
    "limerick",
    "sonnet",
    "quatrain",
    "couplet",
    "tanka",
    "villanelle",
    "ballad",
]

DEFAULT_TOPICS = [
    "nature",  # Single topic so rollouts = samples per form
]


def create_prompt(form_name: str, topic: str) -> str:
    """Create a generation prompt for a poetic form."""
    prompts = {
        "haiku": (
            f"Write a haiku about {topic}. "
            "A haiku has exactly 3 lines with 5-7-5 syllables. "
            "Output ONLY the haiku, nothing else."
        ),
        "limerick": (
            f"Write a limerick about {topic}. "
            "A limerick has 5 lines with AABBA rhyme scheme. "
            "Lines 1,2,5 are longer, lines 3,4 are shorter. "
            "It should be humorous. Output ONLY the limerick."
        ),
        "sonnet": (
            f"Write a Shakespearean sonnet about {topic}. "
            "A Shakespearean sonnet has 14 lines of iambic pentameter "
            "with rhyme scheme ABAB CDCD EFEF GG. "
            "Output ONLY the sonnet."
        ),
        "quatrain": (
            f"Write a quatrain about {topic}. "
            "A quatrain is a 4-line poem with ABAB rhyme scheme. "
            "Each line should have about 8-10 syllables. "
            "Output ONLY the quatrain."
        ),
        "couplet": (
            f"Write a heroic couplet about {topic}. "
            "A heroic couplet is 2 rhyming lines of iambic pentameter (10 syllables each). "
            "Output ONLY the couplet."
        ),
        "tanka": (
            f"Write a tanka about {topic}. "
            "A tanka has exactly 5 lines with 5-7-5-7-7 syllables. "
            "Output ONLY the tanka, nothing else."
        ),
        "villanelle": (
            f"Write a villanelle about {topic}. "
            "A villanelle has 19 lines: five tercets and a quatrain, "
            "with two refrains and two repeating rhymes (A1 b A2 / a b A1 / a b A2 / a b A1 / a b A2 / a b A1 A2). "
            "Output ONLY the villanelle."
        ),
        "ballad": (
            f"Write a ballad stanza about {topic}. "
            "A ballad stanza has 4 lines with ABCB rhyme scheme, "
            "alternating 8 and 6 syllables per line (8-6-8-6). "
            "Output ONLY the ballad stanza."
        ),
    }
    return prompts.get(form_name, f"Write a {form_name} about {topic}.")


def get_constraint(form_name: str):
    """Get the abide constraint for a form."""
    from abide.forms import (
        BalladStanza,
        Haiku,
        HeroicCouplet,
        Limerick,
        Quatrain,
        ShakespeareanSonnet,
        Tanka,
        Villanelle,
    )

    constraints = {
        "haiku": Haiku(syllable_tolerance=1),
        "limerick": Limerick(rhyme_threshold=0.5),
        "sonnet": ShakespeareanSonnet(syllable_tolerance=2, rhyme_threshold=0.4, strict=False),
        "quatrain": Quatrain(rhyme_scheme="ABAB", rhyme_threshold=0.5),
        "couplet": HeroicCouplet(strict=False),
        "tanka": Tanka(syllable_tolerance=1),
        "villanelle": Villanelle(rhyme_threshold=0.4),
        "ballad": BalladStanza(rhyme_scheme="ABCB", rhyme_threshold=0.5),
    }
    return constraints.get(form_name)


def make_abide_reward(form_name: str):
    """Create a verifiers-compatible reward function for a form."""
    constraint = get_constraint(form_name)

    def reward_func(prompt: list[dict], completion: list[dict], info: dict) -> float:
        """Score poem against abide constraint."""
        # Extract the generated text from completion
        poem = ""
        for msg in reversed(completion):
            if msg.get("role") == "assistant":
                poem = msg.get("content", "")
                break

        if not poem:
            return 0.0

        result = constraint.verify(poem)
        return result.score

    reward_func.__name__ = f"abide_{form_name}"
    return reward_func


def run_verifiers_eval(
    model: str,
    forms: list[str],
    topics: list[str],
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
    )

    print("Running verifiers evaluation")
    print(f"Model: {model}")
    print(f"Forms: {', '.join(forms)}")
    print(f"Topics: {len(topics)}")
    print(f"Rollouts per example: {rollouts}")
    print()

    # Create dataset - verifiers expects a HuggingFace Dataset with 'question' column
    data_rows = []
    for form_name in forms:
        for topic in topics:
            prompt_text = create_prompt(form_name, topic)
            data_rows.append(
                {
                    "question": prompt_text,
                    "answer": form_name,  # The form name - used by reward func
                }
            )

    dataset = Dataset.from_list(data_rows)
    print(f"Created dataset with {len(dataset)} examples")
    print()

    # Create a reward function that scores based on the form (passed as 'answer')
    def abide_reward(completion: list[dict], answer: str, **kwargs: object) -> float:
        """Score poem against the target form."""
        # Extract the generated text from completion
        poem = ""
        for msg in reversed(completion):
            if msg.get("role") == "assistant":
                poem = msg.get("content", "")
                break

        if not poem:
            return 0.0

        constraint = get_constraint(answer)
        if constraint is None:
            return 0.0

        result = constraint.verify(poem)
        return result.score

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
    results = env.evaluate_sync(
        client=client,
        model=model,
        num_examples=len(dataset),
        rollouts_per_example=rollouts,
    )

    # Print results
    # GenerateOutputs is a TypedDict with lists: answer, reward, completion, etc.
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

    for form_name, scores in form_scores.items():
        if scores:
            mean_score = sum(scores) / len(scores)
            print(f"{form_name}: mean={mean_score:.2%} (n={len(scores)})")

    # Overall
    all_scores = [s for scores in form_scores.values() for s in scores]
    if all_scores:
        print()
        print(f"Overall mean: {sum(all_scores) / len(all_scores):.2%}")
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
        help=f"Comma-separated list of forms (default: {','.join(OBVIOUS_FORMS[:6])})",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated list of topics (default: built-in topics)",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=2,
        help="Number of rollouts per example (default: 2)",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Get your API key at https://openrouter.ai/keys")
        return 1

    # Parse forms and topics
    forms = args.forms.split(",") if args.forms else OBVIOUS_FORMS[:6]
    topics = args.topics.split(",") if args.topics else DEFAULT_TOPICS

    print("=" * 60)
    print("abide + verifiers Demo Evaluation")
    print("=" * 60)

    try:
        run_verifiers_eval(
            model=args.model,
            forms=forms,
            topics=topics,
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
