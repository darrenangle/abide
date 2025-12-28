#!/usr/bin/env python3
"""
Find forms with enough within-rollout variance for GRPO learning.

For GRPO to work, we need forms where:
- The model sometimes succeeds and sometimes fails on the SAME prompt
- This creates advantage signal (reward - mean_reward != 0)

Forms that are "too easy" (all rollouts ~0.9) or "too hard" (all rollouts ~0.1)
won't produce learning signal.

Ideal forms have:
- Mean reward between 0.3-0.7 (partial success)
- High within-prompt std (variance in rollout scores)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from tqdm import tqdm


@dataclass
class FormAnalysis:
    """Analysis results for a single form."""

    form_name: str
    mean_reward: float
    within_prompt_std: float  # Average std across prompts (what GRPO sees)
    between_prompt_std: float  # Std of prompt means
    min_reward: float
    max_reward: float
    num_prompts: int
    num_rollouts_per_prompt: int

    @property
    def grpo_signal(self) -> float:
        """Estimated GRPO advantage signal strength."""
        # GRPO advantage ≈ within_prompt_std
        # But also need mean in learnable range (not 0 or 1)
        range_factor = 4 * self.mean_reward * (1 - self.mean_reward)  # peaks at 0.5
        return self.within_prompt_std * range_factor

    @property
    def is_learnable(self) -> bool:
        """Whether this form likely has enough signal for GRPO."""
        return (
            self.within_prompt_std > 0.1  # Enough variance
            and 0.2 < self.mean_reward < 0.8  # Not too easy or too hard
        )


def get_forms() -> dict[str, object]:
    """Load all forms from abide."""
    import abide.forms as forms_module

    all_forms = {}
    for name in forms_module.__all__:
        try:
            form_class = getattr(forms_module, name)
            try:
                all_forms[name] = form_class()
            except TypeError:
                # Forms needing params - use defaults
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
        except Exception:
            continue
    return all_forms


def generate_prompts_for_form(form_name: str, form_instance, num_prompts: int = 5) -> list[str]:
    """Generate diverse prompts for a form."""
    topics = [
        "love and loss",
        "nature and seasons",
        "memory and time",
        "hope and despair",
        "journey and discovery",
    ]

    description = form_instance.describe()
    prompts = []

    for topic in topics[:num_prompts]:
        prompt = f"Write a {form_name} poem about {topic}.\nRequirements: {description}\nOutput ONLY the poem, nothing else."
        prompts.append(prompt)

    return prompts


def analyze_form_with_model(
    form_name: str,
    form_instance,
    model_name: str = "google/gemma-3-4b-it",
    num_prompts: int = 3,
    num_rollouts: int = 8,
    vllm_url: str = "http://localhost:8000",
) -> FormAnalysis | None:
    """Analyze a form's variance using vLLM for generation."""
    import requests

    prompts = generate_prompts_for_form(form_name, form_instance, num_prompts)

    all_prompt_rewards = []

    for prompt in prompts:
        # Generate multiple rollouts for this prompt
        try:
            response = requests.post(
                f"{vllm_url}/v1/completions",
                json={
                    "model": model_name,
                    "prompt": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "n": num_rollouts,
                    "stop": ["<end_of_turn>", "<eos>"],
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"  Error generating for {form_name}: {e}")
            return None

        # Score each rollout
        rollout_rewards = []
        for choice in data.get("choices", []):
            text = choice.get("text", "").strip()
            try:
                result = form_instance.verify(text)
                rollout_rewards.append(result.score)
            except Exception:
                rollout_rewards.append(0.0)

        if rollout_rewards:
            all_prompt_rewards.append(rollout_rewards)

    if not all_prompt_rewards:
        return None

    # Calculate statistics
    all_rewards = [r for prompt_rewards in all_prompt_rewards for r in prompt_rewards]
    prompt_means = [np.mean(pr) for pr in all_prompt_rewards]
    prompt_stds = [np.std(pr) for pr in all_prompt_rewards]

    return FormAnalysis(
        form_name=form_name,
        mean_reward=float(np.mean(all_rewards)),
        within_prompt_std=float(np.mean(prompt_stds)),  # This is what GRPO sees
        between_prompt_std=float(np.std(prompt_means)),
        min_reward=float(np.min(all_rewards)),
        max_reward=float(np.max(all_rewards)),
        num_prompts=len(all_prompt_rewards),
        num_rollouts_per_prompt=num_rollouts,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Find forms with GRPO-learnable variance")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Model name")
    parser.add_argument("--vllm-url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--prompts", type=int, default=3, help="Prompts per form")
    parser.add_argument("--rollouts", type=int, default=8, help="Rollouts per prompt")
    parser.add_argument("--output", default="experiments/form_variance.json", help="Output file")
    parser.add_argument("--forms", nargs="*", help="Specific forms to test (default: all)")
    args = parser.parse_args()

    # Check vLLM is running
    import requests

    try:
        requests.get(f"{args.vllm_url}/health", timeout=5)
    except Exception:
        print(f"ERROR: vLLM not running at {args.vllm_url}")
        print(
            "Start it with: CUDA_VISIBLE_DEVICES=0 vf-vllm --model google/gemma-3-4b-it --port 8000"
        )
        sys.exit(1)

    print("=" * 70)
    print("Finding Forms with GRPO-Learnable Variance")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Prompts per form: {args.prompts}")
    print(f"Rollouts per prompt: {args.rollouts}")
    print("=" * 70)

    forms = get_forms()

    if args.forms:
        forms = {k: v for k, v in forms.items() if k in args.forms}

    print(f"\nTesting {len(forms)} forms...\n")

    results = []

    # Load existing results if resuming
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_forms = set()

    if output_path.exists():
        try:
            with output_path.open() as f:
                existing = json.load(f)
                for r in existing.get("results", []):
                    completed_forms.add(r["form_name"])
                    results.append(
                        FormAnalysis(
                            form_name=r["form_name"],
                            mean_reward=r["mean_reward"],
                            within_prompt_std=r["within_prompt_std"],
                            between_prompt_std=r["between_prompt_std"],
                            min_reward=r["min_reward"],
                            max_reward=r["max_reward"],
                            num_prompts=args.prompts,
                            num_rollouts_per_prompt=args.rollouts,
                        )
                    )
                print(f"Resuming: loaded {len(completed_forms)} completed forms")
        except Exception as e:
            print(f"Could not load existing results: {e}")

    for form_name, form_instance in tqdm(forms.items(), desc="Analyzing forms"):
        # Skip already completed forms
        if form_name in completed_forms:
            continue

        analysis = analyze_form_with_model(
            form_name=form_name,
            form_instance=form_instance,
            model_name=args.model,
            num_prompts=args.prompts,
            num_rollouts=args.rollouts,
            vllm_url=args.vllm_url,
        )

        if analysis:
            results.append(analysis)

            # Save incrementally after each form
            with output_path.open("w") as f:
                json.dump(
                    {
                        "model": args.model,
                        "prompts_per_form": args.prompts,
                        "rollouts_per_prompt": args.rollouts,
                        "results": [
                            {
                                "form_name": r.form_name,
                                "mean_reward": r.mean_reward,
                                "within_prompt_std": r.within_prompt_std,
                                "between_prompt_std": r.between_prompt_std,
                                "grpo_signal": r.grpo_signal,
                                "is_learnable": r.is_learnable,
                                "min_reward": r.min_reward,
                                "max_reward": r.max_reward,
                            }
                            for r in results
                        ],
                        "learnable_forms": [r.form_name for r in results if r.is_learnable],
                        "status": "in_progress",
                    },
                    f,
                    indent=2,
                )

    # Sort by GRPO signal strength
    results.sort(key=lambda x: x.grpo_signal, reverse=True)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (sorted by GRPO signal strength)")
    print("=" * 70)
    print(f"{'Form':<30} {'Mean':>6} {'Within-s':>9} {'Signal':>8} {'Learnable':>10}")
    print("-" * 70)

    learnable = []
    for r in results:
        status = "✓ YES" if r.is_learnable else "✗ no"
        print(
            f"{r.form_name:<30} {r.mean_reward:>6.3f} {r.within_prompt_std:>9.3f} {r.grpo_signal:>8.3f} {status:>10}"
        )
        if r.is_learnable:
            learnable.append(r)

    print("-" * 70)
    print(f"\nLearnable forms: {len(learnable)}/{len(results)}")

    if learnable:
        print("\n" + "=" * 70)
        print("FORMS SUITABLE FOR GRPO TRAINING")
        print("=" * 70)
        for r in learnable:
            print(
                f"  - {r.form_name}: mean={r.mean_reward:.2f}, within_std={r.within_prompt_std:.2f}"
            )
    else:
        print("\n⚠ No forms found with sufficient variance for GRPO learning.")
        print("Consider:")
        print("  1. Using a smaller/weaker base model")
        print("  2. Making scoring more granular")
        print("  3. Using SFT instead of GRPO")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(
            {
                "model": args.model,
                "prompts_per_form": args.prompts,
                "rollouts_per_prompt": args.rollouts,
                "results": [
                    {
                        "form_name": r.form_name,
                        "mean_reward": r.mean_reward,
                        "within_prompt_std": r.within_prompt_std,
                        "between_prompt_std": r.between_prompt_std,
                        "grpo_signal": r.grpo_signal,
                        "is_learnable": r.is_learnable,
                        "min_reward": r.min_reward,
                        "max_reward": r.max_reward,
                    }
                    for r in results
                ],
                "learnable_forms": [r.form_name for r in learnable],
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
