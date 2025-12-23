#!/usr/bin/env python3
"""
Evaluate a local model on abide poetry forms.

Usage:
    # Start vf-vllm first:
    CUDA_VISIBLE_DEVICES=0 vf-vllm --model google/gemma-3n-e2b-it --port 8000

    # Run eval:
    python scripts/eval_local.py

    # Save results for comparison:
    python scripts/eval_local.py --output results/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Forms to evaluate - dynamically loaded, but can override with --forms
EVAL_FORMS = None  # Will load all forms dynamically

TOPICS = [
    "the passage of time",
    "love and loss",
    "nature and seasons",
    "memory and dreams",
    "hope and despair",
]


def get_forms(form_names: list[str] | None = None) -> dict[str, object]:
    """Load form instances. If form_names is None, load ALL forms."""
    import abide.forms as forms_module

    all_forms = {}
    names_to_load = form_names if form_names else forms_module.__all__

    for name in names_to_load:
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
                    continue
        except Exception:
            continue

    return all_forms


def evaluate_form(
    client: OpenAI,
    model: str,
    form_name: str,
    form_instance: object,
    topics: list[str],
    samples_per_topic: int = 1,
) -> list[dict]:
    """Evaluate a single form across topics."""
    results = []
    description = form_instance.describe()

    for topic in topics:
        for _ in range(samples_per_topic):
            prompt = (
                f"Write a {form_name} poem about {topic}.\n"
                f"Requirements: {description}\n"
                f"Output ONLY the poem, nothing else."
            )

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.7,
                )
                poem = response.choices[0].message.content.strip()
                result = form_instance.verify(poem)
                score = result.score

                results.append(
                    {
                        "form": form_name,
                        "topic": topic,
                        "score": score,
                        "poem": poem,
                        "passed": score >= 1.0,
                    }
                )

            except Exception as e:
                print(f"  Error: {e}")
                results.append(
                    {
                        "form": form_name,
                        "topic": topic,
                        "score": 0.0,
                        "poem": "",
                        "error": str(e),
                        "passed": False,
                    }
                )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local model on poetry forms")
    parser.add_argument("--model", default="google/gemma-3n-e2b-it", help="Model name")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--samples", type=int, default=1, help="Samples per topic per form")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--forms", type=str, help="Comma-separated form names (default: all)")
    args = parser.parse_args()

    # Setup client
    client = OpenAI(base_url=f"{args.url}/v1", api_key="local", timeout=120.0)

    # Test connection
    try:
        models = client.models.list()
        print(f"Connected to vLLM at {args.url}")
        print(f"Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"Failed to connect to vLLM at {args.url}: {e}")
        return 1

    # Load forms (None means all)
    form_names = args.forms.split(",") if args.forms else None
    forms = get_forms(form_names)
    print(f"\nEvaluating {len(forms)} forms with {args.samples} samples per topic")
    print(f"Forms: {', '.join(forms.keys())}")
    print()

    # Run evaluation
    all_results = []
    form_scores = {}
    start_time = time.time()

    for form_name, form_instance in forms.items():
        print(f"Evaluating {form_name}...", end=" ", flush=True)
        results = evaluate_form(
            client=client,
            model=args.model,
            form_name=form_name,
            form_instance=form_instance,
            topics=TOPICS,
            samples_per_topic=args.samples,
        )
        all_results.extend(results)

        scores = [r["score"] for r in results]
        mean_score = sum(scores) / len(scores) if scores else 0
        passed = sum(1 for r in results if r["passed"])
        form_scores[form_name] = mean_score

        print(f"{mean_score:.1%} ({passed}/{len(results)} passed)")

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    sorted_forms = sorted(form_scores.items(), key=lambda x: -x[1])
    for form_name, score in sorted_forms:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"{form_name:25s} {bar} {score:.1%}")

    overall = sum(form_scores.values()) / len(form_scores) if form_scores else 0
    total_passed = sum(1 for r in all_results if r["passed"])
    print()
    print(f"Overall score: {overall:.1%}")
    print(f"Total passed: {total_passed}/{len(all_results)}")
    print(f"Time: {elapsed:.1f}s")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_score": overall,
            "total_passed": total_passed,
            "total_samples": len(all_results),
            "form_scores": form_scores,
            "results": all_results,
        }

        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
