#!/usr/bin/env python3
"""
Run the abide-major-poetic-forms-mini evaluation.

This script runs the mini evaluation suite against an LLM via OpenRouter.
It generates poems for various forms and measures how well the LLM
adheres to formal constraints.

Usage:
    # Run with default settings (llama-3.1-8b)
    python scripts/run_forms_mini.py

    # Use a specific model
    python scripts/run_forms_mini.py --model anthropic/claude-3-haiku

    # Run only specific forms
    python scripts/run_forms_mini.py --forms haiku,limerick

    # Run with more samples per form
    python scripts/run_forms_mini.py --samples 3

Environment:
    OPENROUTER_API_KEY: Your OpenRouter API key (required)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> int:
    """Run the forms mini evaluation."""
    parser = argparse.ArgumentParser(
        description="Run abide-major-poetic-forms-mini evaluation",
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
        help="Comma-separated list of forms to test (default: all)",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated list of topics (default: built-in topics)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples per form/topic combination (default: 1)",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use async mode for faster evaluation",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent requests in async mode (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output including generated poems",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Get your API key at https://openrouter.ai/keys")
        return 1

    # Parse forms and topics
    forms = args.forms.split(",") if args.forms else None
    topics = args.topics.split(",") if args.topics else None

    print("=" * 60)
    print("abide-major-poetic-forms-mini Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Forms: {forms or 'all'}")
    print(f"Topics: {len(topics) if topics else 'default (10)'}")
    print(f"Samples per task: {args.samples}")
    print(f"Mode: {'async' if args.async_mode else 'sync'}")
    print("=" * 60)
    print()

    try:
        from abide.evals import run_forms_mini_eval

        results = run_forms_mini_eval(
            model=args.model,
            num_samples=args.samples,
            topics=topics,
            forms=forms,
            async_mode=args.async_mode,
            concurrency=args.concurrency,
            verbose=True,  # Always show progress in CLI
        )

        # Print summary
        print(results.summary())
        print()

        # Print detailed results if verbose
        if args.verbose:
            print("-" * 60)
            print("Detailed Results:")
            print("-" * 60)

            for form_name, form_result in results.form_results.items():
                print(f"\n{form_name.upper()}")
                print("-" * 40)

                for i, sample in enumerate(form_result.samples):
                    print(f"\nSample {i + 1}:")
                    print(f"  Score: {sample.score:.2%}")
                    print(f"  Passed: {sample.passed}")
                    print(f"  Time: {sample.generation_time:.2f}s")

                    if sample.error:
                        print(f"  Error: {sample.error}")
                    else:
                        print("  Generated poem:")
                        for line in sample.generated_text.split("\n"):
                            print(f"    {line}")

        # Save detailed results if output specified
        if args.output:
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "model": args.model,
                    "forms": forms,
                    "samples": args.samples,
                },
                "summary": {
                    "total_samples": results.total_samples,
                    "mean_score": results.mean_score,
                    "pass_rate": results.overall_pass_rate,
                    "total_time": results.total_time,
                },
                "form_results": {
                    name: {
                        "mean_score": fr.mean_score,
                        "pass_rate": fr.pass_rate,
                        "best_score": fr.best_score,
                        "samples": [
                            {
                                "prompt": s.prompt,
                                "generated": s.generated_text,
                                "score": s.score,
                                "passed": s.passed,
                                "error": s.error,
                            }
                            for s in fr.samples
                        ],
                    }
                    for name, fr in results.form_results.items()
                },
            }

            output_path = Path(args.output)
            output_path.write_text(json.dumps(output_data, indent=2))
            print(f"\nDetailed results saved to: {args.output}")

        print()
        print("=" * 60)
        print("Evaluation complete!")
        print("=" * 60)

        return 0

    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: uv sync --extra evals")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
