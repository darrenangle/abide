#!/usr/bin/env python3
"""
OpenRouter-based poem generator with abide verification loop.

Generates poems using OpenRouter API (Kimi K2, DeepSeek, etc.) with
retry logic based on abide verification scores.

Usage:
    # Single poem
    python scripts/openrouter_generator.py --form Sonnet --topic "autumn" --tone melancholic

    # Batch generation
    python scripts/openrouter_generator.py --form Sonnet --num 10 --output data/sonnets.jsonl

    # With specific model
    python scripts/openrouter_generator.py --form Haiku --model moonshotai/kimi-k2
"""

import argparse
import fcntl
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent and src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai import OpenAI

import abide.forms as forms_module
from abide import verify
from abide.synth_trace import generate_natural_trace, generate_synth_trace


def get_all_forms():
    """Get all form classes from abide.forms."""
    result = []
    for name in dir(forms_module):
        if name.startswith("_") or name[0].islower():
            continue
        obj = getattr(forms_module, name)
        if isinstance(obj, type) and hasattr(obj, "verify") and hasattr(obj, "instruction"):
            result.append(obj)
    return result


# System prompt for poetry generation
POET_SYSTEM_PROMPT = """You are a master poet skilled in formal verse. When asked to write a poem:

1. First, think through the form's requirements inside <think>...</think> tags
2. Plan your approach: rhyme scheme, meter, structure
3. Then write the poem after </think>

Be precise about syllable counts, rhyme patterns, and structural requirements.
The poem should appear AFTER the </think> tag, not inside it."""

# Default topics and tones
DEFAULT_TOPICS = [
    "autumn leaves falling",
    "ocean waves at sunset",
    "memory of childhood",
    "city lights at night",
    "mountain sunrise",
    "winter's first snow",
    "summer thunderstorm",
    "old photographs",
    "garden in spring",
    "starlight and silence",
]

DEFAULT_TONES = [
    "melancholic",
    "joyful",
    "contemplative",
    "nostalgic",
    "serene",
    "passionate",
    "wistful",
    "hopeful",
]


class OpenRouterGenerator:
    """Generate poems using OpenRouter API with abide verification."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "moonshotai/kimi-k2",
        max_retries: int = 5,
        min_score: float = 0.8,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = model
        self.max_retries = max_retries
        self.min_score = min_score

        # Load all forms
        self.forms = {f.__name__: f for f in get_all_forms()}

    def get_form_instruction(self, form_name: str) -> str:
        """Get the instruction text for a form."""
        if form_name not in self.forms:
            raise ValueError(f"Unknown form: {form_name}")

        form = self.forms[form_name]()
        return form.instruction()

    def generate_raw(self, prompt: str, form_instruction: str) -> str:
        """Generate raw response from OpenRouter."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": POET_SYSTEM_PROMPT},
                {"role": "user", "content": f"{form_instruction}\n\n{prompt}"},
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def generate_with_feedback(
        self,
        prompt: str,
        form_instruction: str,
        previous_response: str,
        rubric: list[dict],
    ) -> str:
        """Generate with feedback from previous failed attempt."""
        # Format rubric feedback
        feedback_lines = ["The previous attempt didn't meet all requirements:"]
        for item in rubric:
            if item["score"] < 0.8:
                feedback_lines.append(
                    f"- {item['criterion']}: {item['score']:.0%} (needs improvement)"
                )

        feedback = "\n".join(feedback_lines)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": POET_SYSTEM_PROMPT},
                {"role": "user", "content": f"{form_instruction}\n\n{prompt}"},
                {"role": "assistant", "content": previous_response},
                {
                    "role": "user",
                    "content": f"{feedback}\n\nPlease try again, paying close attention to the failing constraints.",
                },
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def parse_response(self, response: str) -> tuple[str, str]:
        """
        Parse response into (trace, poem).

        Handles formats:
        - <think>trace</think>poem
        - poem only (generates empty trace)
        """
        # Try to extract think tags
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)

        if think_match:
            trace = f"<think>{think_match.group(1)}</think>"
            # Everything after </think> is the poem
            poem_start = think_match.end()
            poem = response[poem_start:].strip()
        else:
            trace = ""
            poem = response.strip()

        return trace, poem

    def verify_poem(self, poem: str, form_name: str) -> tuple[float, list[dict]]:
        """Verify poem against form constraints."""
        if form_name not in self.forms:
            raise ValueError(f"Unknown form: {form_name}")

        form = self.forms[form_name]()
        result = verify(poem, form)

        rubric = [
            {"criterion": item.criterion, "score": item.score, "passed": item.passed}
            for item in result.rubric
        ]

        return result.score, rubric

    def generate_poem(
        self,
        form_name: str,
        topic: str,
        tone: str,
    ) -> dict[str, Any] | None:
        """
        Generate a verified poem with reasoning traces.

        Returns dict with:
        - form, prompt, poem, score, rubric
        - synth_trace, natural_trace
        - model, attempts

        Returns None if max retries exceeded.
        """
        form_instruction = self.get_form_instruction(form_name)
        prompt = f"Write a {form_name} about {topic} in a {tone} tone."

        previous_response = None
        previous_rubric = None

        for attempt in range(self.max_retries):
            try:
                # Generate
                if previous_response and previous_rubric:
                    response = self.generate_with_feedback(
                        prompt, form_instruction, previous_response, previous_rubric
                    )
                else:
                    response = self.generate_raw(prompt, form_instruction)

                # Parse
                natural_trace, poem = self.parse_response(response)

                if not poem:
                    print(f"  [Attempt {attempt + 1}] Empty poem, retrying...")
                    previous_response = response
                    continue

                # Verify
                score, rubric = self.verify_poem(poem, form_name)

                print(f"  [Attempt {attempt + 1}] Score: {score:.2f}")

                if score >= self.min_score:
                    # Success! Generate SYNTH trace
                    synth_trace = generate_synth_trace(
                        form_name=form_name,
                        topic=topic,
                        tone=tone,
                        form_instruction=form_instruction,
                        rubric=rubric,
                        total_score=score,
                        poem_lines=poem.split("\n"),
                    )

                    # If we didn't get a natural trace from model, synthesize one
                    if not natural_trace:
                        natural_trace = generate_natural_trace(
                            form_name=form_name,
                            topic=topic,
                            tone=tone,
                            form_instruction=form_instruction,
                            rubric=rubric,
                            total_score=score,
                        )

                    return {
                        "form": form_name,
                        "prompt": prompt,
                        "poem": poem,
                        "score": score,
                        "rubric": rubric,
                        "synth_trace": synth_trace,
                        "natural_trace": natural_trace,
                        "model": self.model,
                        "attempts": attempt + 1,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }

                # Not good enough, retry with feedback
                previous_response = response
                previous_rubric = rubric

            except Exception as e:
                print(f"  [Attempt {attempt + 1}] Error: {e}")
                previous_response = None
                previous_rubric = None

        print(f"  Failed after {self.max_retries} attempts")
        return None


def append_to_jsonl(filepath: str, data: dict):
    """Thread-safe append to JSONL file."""
    with Path(filepath).open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(data) + "\n")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_topics(filepath: str | None = None) -> list[str]:
    """Load topics from file or return defaults."""
    if filepath and Path(filepath).exists():
        with Path(filepath).open() as f:
            return [line.strip() for line in f if line.strip()]
    return DEFAULT_TOPICS


def load_tones(filepath: str | None = None) -> list[str]:
    """Load tones from file or return defaults."""
    if filepath and Path(filepath).exists():
        with Path(filepath).open() as f:
            return [line.strip() for line in f if line.strip()]
    return DEFAULT_TONES


def main():
    parser = argparse.ArgumentParser(
        description="Generate poems with OpenRouter + abide verification"
    )
    parser.add_argument("--form", required=True, help="Poetry form name (e.g., Sonnet, Haiku)")
    parser.add_argument("--topic", help="Specific topic (random if not provided)")
    parser.add_argument("--tone", help="Specific tone (random if not provided)")
    parser.add_argument("--num", type=int, default=1, help="Number of poems to generate")
    parser.add_argument("--model", default="moonshotai/kimi-k2", help="OpenRouter model")
    parser.add_argument("--min-score", type=float, default=0.8, help="Minimum score to accept")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per poem")
    parser.add_argument("--output", help="Output JSONL file")
    parser.add_argument("--topics-file", help="File with topics (one per line)")
    parser.add_argument("--tones-file", help="File with tones (one per line)")
    parser.add_argument("--dry-run", action="store_true", help="Print config without generating")

    args = parser.parse_args()

    # Load topics and tones
    topics = load_topics(args.topics_file)
    tones = load_tones(args.tones_file)

    print(f"Form: {args.form}")
    print(f"Model: {args.model}")
    print(f"Target: {args.num} poems with score >= {args.min_score}")
    print(f"Topics: {len(topics)} available")
    print(f"Tones: {len(tones)} available")

    if args.dry_run:
        print("\n[Dry run - no API calls]")
        return

    # Initialize generator
    generator = OpenRouterGenerator(
        model=args.model,
        max_retries=args.max_retries,
        min_score=args.min_score,
    )

    # Generate poems
    successes = 0
    failures = 0

    for i in range(args.num):
        topic = args.topic or random.choice(topics)
        tone = args.tone or random.choice(tones)

        print(f"\n[{i + 1}/{args.num}] Generating {args.form} about '{topic}' ({tone})...")

        result = generator.generate_poem(args.form, topic, tone)

        if result:
            successes += 1
            print(f"  Success! Score: {result['score']:.2f}")

            if args.output:
                append_to_jsonl(args.output, result)
                print(f"  Saved to {args.output}")
            else:
                print(f"\n--- SYNTH Trace ---\n{result['synth_trace']}")
                print(f"\n--- Poem ---\n{result['poem']}")
        else:
            failures += 1

    print("\n=== Summary ===")
    print(f"Successes: {successes}/{args.num}")
    print(f"Failures: {failures}/{args.num}")
    if args.output:
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
