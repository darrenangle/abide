#!/usr/bin/env python3
"""
Recombinant prompt generator for GRPO training.

Generates diverse prompts by combining:
- Forms (abide poetry forms)
- Topics/themes
- Optional style influences

Creates a large dataset for RL training with verifiers.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Topics organized by category
TOPICS = {
    "nature": [
        "the ocean at dawn",
        "autumn leaves falling",
        "a distant mountain",
        "rain on windows",
        "a frozen lake",
        "wildflowers in spring",
        "the desert at night",
        "a thunderstorm approaching",
        "morning fog lifting",
        "the first snow",
        "a river's journey",
        "the moon rising",
        "birdsong at twilight",
        "wind through pine trees",
        "tide pools at low tide",
    ],
    "emotion": [
        "the passage of time",
        "love and loss",
        "memory and dreams",
        "hope and despair",
        "grief transforming",
        "unexpected joy",
        "quiet loneliness",
        "fierce determination",
        "gentle forgiveness",
        "bittersweet nostalgia",
        "burning jealousy",
        "calm acceptance",
        "wild freedom",
        "tender vulnerability",
        "silent strength",
    ],
    "human": [
        "childhood memories",
        "a grandmother's hands",
        "leaving home",
        "returning after years",
        "a stranger's kindness",
        "betrayal by a friend",
        "first love",
        "last words",
        "an empty chair",
        "a letter never sent",
        "the weight of secrets",
        "dancing alone",
        "watching someone sleep",
        "a shared meal",
        "waiting for news",
    ],
    "urban": [
        "city lights at 3am",
        "an abandoned building",
        "subway at rush hour",
        "a street musician",
        "graffiti on concrete",
        "a cafe closing time",
        "traffic sounds fading",
        "neon signs flickering",
        "an empty parking lot",
        "rooftop gardens",
        "the last bus home",
        "a crowded elevator",
        "pigeons on statues",
        "steam from manholes",
        "scaffolding shadows",
    ],
    "abstract": [
        "the nature of silence",
        "what remains unsaid",
        "the space between words",
        "time folding back",
        "the color of absence",
        "where shadows begin",
        "the weight of waiting",
        "echoes of possibility",
        "the texture of regret",
        "boundaries dissolving",
        "the shape of tomorrow",
        "entropy and order",
        "the edge of meaning",
        "recursive patterns",
        "the geometry of loss",
    ],
}

# Flatten all topics
ALL_TOPICS = [topic for topics in TOPICS.values() for topic in topics]

# Style modifiers (optional, can be empty)
STYLE_MODIFIERS = [
    "",  # No modifier
    "with vivid imagery",
    "using concrete details",
    "with emotional restraint",
    "through sensory language",
    "with unexpected metaphors",
    "in a contemplative tone",
    "with rhythmic language",
    "using precise word choice",
    "with subtle irony",
    "through juxtaposition",
    "with musical cadence",
    "using sparse language",
    "with rich texture",
    "through careful observation",
]

# Constraint emphasis variations
CONSTRAINT_EMPHASIS = [
    "Follow the form requirements exactly.",
    "Pay careful attention to the structural constraints.",
    "The form requirements must be met precisely.",
    "Adhere strictly to the specified pattern.",
    "Match the required structure exactly.",
]


def get_forms() -> dict[str, object]:
    """Load ALL training forms from abide.forms."""
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import abide.forms as forms_module

    all_forms = {}
    for name in forms_module.__all__:
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
                    # Skip forms we can't instantiate
                    continue
        except Exception:
            continue

    return all_forms


def generate_prompt(
    form_name: str,
    form_instance: object,
    topic: str,
    style: str = "",
    emphasis: str = "",
) -> str:
    """Generate a single training prompt."""
    description = form_instance.describe()

    parts = [f"Write a {form_name} poem about {topic}."]

    if style:
        parts.append(style.capitalize() + ".")

    parts.append(f"\nRequirements: {description}")

    if emphasis:
        parts.append(f"\n{emphasis}")

    parts.append("\nOutput ONLY the poem, nothing else.")

    return " ".join(parts)


def generate_dataset(
    num_prompts: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """Generate a large dataset of diverse prompts."""
    random.seed(seed)

    forms = get_forms()
    form_names = list(forms.keys())

    dataset = []
    # Ensure at least 1 prompt per form, distribute remainder
    base_per_form = max(1, num_prompts // len(form_names))
    remainder = num_prompts - (base_per_form * len(form_names))

    for i, form_name in enumerate(form_names):
        form_instance = forms[form_name]
        # First forms get extra prompts from remainder
        count = base_per_form + (1 if i < remainder else 0)

        for _ in range(count):
            topic = random.choice(ALL_TOPICS)
            style = random.choice(STYLE_MODIFIERS)
            emphasis = random.choice(CONSTRAINT_EMPHASIS) if random.random() < 0.3 else ""

            prompt = generate_prompt(
                form_name=form_name,
                form_instance=form_instance,
                topic=topic,
                style=style,
                emphasis=emphasis,
            )

            dataset.append(
                {
                    "prompt": [{"role": "user", "content": prompt}],
                    # verifiers passes 'info' dict to reward functions
                    "info": {
                        "form_name": form_name,
                        "topic": topic,
                        "style": style,
                    },
                }
            )

    # Shuffle the dataset
    random.shuffle(dataset)

    return dataset


def generate_verifiers_dataset(num_prompts: int = 10000, seed: int = 42):
    """Generate dataset in verifiers-compatible format."""
    from datasets import Dataset

    data = generate_dataset(num_prompts=num_prompts, seed=seed)
    return Dataset.from_list(data)


def main():
    """Generate and save a training dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate GRPO training prompts")
    parser.add_argument("--num", type=int, default=10000, help="Number of prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/grpo_prompts.jsonl")
    parser.add_argument("--stats", action="store_true", help="Print statistics only")
    args = parser.parse_args()

    if args.stats:
        forms = get_forms()
        print(f"Forms: {len(forms)}")
        print(f"Topics: {len(ALL_TOPICS)}")
        print(f"Styles: {len(STYLE_MODIFIERS)}")
        print(f"Emphases: {len(CONSTRAINT_EMPHASIS)}")
        print(
            f"Potential combinations: {len(forms) * len(ALL_TOPICS) * len(STYLE_MODIFIERS) * len(CONSTRAINT_EMPHASIS):,}"
        )
        return

    print(f"Generating {args.num} prompts with seed {args.seed}...")
    dataset = generate_dataset(num_prompts=args.num, seed=args.seed)

    # Count forms
    form_counts = {}
    for item in dataset:
        form_counts[item["form_name"]] = form_counts.get(item["form_name"], 0) + 1

    print("\nForm distribution:")
    for form, count in sorted(form_counts.items()):
        print(f"  {form}: {count}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(dataset)} prompts to {output_path}")


if __name__ == "__main__":
    main()
