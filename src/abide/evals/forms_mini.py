"""
abide-major-poetic-forms-mini evaluation.

A small, fast evaluation suite for testing LLM poetry generation
against major poetic forms. Includes forms of varying difficulty:
- Simple: Haiku, Limerick
- Medium: Quatrain, Couplet
- Complex: Sonnet, Villanelle

This eval is designed to run quickly (<5 minutes) while still
providing meaningful signal on poetry generation capabilities.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from abide.evals.client import OpenRouterClient
from abide.evals.runner import EvalConfig, EvalResult, EvalRunner, EvalTask
from abide.forms import (
    Haiku,
    HeroicCouplet,
    Limerick,
    Quatrain,
    ShakespeareanSonnet,
    Tanka,
)

if TYPE_CHECKING:
    from abide.constraints import Constraint


@dataclass
class FormSpec:
    """Specification for a form in the eval."""

    name: str
    constraint: Constraint
    prompt_template: str
    difficulty: str  # "simple", "medium", "complex"
    description: str


# The forms included in the mini eval
MAJOR_FORMS_MINI: list[FormSpec] = [
    # Simple forms (structure only)
    FormSpec(
        name="haiku",
        constraint=Haiku(syllable_tolerance=1),
        prompt_template=(
            "Write a haiku about {topic}. "
            "A haiku has exactly 3 lines with 5-7-5 syllables. "
            "Output ONLY the haiku, nothing else."
        ),
        difficulty="simple",
        description="Japanese 3-line poem with 5-7-5 syllables",
    ),
    FormSpec(
        name="tanka",
        constraint=Tanka(syllable_tolerance=1),
        prompt_template=(
            "Write a tanka about {topic}. "
            "A tanka has exactly 5 lines with 5-7-5-7-7 syllables. "
            "Output ONLY the tanka, nothing else."
        ),
        difficulty="simple",
        description="Japanese 5-line poem extending the haiku",
    ),
    # Medium forms (structure + rhyme)
    FormSpec(
        name="limerick",
        constraint=Limerick(rhyme_threshold=0.5),
        prompt_template=(
            "Write a limerick about {topic}. "
            "A limerick has 5 lines with AABBA rhyme scheme. "
            "Lines 1,2,5 are longer (8-9 syllables), lines 3,4 are shorter (5-6). "
            "It should be humorous. Output ONLY the limerick."
        ),
        difficulty="medium",
        description="Humorous 5-line poem with AABBA rhyme",
    ),
    FormSpec(
        name="quatrain",
        constraint=Quatrain(rhyme_scheme="ABAB", rhyme_threshold=0.5),
        prompt_template=(
            "Write a quatrain about {topic}. "
            "A quatrain is a 4-line poem with ABAB rhyme scheme. "
            "Each line should have about 8-10 syllables. "
            "Output ONLY the quatrain."
        ),
        difficulty="medium",
        description="4-line stanza with alternating rhyme",
    ),
    FormSpec(
        name="heroic_couplet",
        constraint=HeroicCouplet(strict=False),
        prompt_template=(
            "Write a heroic couplet about {topic}. "
            "A heroic couplet is 2 rhyming lines of iambic pentameter (10 syllables each). "
            "The lines must rhyme with each other. "
            "Output ONLY the couplet."
        ),
        difficulty="medium",
        description="Two rhyming lines of iambic pentameter",
    ),
    # Complex forms (all constraints)
    FormSpec(
        name="shakespearean_sonnet",
        constraint=ShakespeareanSonnet(
            syllable_tolerance=2,
            rhyme_threshold=0.4,
            strict=False,
        ),
        prompt_template=(
            "Write a Shakespearean sonnet about {topic}. "
            "A Shakespearean sonnet has:\n"
            "- Exactly 14 lines\n"
            "- Iambic pentameter (10 syllables per line)\n"
            "- Rhyme scheme: ABAB CDCD EFEF GG\n"
            "- A volta (turn) before the final couplet\n"
            "Output ONLY the sonnet."
        ),
        difficulty="complex",
        description="14-line poem with ABAB CDCD EFEF GG rhyme",
    ),
]

# Topics for variety in generation
DEFAULT_TOPICS = [
    "the changing seasons",
    "love and loss",
    "nature's beauty",
    "the passage of time",
    "hope and perseverance",
    "memories of childhood",
    "the ocean at sunset",
    "a quiet morning",
    "the stars at night",
    "finding peace",
]


def create_forms_mini_tasks(
    topics: list[str] | None = None,
    forms: list[str] | None = None,
) -> list[EvalTask]:
    """
    Create evaluation tasks for the forms mini eval.

    Args:
        topics: List of topics to use (default: DEFAULT_TOPICS)
        forms: List of form names to include (default: all)

    Returns:
        List of EvalTask objects ready for evaluation
    """
    topics = topics or DEFAULT_TOPICS
    forms_to_use = MAJOR_FORMS_MINI

    if forms is not None:
        forms_to_use = [f for f in forms_to_use if f.name in forms]

    tasks = []
    for form_spec in forms_to_use:
        # Create one task per topic
        for topic in topics:
            prompt = form_spec.prompt_template.format(topic=topic)
            tasks.append(
                EvalTask(
                    form_name=form_spec.name,
                    constraint=form_spec.constraint,
                    prompt=prompt,
                    description=f"{form_spec.name} about {topic}",
                )
            )

    return tasks


def create_forms_mini_dataset() -> list[dict[str, str]]:
    """
    Create a dataset in verifiers-compatible format.

    Returns a list of dicts with 'prompt' and 'answer' (form name) keys,
    suitable for use with verifiers SingleTurnEnv.
    """
    tasks = create_forms_mini_tasks()
    return [
        {
            "prompt": task.prompt,
            "answer": task.form_name,  # The expected form
        }
        for task in tasks
    ]


def run_forms_mini_eval(
    client: OpenRouterClient | None = None,
    model: str | None = None,
    num_samples: int = 1,
    topics: list[str] | None = None,
    forms: list[str] | None = None,
    async_mode: bool = False,
    concurrency: int = 5,
) -> EvalResult:
    """
    Run the forms mini evaluation.

    This is the main entry point for running the eval. It will:
    1. Create evaluation tasks for each form/topic combination
    2. Generate poems using the specified LLM
    3. Verify each poem against its target form
    4. Return aggregated results

    Args:
        client: OpenRouter client (created if not provided)
        model: Model to use for generation
        num_samples: Number of generations per form/topic
        topics: Topics to use (default: DEFAULT_TOPICS)
        forms: Form names to include (default: all)
        async_mode: Whether to run asynchronously
        concurrency: Max concurrent requests in async mode

    Returns:
        EvalResult with all metrics and samples

    Example:
        >>> from abide.evals import run_forms_mini_eval
        >>> results = run_forms_mini_eval(model="meta-llama/llama-3.1-8b-instruct")
        >>> print(results.summary())
    """
    # Create client if needed
    own_client = client is None
    if client is None:
        client = OpenRouterClient()

    try:
        # Create tasks
        tasks = create_forms_mini_tasks(topics=topics, forms=forms)

        # Configure eval
        config = EvalConfig(
            model=model or client.DEFAULT_MODEL,
            num_samples=num_samples,
            temperature=0.8,
            max_tokens=1024,
        )

        # Create runner
        runner = EvalRunner(client)

        # Run eval
        if async_mode:
            results = asyncio.run(runner.arun(tasks, config, concurrency=concurrency))
        else:
            results = runner.run(tasks, config)

        return results

    finally:
        if own_client:
            client.close()


# Convenience function for quick testing
def quick_test(
    form_name: str = "haiku",
    topic: str = "nature",
    model: str | None = None,
) -> tuple[str, float, bool]:
    """
    Quick test of a single form generation.

    Args:
        form_name: Name of form to test
        topic: Topic for the poem
        model: Model to use

    Returns:
        Tuple of (generated_poem, score, passed)
    """
    form_spec = next(
        (f for f in MAJOR_FORMS_MINI if f.name == form_name),
        MAJOR_FORMS_MINI[0],
    )

    with OpenRouterClient() as client:
        prompt = form_spec.prompt_template.format(topic=topic)
        poem = client.generate_poem(
            prompt,
            model=model,
            temperature=0.8,
        )

        result = form_spec.constraint.verify(poem)
        return poem, result.score, result.passed
