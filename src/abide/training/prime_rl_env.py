"""Modern verifiers/prime-rl environment for Abide poetry training."""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

from abide.forms.catalog import (
    RL_DEFAULT_FORM_NAMES,
    WELL_KNOWN_FORM_NAMES,
    instantiate_form,
    load_form_instances,
    load_rl_default_form_instances,
    load_well_known_form_instances,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import verifiers as vf
    from abide.constraints import Constraint


DEFAULT_ENV_ID = "abide-poetry-forms"
PRIME_RL_DEFAULT_MODEL = "google/gemma-4-E2B-it"
SUPPORTED_FORM_SETS = ("all", "rl_default", "well_known")

_TOPICS = (
    "rain at dusk",
    "a train platform at dawn",
    "memory and grief",
    "salt wind over the gulf",
    "a house after everyone leaves",
    "winter branches and fog",
    "streetlights after midnight",
    "a long friendship under strain",
    "migration and return",
    "a market in summer heat",
    "river stones and patience",
    "music heard through apartment walls",
)

_TONES = (
    "meditative",
    "restrained",
    "playful",
    "melancholic",
    "tender",
    "severe",
    "wry",
    "reverent",
)

_PERSPECTIVES = (
    "first person singular",
    "first person plural",
    "second person",
    "third person limited",
)

_FRAMINGS = (
    "anchored in a specific scene",
    "built around a single vivid image",
    "moving from concrete detail to reflection",
    "keeping diction plain and precise",
    "using tactile detail rather than abstraction",
)


def normalize_generated_poem(text: str) -> str:
    """Normalize raw model output into poem text before verification."""
    cleaned = text

    if "```" in cleaned:
        code_block_match = re.search(r"```(?:\w*\n)?(.*?)```", cleaned, re.DOTALL)
        if code_block_match:
            cleaned = code_block_match.group(1)

    for token in (
        "<think>",
        "</think>",
        "<|think|>",
        "<|thinking|>",
        "<|/thinking|>",
        "<|im_end|>",
        "<|im_start|>",
        "<|end_of_text|>",
        "<|begin_of_text|>",
        "<|endoftext|>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<bos>",
        "<eos>",
    ):
        cleaned = cleaned.replace(token, "")

    preambles = (
        "No explanations.",
        "Here is the poem:",
        "Here's the poem:",
        "The poem:",
    )
    stripped = cleaned.strip()
    for preamble in preambles:
        if stripped.startswith(preamble):
            stripped = stripped[len(preamble) :].strip()
            break

    return stripped


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)
    return str(completion)


def _parse_form_names(form_names: str | Sequence[str] | None) -> list[str] | None:
    if form_names is None:
        return None
    if isinstance(form_names, str):
        parsed = [name.strip() for name in form_names.split(",") if name.strip()]
        return parsed or None
    parsed = [name.strip() for name in form_names if name.strip()]
    return parsed or None


def resolve_prime_rl_form_instances(
    *,
    form_set: str = "well_known",
    form_name: str | None = None,
    form_names: str | Sequence[str] | None = None,
) -> dict[str, Constraint]:
    """Resolve abide forms for the modern prime-rl environment."""
    explicit_names = _parse_form_names(form_names)
    if form_name:
        training_profile = form_name in set(RL_DEFAULT_FORM_NAMES) | set(WELL_KNOWN_FORM_NAMES)
        return {
            form_name: instantiate_form(
                form_name,
                training_profile=training_profile,
            )
        }

    if explicit_names:
        return {
            name: instantiate_form(
                name,
                training_profile=name in set(RL_DEFAULT_FORM_NAMES) | set(WELL_KNOWN_FORM_NAMES),
            )
            for name in explicit_names
        }

    if form_set == "well_known":
        return load_well_known_form_instances()
    if form_set == "rl_default":
        return load_rl_default_form_instances()
    if form_set == "all":
        return load_form_instances()

    valid = ", ".join(SUPPORTED_FORM_SETS)
    raise ValueError(f"Unsupported form_set {form_set!r}. Expected one of: {valid}")


def _build_prompt(form_name: str, description: str, *, rng: random.Random) -> str:
    topic = rng.choice(_TOPICS)
    tone = rng.choice(_TONES)
    perspective = rng.choice(_PERSPECTIVES)
    framing = rng.choice(_FRAMINGS)
    return (
        f"Write a {form_name} about {topic}. Keep the tone {tone}, use {perspective}, "
        f"and keep the poem {framing}. Satisfy this structural brief: {description}. "
        "Return only the poem."
    )


def build_prime_rl_dataset(
    *,
    num_prompts: int = 4096,
    seed: int = 42,
    form_set: str = "well_known",
    form_name: str | None = None,
    form_names: str | Sequence[str] | None = None,
) -> Any:
    """Build a single-turn verifiers dataset for prime-rl training."""
    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("datasets is required to build the prime-rl Abide dataset.") from e

    forms = resolve_prime_rl_form_instances(
        form_set=form_set,
        form_name=form_name,
        form_names=form_names,
    )
    if not forms:
        raise ValueError("At least one form must be selected for the prime-rl environment.")

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    form_names_cycle = list(forms)
    rng.shuffle(form_names_cycle)

    for idx in range(num_prompts):
        current_form_name = form_names_cycle[idx % len(form_names_cycle)]
        current_form = forms[current_form_name]
        prompt = _build_prompt(current_form_name, current_form.describe(), rng=rng)
        records.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "info": {"form_name": current_form_name},
            }
        )

    return Dataset.from_list(records)


def build_prime_rl_rubric(forms: dict[str, Constraint]) -> Any:
    """Create a verifiers rubric for the selected Abide forms."""
    try:
        import verifiers as vf
    except ImportError as e:
        raise ImportError("verifiers is required to build the prime-rl Abide rubric.") from e

    parser = vf.MaybeThinkParser() if hasattr(vf, "MaybeThinkParser") else None

    def abide_reward(
        completion: Any,
        info: dict[str, Any] | None = None,
        parser: Any | None = None,
        **_kwargs: Any,
    ) -> float:
        active_parser = parser or parser_fallback
        if active_parser is not None:
            poem = active_parser.parse_answer(completion) or ""
        else:
            poem = normalize_generated_poem(_completion_to_text(completion))

        poem = normalize_generated_poem(poem)
        if len(poem.strip()) < 10:
            return 0.0

        info_dict = info if isinstance(info, dict) else {}
        form_name = info_dict.get("form_name")
        if not isinstance(form_name, str):
            return 0.0

        form = forms.get(form_name)
        if form is None:
            return 0.0

        result = form.verify(poem)
        return float(result.score)

    def abide_pass(
        completion: Any,
        info: dict[str, Any] | None = None,
        parser: Any | None = None,
        **_kwargs: Any,
    ) -> float:
        active_parser = parser or parser_fallback
        if active_parser is not None:
            poem = active_parser.parse_answer(completion) or ""
        else:
            poem = normalize_generated_poem(_completion_to_text(completion))

        poem = normalize_generated_poem(poem)
        if len(poem.strip()) < 10:
            return 0.0

        info_dict = info if isinstance(info, dict) else {}
        form_name = info_dict.get("form_name")
        if not isinstance(form_name, str):
            return 0.0

        form = forms.get(form_name)
        if form is None:
            return 0.0

        result = form.verify(poem)
        return 1.0 if result.passed else 0.0

    parser_fallback = parser
    abide_reward.__name__ = "abide_reward"
    abide_pass.__name__ = "abide_pass"
    return vf.Rubric(
        funcs=[abide_reward, abide_pass],
        weights=[1.0, 0.0],
        parser=parser,
    )


def load_prime_rl_environment(
    *,
    num_prompts: int = 4096,
    seed: int = 42,
    form_set: str = "well_known",
    form_name: str | None = None,
    form_names: str | Sequence[str] | None = None,
) -> vf.Environment:
    """Load the installable Abide verifiers environment used by prime-rl."""
    try:
        import verifiers as vf
    except ImportError as e:
        raise ImportError("verifiers is required to load the prime-rl Abide environment.") from e

    forms = resolve_prime_rl_form_instances(
        form_set=form_set,
        form_name=form_name,
        form_names=form_names,
    )
    dataset = build_prime_rl_dataset(
        num_prompts=num_prompts,
        seed=seed,
        form_set=form_set,
        form_name=form_name,
        form_names=form_names,
    )
    rubric = build_prime_rl_rubric(forms)
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)


def load_environment(
    num_prompts: int = 4096,
    seed: int = 42,
    form_set: str = "well_known",
    form_name: str | None = None,
    form_names: str | Sequence[str] | None = None,
) -> vf.Environment:
    """Verifiers environment entrypoint expected by load_environment/env-server."""
    return load_prime_rl_environment(
        num_prompts=num_prompts,
        seed=seed,
        form_set=form_set,
        form_name=form_name,
        form_names=form_names,
    )


__all__ = [
    "DEFAULT_ENV_ID",
    "PRIME_RL_DEFAULT_MODEL",
    "SUPPORTED_FORM_SETS",
    "build_prime_rl_dataset",
    "build_prime_rl_rubric",
    "load_environment",
    "load_prime_rl_environment",
    "normalize_generated_poem",
    "resolve_prime_rl_form_instances",
]
