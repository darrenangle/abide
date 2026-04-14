"""Task and validation helpers for codex-authored SFT warmup poems."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from abide.training.prime_rl_env import (
    build_prime_rl_prompt_records,
    normalize_generated_poem,
    resolve_prime_rl_form_instances,
)
from abide.training.synthetic_sft import _SEED_POEMS, humanize_form_name

HARD_FORM_NAMES: tuple[str, ...] = (
    "Tanka",
    "PetrarchanSonnet",
    "Villanelle",
    "Ghazal",
    "Sestina",
)
FormSet = Literal["hard_forms", "all_forms"]
PromptMode = Literal["prime_rl", "brief_only"]
DEFAULT_FORM_SET: FormSet = "hard_forms"
DEFAULT_PROMPT_MODE: PromptMode = "prime_rl"
_WARMUP_QUALITY_GUIDANCE = (
    "Write an actual poem, not a placeholder or verifier shell. "
    "Use concrete language, vary the diction, and avoid filler like repeated single letters, "
    "repeating the same line verbatim, or obvious template scaffolds unless the form itself "
    "explicitly requires repetition."
)
_FORM_SPECIFIC_QUALITY_GUIDANCE: dict[str, str] = {
    "Abecedarian": (
        "Make each line begin with a real opening word for its required letter, "
        "and let the poem move through one coherent scene instead of repeating a fixed phrase."
    ),
}
_WORD_RE = re.compile(r"[A-Za-z']+")


def _slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _parse_form_names(
    form_names: str | list[str] | tuple[str, ...] | None,
    *,
    form_set: FormSet,
) -> list[str]:
    if form_names is None:
        if form_set == "hard_forms":
            selected_form_names = list(HARD_FORM_NAMES)
        elif form_set == "all_forms":
            selected_form_names = list(resolve_prime_rl_form_instances(form_set="all"))
        else:
            raise ValueError(f"Unsupported form_set: {form_set}")
    elif isinstance(form_names, str):
        selected_form_names = [name.strip() for name in form_names.split(",") if name.strip()]
    else:
        selected_form_names = [name.strip() for name in form_names if name.strip()]
    if not selected_form_names:
        raise ValueError("At least one form is required.")
    return selected_form_names


def _build_brief_only_prompt(form_name: str, structural_brief: str) -> str:
    form_label = humanize_form_name(form_name)
    form_specific = _FORM_SPECIFIC_QUALITY_GUIDANCE.get(form_name, "")
    extra = f" {form_specific}" if form_specific else ""
    return (
        f"Write a {form_label} that satisfies this exact structural brief: {structural_brief}. "
        f"{_WARMUP_QUALITY_GUIDANCE}{extra} Return only the poem."
    )


def _augment_prompt_for_warmup(form_name: str, prompt: str) -> str:
    stripped = prompt.strip()
    form_specific = _FORM_SPECIFIC_QUALITY_GUIDANCE.get(form_name, "")
    extra = f" {form_specific}" if form_specific else ""
    if stripped.endswith("Return only the poem."):
        prefix = stripped.removesuffix("Return only the poem.").rstrip()
        return f"{prefix} {_WARMUP_QUALITY_GUIDANCE}{extra} Return only the poem."
    return f"{stripped} {_WARMUP_QUALITY_GUIDANCE}{extra}"


def _tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _collect_quality_issues(form_name: str, poem: str) -> list[str]:
    lines = [line.strip() for line in poem.splitlines() if line.strip()]
    words = _tokenize_words(poem)
    if not words:
        return ["empty_after_normalization"]

    issues: list[str] = []
    unique_words = len(set(words))
    unique_ratio = unique_words / len(words)
    dominant_frequency = max(Counter(words).values())

    if len(words) >= 10 and unique_words <= 2:
        issues.append("too_few_unique_words")
    if len(words) >= 12 and unique_ratio < 0.2:
        issues.append("low_unique_token_ratio")
    if len(words) >= 10 and dominant_frequency / len(words) > 0.55:
        issues.append("dominant_repeated_token")
    if len(lines) >= 4:
        repeated_lines = max(Counter(lines).values())
        if repeated_lines >= max(3, len(lines) // 2):
            issues.append("too_many_repeated_lines")
    if len(words) >= 8 and all(len(word) <= 2 for word in words):
        issues.append("all_words_too_short")
    if form_name in {"PrimeVerse", "NumericalEcho", "ModularVerse"} and unique_words <= 3:
        issues.append("count_form_placeholder_shell")
    return issues


def _failed_rubric_items(verification_dict: dict[str, Any]) -> list[str]:
    rubric = verification_dict.get("rubric")
    if not isinstance(rubric, list):
        return []
    failures: list[str] = []
    for item in rubric:
        if not isinstance(item, dict):
            continue
        if bool(item.get("passed", False)):
            continue
        text = str(item.get("text", "")).strip()
        if text:
            failures.append(text)
    return failures[:5]


def _build_retry_feedback(failure: dict[str, Any]) -> str:
    error = str(failure.get("error", "verification_failed"))
    score = failure.get("score")
    if error == "too_close_to_seed":
        return (
            "Your previous draft was too close to an existing seed poem. Write a clearly new poem."
        )
    if error == "too_close_to_peer":
        return "Your previous draft was too close to another accepted poem for the same form. Change the language and imagery substantially."
    if error == "duplicate_response":
        return (
            "Your previous draft duplicated another response for the same form. Write a new poem."
        )
    if error == "degenerate_response":
        quality_issues = failure.get("quality_issues") or []
        issues_text = (
            ", ".join(str(issue) for issue in quality_issues) or "degenerate placeholder structure"
        )
        return f"Your previous draft passed structure but was rejected for low-quality generation: {issues_text}. Rewrite it as a real poem with varied language."
    if error == "judge_reject":
        label = str(failure.get("judge_label", "not_poem_like"))
        reason = str(failure.get("judge_reason", "")).strip()
        detail = f"{label}: {reason}" if reason else label
        return f"Your previous draft passed structure but was rejected by the poemness judge: {detail}. Rewrite it as a more natural, poem-like example."
    rubric_failures = failure.get("rubric_failures") or []
    if rubric_failures:
        return (
            f"Your previous draft failed structural checks (score={score}): "
            + "; ".join(str(item) for item in rubric_failures)
            + ". Fix the structure and return only the poem."
        )
    return f"Your previous draft did not pass automated validation (score={score}). Fix the structure and return only the poem."


def build_codex_spark_warmup_tasks(
    *,
    form_set: FormSet = DEFAULT_FORM_SET,
    form_names: str | list[str] | tuple[str, ...] | None = None,
    tasks_per_form: int = 25,
    seed: int = 42,
    prompt_mode: PromptMode = DEFAULT_PROMPT_MODE,
) -> list[dict[str, Any]]:
    """Build deterministic warmup-writing tasks for selected forms."""
    selected_form_names = _parse_form_names(form_names, form_set=form_set)
    if tasks_per_form < 1:
        raise ValueError("tasks_per_form must be at least 1.")
    if prompt_mode not in {"prime_rl", "brief_only"}:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    forms = resolve_prime_rl_form_instances(form_names=selected_form_names)
    tasks: list[dict[str, Any]] = []

    for form_index, form_name in enumerate(selected_form_names):
        structural_brief = forms[form_name].describe()
        if prompt_mode == "prime_rl":
            prompt_rows = [
                _augment_prompt_for_warmup(form_name, str(prompt_record["prompt"][0]["content"]))
                for prompt_record in build_prime_rl_prompt_records(
                    num_prompts=tasks_per_form,
                    seed=seed + form_index * 1000,
                    form_name=form_name,
                )
            ]
        else:
            prompt_rows = [_build_brief_only_prompt(form_name, structural_brief)] * tasks_per_form
        form_slug = _slugify(form_name)
        for prompt_index, prompt in enumerate(prompt_rows, start=1):
            tasks.append(
                {
                    "task_id": f"{form_slug}-{prompt_index:03d}",
                    "form_name": form_name,
                    "structural_brief": structural_brief,
                    "prompt": prompt,
                    "messages": [{"role": "user", "content": prompt}],
                }
            )

    return tasks


def write_codex_spark_warmup_tasks(tasks: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write codex-spark warmup tasks to JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=True, sort_keys=True) + "\n")
    return output


def validate_codex_spark_candidates(
    candidate_rows: list[dict[str, Any]],
    *,
    task_rows: list[dict[str, Any]],
    require_passed: bool = True,
    min_score: float = 0.0,
    require_unique_per_form: bool = True,
    max_seed_similarity: float = 0.92,
    max_peer_similarity: float = 0.9,
    enforce_quality: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate spark-authored poem candidates and convert them into SFT rows."""
    task_by_id = {str(task["task_id"]): task for task in task_rows}
    forms = resolve_prime_rl_form_instances(
        form_names=sorted({str(task["form_name"]) for task in task_rows})
    )
    seed_poems_by_form: dict[str, list[str]] = {}
    for seed in _SEED_POEMS:
        seed_poems_by_form.setdefault(seed.form_name, []).append(seed.poem)

    validated_records: list[dict[str, Any]] = []
    per_form_counts: Counter[str] = Counter()
    per_form_passes: Counter[str] = Counter()
    per_form_unique_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    seen_poems_by_form: dict[str, set[str]] = {}
    accepted_poems_by_form: dict[str, list[str]] = {}
    observed_max_seed_similarity = 0.0
    observed_max_peer_similarity = 0.0

    for row in candidate_rows:
        task_id = str(row.get("task_id", "")).strip()
        response = str(row.get("response", "")).strip()
        if not task_id or task_id not in task_by_id:
            failures.append(
                {
                    "task_id": task_id,
                    "error": "unknown_task_id",
                }
            )
            continue
        if not response:
            failures.append(
                {
                    "task_id": task_id,
                    "error": "empty_response",
                }
            )
            continue

        task = task_by_id[task_id]
        form_name = str(task["form_name"])
        normalized_poem = normalize_generated_poem(response)
        verification = forms[form_name].verify(normalized_poem)
        verification_dict = verification.to_dict()
        score = float(verification.score)
        passed = bool(verification.passed)
        quality_issues = (
            _collect_quality_issues(form_name, normalized_poem) if enforce_quality else []
        )
        is_duplicate = normalized_poem in seen_poems_by_form.setdefault(form_name, set())
        max_reference_similarity = max(
            (
                SequenceMatcher(
                    None,
                    normalized_poem.casefold(),
                    reference_poem.casefold(),
                ).ratio()
                for reference_poem in seed_poems_by_form.get(form_name, [])
            ),
            default=0.0,
        )
        max_peer_similarity_score = max(
            (
                SequenceMatcher(
                    None,
                    normalized_poem.casefold(),
                    accepted_poem.casefold(),
                ).ratio()
                for accepted_poem in accepted_poems_by_form.setdefault(form_name, [])
            ),
            default=0.0,
        )
        observed_max_seed_similarity = max(observed_max_seed_similarity, max_reference_similarity)
        observed_max_peer_similarity = max(observed_max_peer_similarity, max_peer_similarity_score)

        per_form_counts[form_name] += 1
        if passed:
            per_form_passes[form_name] += 1
        if not is_duplicate:
            per_form_unique_counts[form_name] += 1

        accepted = score >= min_score and (passed or not require_passed) and not quality_issues
        if require_unique_per_form and is_duplicate:
            accepted = False
        if max_reference_similarity > max_seed_similarity:
            accepted = False
        if max_peer_similarity_score > max_peer_similarity:
            accepted = False
        if not accepted:
            if is_duplicate:
                error = "duplicate_response"
            elif max_reference_similarity > max_seed_similarity:
                error = "too_close_to_seed"
            elif max_peer_similarity_score > max_peer_similarity:
                error = "too_close_to_peer"
            elif quality_issues:
                error = "degenerate_response"
            else:
                error = "verification_failed"
            failures.append(
                {
                    "task_id": task_id,
                    "form_name": form_name,
                    "score": score,
                    "passed": passed,
                    "error": error,
                    "quality_issues": quality_issues,
                    "max_seed_similarity": max_reference_similarity,
                    "max_peer_similarity": max_peer_similarity_score,
                    "rubric": verification_dict["rubric"],
                    "rubric_failures": _failed_rubric_items(verification_dict),
                }
            )
            continue

        seen_poems_by_form[form_name].add(normalized_poem)
        accepted_poems_by_form[form_name].append(normalized_poem)

        validated_records.append(
            {
                "task_id": task_id,
                "form_name": form_name,
                "prompt": task["prompt"],
                "response": normalized_poem,
                "messages": [
                    {"role": "user", "content": task["prompt"]},
                    {"role": "assistant", "content": normalized_poem},
                ],
                "structural_brief": task["structural_brief"],
                "verifier_score": score,
                "verifier_passed": passed,
            }
        )

    summary = {
        "num_tasks": len(task_rows),
        "num_candidates": len(candidate_rows),
        "num_validated": len(validated_records),
        "num_failures": len(failures),
        "per_form_candidate_counts": dict(sorted(per_form_counts.items())),
        "per_form_pass_counts": dict(sorted(per_form_passes.items())),
        "per_form_unique_counts": dict(sorted(per_form_unique_counts.items())),
        "max_seed_similarity": observed_max_seed_similarity,
        "max_peer_similarity": observed_max_peer_similarity,
        "failed_task_ids": [failure["task_id"] for failure in failures],
        "failures": failures,
    }
    return validated_records, summary


def build_codex_spark_retry_tasks(
    *,
    task_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build retry tasks that include the rejected draft and concrete validator feedback."""
    task_by_id = {str(task["task_id"]): task for task in task_rows}
    candidate_by_task_id = {
        str(candidate.get("task_id", "")).strip(): str(candidate.get("response", "")).strip()
        for candidate in candidate_rows
    }
    retry_tasks: list[dict[str, Any]] = []

    for failure in failure_rows:
        task_id = str(failure.get("task_id", "")).strip()
        if not task_id or task_id not in task_by_id:
            continue
        task = task_by_id[task_id]
        previous_response = candidate_by_task_id.get(task_id, "")
        feedback = _build_retry_feedback(failure)
        retry_prompt = (
            f"{feedback} Original structural brief: {task['structural_brief']}. "
            f"{_WARMUP_QUALITY_GUIDANCE} Return only the new poem."
        )
        retry_tasks.append(
            {
                "task_id": task_id,
                "form_name": task["form_name"],
                "structural_brief": task["structural_brief"],
                "prompt": retry_prompt,
                "messages": [
                    {"role": "user", "content": str(task["prompt"])},
                    {"role": "assistant", "content": previous_response},
                    {"role": "user", "content": retry_prompt},
                ],
                "retry_round": int(task.get("retry_round", 0)) + 1,
                "previous_response": previous_response,
                "failure": failure,
            }
        )

    return retry_tasks


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL rows from disk."""
    input_path = Path(path)
    return [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]


def write_jsonl_rows(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write arbitrary JSONL rows to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return output


__all__ = [
    "DEFAULT_FORM_SET",
    "DEFAULT_PROMPT_MODE",
    "HARD_FORM_NAMES",
    "build_codex_spark_retry_tasks",
    "build_codex_spark_warmup_tasks",
    "read_jsonl_rows",
    "validate_codex_spark_candidates",
    "write_codex_spark_warmup_tasks",
    "write_jsonl_rows",
]
