"""Task and validation helpers for codex-spark-authored SFT warmup poems."""

from __future__ import annotations

import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from abide.training.prime_rl_env import (
    build_prime_rl_prompt_records,
    normalize_generated_poem,
    resolve_prime_rl_form_instances,
)
from abide.training.synthetic_sft import _SEED_POEMS

HARD_FORM_NAMES: tuple[str, ...] = (
    "Tanka",
    "PetrarchanSonnet",
    "Villanelle",
    "Ghazal",
    "Sestina",
)


def _slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def build_codex_spark_warmup_tasks(
    *,
    form_names: str | list[str] | tuple[str, ...] = HARD_FORM_NAMES,
    tasks_per_form: int = 25,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build deterministic warmup-writing tasks for a fixed set of forms."""
    if isinstance(form_names, str):
        selected_form_names = [name.strip() for name in form_names.split(",") if name.strip()]
    else:
        selected_form_names = [name.strip() for name in form_names if name.strip()]
    if not selected_form_names:
        raise ValueError("At least one form is required.")
    if tasks_per_form < 1:
        raise ValueError("tasks_per_form must be at least 1.")

    forms = resolve_prime_rl_form_instances(form_names=selected_form_names)
    tasks: list[dict[str, Any]] = []

    for form_index, form_name in enumerate(selected_form_names):
        prompts = build_prime_rl_prompt_records(
            num_prompts=tasks_per_form,
            seed=seed + form_index * 1000,
            form_name=form_name,
        )
        structural_brief = forms[form_name].describe()
        form_slug = _slugify(form_name)
        for prompt_index, prompt_record in enumerate(prompts, start=1):
            prompt = str(prompt_record["prompt"][0]["content"])
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
        score = float(verification.score)
        passed = bool(verification.passed)
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

        accepted = score >= min_score and (passed or not require_passed)
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
            else:
                error = "verification_failed"
            failures.append(
                {
                    "task_id": task_id,
                    "form_name": form_name,
                    "score": score,
                    "passed": passed,
                    "error": error,
                    "max_seed_similarity": max_reference_similarity,
                    "max_peer_similarity": max_peer_similarity_score,
                    "rubric": verification.to_dict()["rubric"],
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
    "HARD_FORM_NAMES",
    "build_codex_spark_warmup_tasks",
    "read_jsonl_rows",
    "validate_codex_spark_candidates",
    "write_codex_spark_warmup_tasks",
    "write_jsonl_rows",
]
