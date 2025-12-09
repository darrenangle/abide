"""
Eval runner for abide poetic forms.

Provides a simple loop for:
1. Generate prompts for poetic forms
2. Request completions from LLM
3. Verify completions with abide constraints
4. Collect metrics
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from abide.constraints import Constraint
    from abide.evals.client import OpenRouterClient


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    # LLM settings
    model: str = "meta-llama/llama-3.1-8b-instruct"
    temperature: float = 0.8
    max_tokens: int = 1024

    # Eval settings
    num_samples: int = 5  # Number of generations per form
    timeout_per_sample: float = 60.0  # Seconds

    # Prompt settings
    system_prompt: str | None = None

    # Logging
    verbose: bool = False  # Print progress during evaluation


@dataclass
class SampleResult:
    """Result of a single generation attempt."""

    form_name: str
    prompt: str
    generated_text: str
    score: float
    passed: bool
    verification_details: dict[str, Any]
    generation_time: float
    model: str
    error: str | None = None


@dataclass
class FormResult:
    """Aggregated results for a single poetic form."""

    form_name: str
    samples: list[SampleResult]
    mean_score: float
    pass_rate: float
    best_score: float
    best_sample: SampleResult | None


@dataclass
class EvalResult:
    """Complete evaluation results."""

    config: EvalConfig
    form_results: dict[str, FormResult]
    total_samples: int
    mean_score: float
    overall_pass_rate: float
    total_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Evaluation Results ({self.config.model})",
            "=" * 50,
            f"Total samples: {self.total_samples}",
            f"Mean score: {self.mean_score:.2%}",
            f"Pass rate: {self.overall_pass_rate:.2%}",
            f"Total time: {self.total_time:.1f}s",
            "",
            "Per-form results:",
        ]

        for name, result in self.form_results.items():
            lines.append(
                f"  {name}: score={result.mean_score:.2%}, "
                f"pass_rate={result.pass_rate:.2%} "
                f"(best={result.best_score:.2%})"
            )

        return "\n".join(lines)


@dataclass
class EvalTask:
    """A single evaluation task (form + prompt)."""

    form_name: str
    constraint: Constraint
    prompt: str
    description: str = ""


class EvalRunner:
    """
    Runner for poetic form evaluations.

    This implements the core eval loop:
    1. For each form, generate a prompt
    2. Request LLM completion
    3. Verify with abide constraint
    4. Collect and aggregate metrics

    Example:
        >>> from abide.evals import OpenRouterClient, EvalRunner, EvalConfig
        >>> from abide.forms import Haiku
        >>>
        >>> client = OpenRouterClient()
        >>> runner = EvalRunner(client)
        >>>
        >>> tasks = [EvalTask("haiku", Haiku(), "Write a haiku about nature")]
        >>> results = runner.run(tasks, EvalConfig(num_samples=3))
        >>> print(results.summary())
    """

    def __init__(self, client: OpenRouterClient) -> None:
        """
        Initialize runner.

        Args:
            client: OpenRouter client for LLM inference
        """
        self.client = client

    def run(
        self,
        tasks: list[EvalTask],
        config: EvalConfig | None = None,
    ) -> EvalResult:
        """
        Run evaluation synchronously.

        Args:
            tasks: List of eval tasks (form + prompt pairs)
            config: Evaluation configuration

        Returns:
            EvalResult with all metrics
        """
        config = config or EvalConfig()
        start_time = time.time()

        all_samples: list[SampleResult] = []
        form_samples: dict[str, list[SampleResult]] = {}

        total_tasks = len(tasks) * config.num_samples
        current = 0

        for task in tasks:
            if task.form_name not in form_samples:
                form_samples[task.form_name] = []

            for sample_idx in range(config.num_samples):
                current += 1
                if config.verbose:
                    print(
                        f"[{current}/{total_tasks}] {task.form_name}: "
                        f"Generating sample {sample_idx + 1}/{config.num_samples}...",
                        flush=True,
                    )

                sample = self._run_single(task, config)
                all_samples.append(sample)
                form_samples[task.form_name].append(sample)

                if config.verbose:
                    status = "PASS" if sample.passed else "FAIL"
                    if sample.error:
                        status = f"ERROR: {sample.error[:50]}"
                    print(
                        f"         -> score={sample.score:.2%} [{status}] "
                        f"({sample.generation_time:.1f}s)",
                        flush=True,
                    )

        total_time = time.time() - start_time

        return self._aggregate_results(all_samples, form_samples, config, total_time)

    async def arun(
        self,
        tasks: list[EvalTask],
        config: EvalConfig | None = None,
        concurrency: int = 5,
    ) -> EvalResult:
        """
        Run evaluation asynchronously.

        Args:
            tasks: List of eval tasks
            config: Evaluation configuration
            concurrency: Maximum concurrent requests

        Returns:
            EvalResult with all metrics
        """
        config = config or EvalConfig()
        start_time = time.time()

        # Create all sample tasks
        sample_tasks = [(task, i) for task in tasks for i in range(config.num_samples)]

        # Run with concurrency limit
        semaphore = asyncio.Semaphore(concurrency)
        all_samples = await asyncio.gather(
            *[self._arun_single_with_semaphore(task, config, semaphore) for task, _ in sample_tasks]
        )

        total_time = time.time() - start_time

        # Group by form
        form_samples: dict[str, list[SampleResult]] = {}
        for sample in all_samples:
            if sample.form_name not in form_samples:
                form_samples[sample.form_name] = []
            form_samples[sample.form_name].append(sample)

        return self._aggregate_results(list(all_samples), form_samples, config, total_time)

    def _run_single(self, task: EvalTask, config: EvalConfig) -> SampleResult:
        """Run a single generation + verification."""
        start_time = time.time()
        error = None
        generated_text = ""
        score = 0.0
        passed = False
        details: dict[str, Any] = {}

        try:
            generated_text = self.client.generate_poem(
                form_instruction=task.prompt,
                model=config.model,
                system_prompt=config.system_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            result = task.constraint.verify(generated_text)
            score = result.score
            passed = result.passed
            details = {
                "constraint_name": result.constraint_name,
                "constraint_type": result.constraint_type.name,
                "rubric": [str(item) for item in result.rubric],
                "details": result.details,
            }

        except Exception as e:
            error = str(e)

        generation_time = time.time() - start_time

        return SampleResult(
            form_name=task.form_name,
            prompt=task.prompt,
            generated_text=generated_text,
            score=score,
            passed=passed,
            verification_details=details,
            generation_time=generation_time,
            model=config.model,
            error=error,
        )

    async def _arun_single_with_semaphore(
        self,
        task: EvalTask,
        config: EvalConfig,
        semaphore: asyncio.Semaphore,
    ) -> SampleResult:
        """Run single sample with semaphore for concurrency control."""
        async with semaphore:
            return await self._arun_single(task, config)

    async def _arun_single(self, task: EvalTask, config: EvalConfig) -> SampleResult:
        """Run a single generation + verification asynchronously."""
        start_time = time.time()
        error = None
        generated_text = ""
        score = 0.0
        passed = False
        details: dict[str, Any] = {}

        try:
            generated_text = await self.client.agenerate_poem(
                form_instruction=task.prompt,
                model=config.model,
                system_prompt=config.system_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            result = task.constraint.verify(generated_text)
            score = result.score
            passed = result.passed
            details = {
                "constraint_name": result.constraint_name,
                "constraint_type": result.constraint_type.name,
                "rubric": [str(item) for item in result.rubric],
                "details": result.details,
            }

        except Exception as e:
            error = str(e)

        generation_time = time.time() - start_time

        return SampleResult(
            form_name=task.form_name,
            prompt=task.prompt,
            generated_text=generated_text,
            score=score,
            passed=passed,
            verification_details=details,
            generation_time=generation_time,
            model=config.model,
            error=error,
        )

    def _aggregate_results(
        self,
        all_samples: list[SampleResult],
        form_samples: dict[str, list[SampleResult]],
        config: EvalConfig,
        total_time: float,
    ) -> EvalResult:
        """Aggregate sample results into final metrics."""
        form_results: dict[str, FormResult] = {}

        for form_name, samples in form_samples.items():
            valid_samples = [s for s in samples if s.error is None]

            if valid_samples:
                scores = [s.score for s in valid_samples]
                mean_score = sum(scores) / len(scores)
                pass_rate = sum(1 for s in valid_samples if s.passed) / len(valid_samples)
                best_score = max(scores)
                best_sample = max(valid_samples, key=lambda s: s.score)
            else:
                mean_score = 0.0
                pass_rate = 0.0
                best_score = 0.0
                best_sample = None

            form_results[form_name] = FormResult(
                form_name=form_name,
                samples=samples,
                mean_score=mean_score,
                pass_rate=pass_rate,
                best_score=best_score,
                best_sample=best_sample,
            )

        # Overall metrics
        valid_samples = [s for s in all_samples if s.error is None]
        if valid_samples:
            mean_score = sum(s.score for s in valid_samples) / len(valid_samples)
            overall_pass_rate = sum(1 for s in valid_samples if s.passed) / len(valid_samples)
        else:
            mean_score = 0.0
            overall_pass_rate = 0.0

        return EvalResult(
            config=config,
            form_results=form_results,
            total_samples=len(all_samples),
            mean_score=mean_score,
            overall_pass_rate=overall_pass_rate,
            total_time=total_time,
            metadata={
                "error_count": len([s for s in all_samples if s.error is not None]),
            },
        )
