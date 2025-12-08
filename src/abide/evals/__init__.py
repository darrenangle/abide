"""
Evaluation framework for abide poetic forms.

Provides:
- OpenRouter client for LLM inference
- Integration with verifiers framework
- Pre-built eval suites for poetic forms

Quick Start:
    >>> from abide.evals import run_forms_mini_eval
    >>> results = run_forms_mini_eval(model="meta-llama/llama-3.1-8b-instruct")
    >>> print(results.summary())

Verifiers Integration:
    >>> from abide.evals import make_abide_rubric
    >>> from abide.forms import Haiku, Sonnet
    >>> rubric = make_abide_rubric([Haiku(), Sonnet()])
"""

from abide.evals.client import CompletionResponse, Message, OpenRouterClient
from abide.evals.forms_mini import (
    DEFAULT_TOPICS,
    MAJOR_FORMS_MINI,
    FormSpec,
    create_forms_mini_dataset,
    create_forms_mini_tasks,
    quick_test,
    run_forms_mini_eval,
)
from abide.evals.runner import (
    EvalConfig,
    EvalResult,
    EvalRunner,
    EvalTask,
    FormResult,
    SampleResult,
)
from abide.evals.verifiers_compat import (
    make_abide_rubric,
    make_constraint_funcs,
    make_forms_mini_rubric,
    make_pass_func,
    make_reward_func,
)

__all__ = [
    "DEFAULT_TOPICS",
    # Forms mini eval
    "MAJOR_FORMS_MINI",
    "CompletionResponse",
    "EvalConfig",
    "EvalResult",
    # Runner
    "EvalRunner",
    "EvalTask",
    "FormResult",
    "FormSpec",
    "Message",
    # Client
    "OpenRouterClient",
    "SampleResult",
    "create_forms_mini_dataset",
    "create_forms_mini_tasks",
    "make_abide_rubric",
    "make_constraint_funcs",
    "make_forms_mini_rubric",
    "make_pass_func",
    # Verifiers integration
    "make_reward_func",
    "quick_test",
    "run_forms_mini_eval",
]
