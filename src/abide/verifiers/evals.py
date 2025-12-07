"""
Pre-built evaluations for poetic form verification.

Provides AbideMajorPoeticForms - a comprehensive eval composed
from all primitive form constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from abide.constraints import Constraint, VerificationResult, WeightedSum
from abide.forms import (
    Haiku,
    Limerick,
    PetrarchanSonnet,
    Sestina,
    ShakespeareanSonnet,
    SpenserianSonnet,
    Tanka,
    Villanelle,
)
from abide.verifiers.reward import PoeticFormReward, RewardOutput


@dataclass
class EvalItem:
    """
    A single item in a poetic forms evaluation.

    Attributes:
        name: Name of the form being tested
        constraint: The form constraint
        weight: Relative weight in aggregate scoring
        description: Human-readable description
    """

    name: str
    constraint: Constraint
    weight: float = 1.0
    description: str = ""


@dataclass
class EvalResult:
    """
    Result of evaluating a poem against multiple forms.

    Attributes:
        poem: The poem that was evaluated
        results: Per-form verification results
        best_match: Name of the best-matching form
        best_score: Score for the best match
        aggregate_score: Weighted aggregate score
    """

    poem: str
    results: dict[str, VerificationResult]
    best_match: str
    best_score: float
    aggregate_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class AbideMajorPoeticForms:
    """
    Comprehensive evaluation of major poetic forms.

    This eval tests poems against all major forms and provides:
    - Per-form scores
    - Best-match identification
    - Aggregate scoring
    - Detailed rubrics

    Forms included:
    - Haiku (5-7-5 syllables)
    - Tanka (5-7-5-7-7 syllables)
    - Villanelle (19 lines, refrains, ABA rhyme)
    - Sestina (39 lines, end-word rotation)
    - Shakespearean Sonnet (14 lines, ABAB CDCD EFEF GG)
    - Petrarchan Sonnet (14 lines, ABBAABBA + sestet)
    - Spenserian Sonnet (14 lines, ABAB BCBC CDCD EE)
    - Limerick (5 lines, AABBA)

    Example:
        >>> eval_suite = AbideMajorPoeticForms()
        >>> result = eval_suite.evaluate(poem)
        >>> print(f"Best match: {result.best_match} ({result.best_score:.2f})")
    """

    def __init__(
        self,
        strict: bool = False,
        syllable_tolerance: int = 1,
        rhyme_threshold: float = 0.6,
    ) -> None:
        """
        Initialize the major poetic forms eval.

        Args:
            strict: If True, use strict mode for all forms
            syllable_tolerance: Syllable tolerance for syllable-based forms
            rhyme_threshold: Rhyme threshold for rhyme-based forms
        """
        self.strict = strict
        self.syllable_tolerance = syllable_tolerance
        self.rhyme_threshold = rhyme_threshold

        # Build eval items
        self._items = self._build_items()

    def _build_items(self) -> list[EvalItem]:
        """Build the list of eval items."""
        return [
            EvalItem(
                name="haiku",
                constraint=Haiku(
                    syllable_tolerance=self.syllable_tolerance,
                    strict=self.strict,
                ),
                weight=1.0,
                description="Japanese 3-line poem with 5-7-5 syllable pattern",
            ),
            EvalItem(
                name="tanka",
                constraint=Tanka(
                    syllable_tolerance=self.syllable_tolerance,
                    strict=self.strict,
                ),
                weight=1.0,
                description="Japanese 5-line poem with 5-7-5-7-7 syllable pattern",
            ),
            EvalItem(
                name="villanelle",
                constraint=Villanelle(
                    rhyme_threshold=self.rhyme_threshold,
                    strict=self.strict,
                ),
                weight=1.5,
                description="19-line poem with two refrains and ABA rhyme scheme",
            ),
            EvalItem(
                name="sestina",
                constraint=Sestina(
                    strict=self.strict,
                ),
                weight=2.0,
                description="39-line poem with end-word rotation pattern",
            ),
            EvalItem(
                name="shakespearean_sonnet",
                constraint=ShakespeareanSonnet(
                    syllable_tolerance=self.syllable_tolerance,
                    rhyme_threshold=self.rhyme_threshold,
                    strict=self.strict,
                ),
                weight=1.5,
                description="14-line poem with ABAB CDCD EFEF GG rhyme scheme",
            ),
            EvalItem(
                name="petrarchan_sonnet",
                constraint=PetrarchanSonnet(
                    syllable_tolerance=self.syllable_tolerance,
                    rhyme_threshold=self.rhyme_threshold,
                    strict=self.strict,
                ),
                weight=1.5,
                description="14-line poem with ABBAABBA + sestet rhyme scheme",
            ),
            EvalItem(
                name="spenserian_sonnet",
                constraint=SpenserianSonnet(
                    syllable_tolerance=self.syllable_tolerance,
                    rhyme_threshold=self.rhyme_threshold,
                    strict=self.strict,
                ),
                weight=1.5,
                description="14-line poem with ABAB BCBC CDCD EE rhyme scheme",
            ),
            EvalItem(
                name="limerick",
                constraint=Limerick(
                    rhyme_threshold=self.rhyme_threshold,
                    strict=self.strict,
                ),
                weight=1.0,
                description="5-line humorous poem with AABBA rhyme scheme",
            ),
        ]

    @property
    def forms(self) -> list[str]:
        """List of form names in this eval."""
        return [item.name for item in self._items]

    def __iter__(self) -> Iterator[EvalItem]:
        """Iterate over eval items."""
        return iter(self._items)

    def __len__(self) -> int:
        """Number of forms in this eval."""
        return len(self._items)

    def evaluate(self, poem: str) -> EvalResult:
        """
        Evaluate a poem against all major forms.

        Args:
            poem: The poem text to evaluate

        Returns:
            EvalResult with per-form scores and best match
        """
        results: dict[str, VerificationResult] = {}
        scores: dict[str, float] = {}

        for item in self._items:
            result = item.constraint.verify(poem)
            results[item.name] = result
            scores[item.name] = result.score

        # Find best match
        best_match = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best_match]

        # Compute weighted aggregate
        total_weight = sum(item.weight for item in self._items)
        weighted_sum = sum(
            scores[item.name] * item.weight
            for item in self._items
        )
        aggregate_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return EvalResult(
            poem=poem,
            results=results,
            best_match=best_match,
            best_score=best_score,
            aggregate_score=aggregate_score,
            metadata={
                "all_scores": scores,
                "form_count": len(self._items),
            },
        )

    def get_reward_function(self, form_name: str) -> PoeticFormReward:
        """
        Get a reward function for a specific form.

        Args:
            form_name: Name of the form (e.g., "haiku", "villanelle")

        Returns:
            PoeticFormReward for the specified form

        Raises:
            ValueError: If form_name is not in this eval
        """
        for item in self._items:
            if item.name == form_name:
                return PoeticFormReward(item.constraint, name=form_name)
        raise ValueError(f"Unknown form: {form_name}. Available: {self.forms}")

    def get_all_reward_functions(self) -> dict[str, PoeticFormReward]:
        """Get reward functions for all forms."""
        return {
            item.name: PoeticFormReward(item.constraint, name=item.name)
            for item in self._items
        }

    def describe(self) -> str:
        """Get description of this eval."""
        lines = ["Abide Major Poetic Forms Evaluation", "=" * 40]
        for item in self._items:
            lines.append(f"\n{item.name} (weight={item.weight})")
            lines.append(f"  {item.description}")
        return "\n".join(lines)


def make_poetic_forms_eval(
    forms: Sequence[str] | None = None,
    **kwargs: Any,
) -> AbideMajorPoeticForms:
    """
    Factory function to create a poetic forms eval.

    Args:
        forms: Optional list of form names to include (default: all)
        **kwargs: Additional arguments for AbideMajorPoeticForms

    Returns:
        Configured AbideMajorPoeticForms instance
    """
    eval_suite = AbideMajorPoeticForms(**kwargs)

    if forms is not None:
        # Filter to requested forms
        eval_suite._items = [
            item for item in eval_suite._items
            if item.name in forms
        ]

    return eval_suite
