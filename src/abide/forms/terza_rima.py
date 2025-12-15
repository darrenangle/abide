"""
Terza Rima form template.

A poem with interlocking tercets in ABA BCB CDC pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    Constraint,
    ConstraintType,
    RubricItem,
    VerificationResult,
)
from abide.primitives import extract_end_words, rhyme_score

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class TerzaRima(Constraint):
    """
    Terza Rima: interlocking tercets with ABA BCB CDC chain rhyme.

    Structure:
    - Multiple tercets (3-line stanzas)
    - Rhyme scheme: ABA BCB CDC DED...
    - Middle line of each tercet rhymes with outer lines of next
    - Typically ends with single line or couplet

    Famous example:
        "Ode to the West Wind" by Percy Bysshe Shelley

    Examples:
        >>> terza_rima = TerzaRima()
        >>> result = terza_rima.verify(poem)
    """

    name = "Terza Rima"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        min_tercets: int = 3,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize terza rima constraint.

        Args:
            weight: Relative weight for composition
            min_tercets: Minimum number of tercets required
            rhyme_threshold: Minimum score for rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_tercets = min_tercets
        self.rhyme_threshold = rhyme_threshold
        self.strict = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        end_words = extract_end_words(structure)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        # Check we have enough lines for tercets
        # Minimum: min_tercets * 3 lines, possibly +1 or +2 for ending
        min_lines = self.min_tercets * 3
        if structure.line_count < min_lines:
            rubric.append(
                RubricItem(
                    criterion="Minimum lines",
                    expected=f"at least {min_lines}",
                    actual=str(structure.line_count),
                    score=(structure.line_count / min_lines) ** 2,  # Quadratic
                    passed=False,
                )
            )
            scores.append((structure.line_count / min_lines) ** 2)
        else:
            rubric.append(
                RubricItem(
                    criterion="Minimum lines",
                    expected=f"at least {min_lines}",
                    actual=str(structure.line_count),
                    score=1.0,
                    passed=True,
                )
            )
            scores.append(1.0)

        # Check ABA pattern within each tercet
        num_tercets = structure.line_count // 3
        tercet_scores = []

        for i in range(num_tercets):
            base = i * 3
            if base + 2 < len(end_words):
                word_a1 = end_words[base]  # Line 1
                word_b = end_words[base + 1]  # Line 2 (middle)
                word_a2 = end_words[base + 2]  # Line 3

                # Lines 1 and 3 should rhyme (A...A)
                score_aa = rhyme_score(word_a1, word_a2)
                passed_aa = score_aa >= self.rhyme_threshold

                rubric.append(
                    RubricItem(
                        criterion=f"Tercet {i + 1}: L1 rhymes with L3",
                        expected=f"rhyme >= {self.rhyme_threshold}",
                        actual=f"'{word_a1}' / '{word_a2}' = {score_aa:.2f}",
                        score=score_aa,
                        passed=passed_aa,
                    )
                )
                tercet_scores.append(score_aa)

        if tercet_scores:
            scores.append(sum(tercet_scores) / len(tercet_scores))

        # Check chain rhyme (middle of tercet N rhymes with outer of tercet N+1)
        chain_scores = []
        for i in range(num_tercets - 1):
            base_current = i * 3
            base_next = (i + 1) * 3

            if base_current + 1 < len(end_words) and base_next < len(end_words):
                word_b = end_words[base_current + 1]  # Middle of current
                word_c1 = end_words[base_next]  # First of next
                word_c2 = end_words[base_next + 2] if base_next + 2 < len(end_words) else ""

                # B should rhyme with C (outer lines of next tercet)
                if word_c2:
                    score_bc1 = rhyme_score(word_b, word_c1)
                    score_bc2 = rhyme_score(word_b, word_c2)
                    avg_score = (score_bc1 + score_bc2) / 2

                    rubric.append(
                        RubricItem(
                            criterion=f"Chain: Tercet {i + 1} middle -> Tercet {i + 2} outer",
                            expected=f"'{word_b}' rhymes with '{word_c1}'/'{word_c2}'",
                            actual=f"scores: {score_bc1:.2f}, {score_bc2:.2f}",
                            score=avg_score,
                            passed=avg_score >= self.rhyme_threshold,
                        )
                    )
                    chain_scores.append(avg_score)

        if chain_scores:
            scores.append(sum(chain_scores) / len(chain_scores))

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric) if self.strict else overall_score >= 0.6

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "line_count": structure.line_count,
                "tercet_count": num_tercets,
            },
        )

    def describe(self) -> str:
        return f"Terza Rima: {self.min_tercets}+ tercets with ABA BCB chain rhyme"
