"""
Pantoum form template.

A poem of interlocking quatrains with line repetition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    Constraint,
    ConstraintType,
    RubricItem,
    VerificationResult,
)
from abide.primitives import (
    jaro_winkler_similarity,
    normalized_levenshtein,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Pantoum(Constraint):
    """
    Pantoum: quatrains with interlocking line repetitions.

    Structure:
    - Multiple quatrains (4-line stanzas), typically 4+
    - Line 2 of stanza N becomes line 1 of stanza N+1
    - Line 4 of stanza N becomes line 3 of stanza N+1
    - Final stanza: lines 1 and 3 of first stanza return
    - Rhyme scheme: ABAB per stanza (optional)

    Examples:
        >>> pantoum = Pantoum()
        >>> result = pantoum.verify(poem)
    """

    name = "Pantoum"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        min_stanzas: int = 4,
        refrain_threshold: float = 0.85,
        check_circular: bool = True,
        strict: bool = False,
    ) -> None:
        """
        Initialize pantoum constraint.

        Args:
            weight: Relative weight for composition
            min_stanzas: Minimum number of quatrains required
            refrain_threshold: Minimum similarity for line repetitions
            check_circular: Whether to check that final stanza closes the loop
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_stanzas = min_stanzas
        self.refrain_threshold = refrain_threshold
        self.check_circular = check_circular
        self.strict = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        # Check minimum stanza count
        if structure.stanza_count < self.min_stanzas:
            # Quadratic penalty for stricter GRPO training
            linear_score = structure.stanza_count / self.min_stanzas
            rubric.append(
                RubricItem(
                    criterion="Minimum stanzas",
                    expected=f"at least {self.min_stanzas}",
                    actual=str(structure.stanza_count),
                    score=linear_score**2,
                    passed=False,
                )
            )
            scores.append(linear_score**2)
        else:
            rubric.append(
                RubricItem(
                    criterion="Minimum stanzas",
                    expected=f"at least {self.min_stanzas}",
                    actual=str(structure.stanza_count),
                    score=1.0,
                    passed=True,
                )
            )
            scores.append(1.0)

        # Check all stanzas are quatrains
        quatrain_scores = []
        for i, stanza in enumerate(structure.stanzas):
            if len(stanza) == 4:
                quatrain_scores.append(1.0)
            else:
                quatrain_scores.append(0.5)
                rubric.append(
                    RubricItem(
                        criterion=f"Stanza {i + 1} is quatrain",
                        expected="4 lines",
                        actual=f"{len(stanza)} lines",
                        score=0.5,
                        passed=False,
                    )
                )
        if quatrain_scores:
            avg_quatrain = sum(quatrain_scores) / len(quatrain_scores)
            scores.append(avg_quatrain)

        # Check line repetitions between stanzas
        repetition_scores = []
        for i in range(len(structure.stanzas) - 1):
            current = structure.stanzas[i]
            next_stanza = structure.stanzas[i + 1]

            if len(current) >= 4 and len(next_stanza) >= 4:
                # Line 2 of current -> Line 1 of next
                score_2_to_1 = self._line_similarity(current[1], next_stanza[0])
                passed_2_to_1 = score_2_to_1 >= self.refrain_threshold
                rubric.append(
                    RubricItem(
                        criterion=f"Stanza {i + 1} L2 -> Stanza {i + 2} L1",
                        expected=current[1][:40],
                        actual=next_stanza[0][:40],
                        score=score_2_to_1,
                        passed=passed_2_to_1,
                    )
                )
                repetition_scores.append(score_2_to_1)

                # Line 4 of current -> Line 3 of next
                score_4_to_3 = self._line_similarity(current[3], next_stanza[2])
                passed_4_to_3 = score_4_to_3 >= self.refrain_threshold
                rubric.append(
                    RubricItem(
                        criterion=f"Stanza {i + 1} L4 -> Stanza {i + 2} L3",
                        expected=current[3][:40],
                        actual=next_stanza[2][:40],
                        score=score_4_to_3,
                        passed=passed_4_to_3,
                    )
                )
                repetition_scores.append(score_4_to_3)

        if repetition_scores:
            scores.append(sum(repetition_scores) / len(repetition_scores))

        # Check circular closing (optional)
        if self.check_circular and len(structure.stanzas) >= 2:
            first = structure.stanzas[0]
            last = structure.stanzas[-1]

            if len(first) >= 4 and len(last) >= 4:
                # Last stanza L2 should match first stanza L1
                score_close_1 = self._line_similarity(last[1], first[0])
                # Last stanza L4 should match first stanza L3
                score_close_3 = self._line_similarity(last[3], first[2])

                rubric.append(
                    RubricItem(
                        criterion="Circular close: last L2 = first L1",
                        expected=first[0][:40],
                        actual=last[1][:40],
                        score=score_close_1,
                        passed=score_close_1 >= self.refrain_threshold,
                    )
                )
                rubric.append(
                    RubricItem(
                        criterion="Circular close: last L4 = first L3",
                        expected=first[2][:40],
                        actual=last[3][:40],
                        score=score_close_3,
                        passed=score_close_3 >= self.refrain_threshold,
                    )
                )
                scores.append((score_close_1 + score_close_3) / 2)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric) if self.strict else overall_score >= 0.6

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"stanza_count": structure.stanza_count},
        )

    def _line_similarity(self, line1: str, line2: str) -> float:
        """Compute similarity between two lines."""
        l1 = self._normalize_line(line1)
        l2 = self._normalize_line(line2)

        if l1 == l2:
            return 1.0

        lev = normalized_levenshtein(l1, l2)
        jw = jaro_winkler_similarity(l1, l2)
        return (lev + jw) / 2

    def _normalize_line(self, line: str) -> str:
        """Normalize line for comparison."""
        import re

        line = line.lower().strip()
        line = re.sub(r"[^\w\s']", "", line)
        return " ".join(line.split())

    def describe(self) -> str:
        return f"Pantoum: {self.min_stanzas}+ quatrains with interlocking line repetition"
