"""
Blues Poem form template.

A poem with AAB tercet structure and line repetition.
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
    rhyme_score,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class BluesPoem(Constraint):
    """
    Blues Poem: AAB tercets with line repetition.

    Structure:
    - Multiple tercets (3-line stanzas)
    - Line 1: statement/question
    - Line 2: repetition/variation of line 1
    - Line 3: response/resolution that rhymes with lines 1-2
    - Each stanza is self-contained

    Famous example:
        "The Weary Blues" by Langston Hughes

    Examples:
        >>> blues = BluesPoem()
        >>> result = blues.verify(poem)
    """

    name = "Blues Poem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        min_stanzas: int = 2,
        repetition_threshold: float = 0.6,
        rhyme_threshold: float = 0.5,
        strict: bool = False,
    ) -> None:
        """
        Initialize blues poem constraint.

        Args:
            weight: Relative weight for composition
            min_stanzas: Minimum number of AAB tercets
            repetition_threshold: Minimum similarity for L1-L2 repetition
            rhyme_threshold: Minimum score for A-B rhyme
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_stanzas = min_stanzas
        self.repetition_threshold = repetition_threshold
        self.rhyme_threshold = rhyme_threshold
        self.strict = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        # Check stanza structure - should be tercets
        num_tercets = 0
        for stanza in structure.stanzas:
            if len(stanza) == 3:
                num_tercets += 1

        # Also check if poem is flat (no stanza breaks) - count by lines
        if structure.stanza_count == 1:
            num_tercets = structure.line_count // 3

        if num_tercets >= self.min_stanzas:
            rubric.append(
                RubricItem(
                    criterion="Minimum tercets",
                    expected=f"at least {self.min_stanzas}",
                    actual=str(num_tercets),
                    score=1.0,
                    passed=True,
                )
            )
            scores.append(1.0)
        else:
            # Quadratic penalty for stricter GRPO training
            linear_score = num_tercets / self.min_stanzas if self.min_stanzas > 0 else 0
            rubric.append(
                RubricItem(
                    criterion="Minimum tercets",
                    expected=f"at least {self.min_stanzas}",
                    actual=str(num_tercets),
                    score=linear_score**2,
                    passed=False,
                )
            )
            scores.append(linear_score**2)

        # Check AAB pattern in each tercet
        # If stanzas are defined, use them; otherwise chunk by 3 lines
        tercets = []
        if structure.stanza_count > 1:
            tercets = [s for s in structure.stanzas if len(s) >= 3]
        else:
            # Chunk lines into groups of 3
            for i in range(0, len(structure.lines) - 2, 3):
                tercets.append((structure.lines[i], structure.lines[i + 1], structure.lines[i + 2]))

        repetition_scores = []
        rhyme_scores = []

        for i, tercet in enumerate(tercets):
            if len(tercet) >= 3:
                line1, line2, line3 = tercet[0], tercet[1], tercet[2]

                # Check L1-L2 repetition/variation
                rep_score = self._line_similarity(line1, line2)
                rep_passed = rep_score >= self.repetition_threshold

                rubric.append(
                    RubricItem(
                        criterion=f"Stanza {i + 1}: L1-L2 repetition",
                        expected=f"similarity >= {self.repetition_threshold}",
                        actual=f"{rep_score:.2f}",
                        score=rep_score,
                        passed=rep_passed,
                        explanation=f"'{line1[:30]}...' vs '{line2[:30]}...'",
                    )
                )
                repetition_scores.append(rep_score)

                # Check that L3 rhymes with L1 and L2
                end1 = self._get_end_word(line1)
                end2 = self._get_end_word(line2)
                end3 = self._get_end_word(line3)

                rhyme_1_3 = rhyme_score(end1, end3) if end1 and end3 else 0.0
                rhyme_2_3 = rhyme_score(end2, end3) if end2 and end3 else 0.0
                avg_rhyme = (rhyme_1_3 + rhyme_2_3) / 2

                rubric.append(
                    RubricItem(
                        criterion=f"Stanza {i + 1}: L3 rhymes with L1/L2",
                        expected=f"rhyme >= {self.rhyme_threshold}",
                        actual=f"'{end1}'/'{end2}' vs '{end3}' = {avg_rhyme:.2f}",
                        score=avg_rhyme,
                        passed=avg_rhyme >= self.rhyme_threshold,
                    )
                )
                rhyme_scores.append(avg_rhyme)

        if repetition_scores:
            scores.append(sum(repetition_scores) / len(repetition_scores))
        if rhyme_scores:
            scores.append(sum(rhyme_scores) / len(rhyme_scores))

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric) if self.strict else overall_score >= 0.5

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"tercet_count": len(tercets)},
        )

    def _line_similarity(self, line1: str, line2: str) -> float:
        """Compute similarity between two lines."""
        l1 = self._normalize_line(line1)
        l2 = self._normalize_line(line2)

        if l1 == l2:
            return 1.0

        # For blues, we expect variation so use a more lenient metric
        lev = normalized_levenshtein(l1, l2)
        jw = jaro_winkler_similarity(l1, l2)

        # Also check if one line contains most of the other
        words1 = set(l1.split())
        words2 = set(l2.split())
        overlap = len(words1 & words2) / max(len(words1), len(words2)) if words1 and words2 else 0.0

        return max(lev, jw, overlap)

    def _normalize_line(self, line: str) -> str:
        """Normalize line for comparison."""
        import re

        line = line.lower().strip()
        line = re.sub(r"[^\w\s']", "", line)
        return " ".join(line.split())

    def _get_end_word(self, line: str) -> str:
        """Get last word of line."""
        import re

        words = re.findall(r"\b\w+\b", line)
        return words[-1].lower() if words else ""

    def describe(self) -> str:
        return f"Blues Poem: {self.min_stanzas}+ AAB tercets with line repetition"
