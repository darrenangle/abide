"""
Rondeau form template.

A 15-line poem with rentrement (refrain from opening).
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
    extract_end_words,
    jaro_winkler_similarity,
    rhyme_score,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Rondeau(Constraint):
    """
    Rondeau: 15 lines with AABBA AABR AABBAR rhyme and rentrement.

    Structure:
    - 15 lines in 3 stanzas (5-4-6 or presented as single block)
    - Rhyme scheme: AABBA AABR AABBAR
    - R = rentrement (first phrase of line 1, unrhymed)
    - Rentrement appears at lines 9 and 15
    - Only two rhyme sounds used (A and B)

    Famous example:
        "In Flanders Fields" by John McCrae (variant)

    Examples:
        >>> rondeau = Rondeau()
        >>> result = rondeau.verify(poem)
    """

    name = "Rondeau"
    constraint_type = ConstraintType.COMPOSITE

    # Full rhyme scheme including R for rentrement positions
    RHYME_SCHEME = "AABBAAABXAABBAAX"  # X marks rentrement (don't check rhyme)
    LINE_COUNT = 15

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        refrain_threshold: float = 0.8,
        strict: bool = False,
    ) -> None:
        """
        Initialize rondeau constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum score for rhymes
            refrain_threshold: Minimum similarity for rentrement
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_threshold = rhyme_threshold
        self.refrain_threshold = refrain_threshold
        self.strict = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        # Check line count (should be 15, or 13 if rentrement counted separately)
        expected_lines = self.LINE_COUNT
        if structure.line_count == expected_lines:
            rubric.append(
                RubricItem(
                    criterion="Line count",
                    expected=str(expected_lines),
                    actual=str(structure.line_count),
                    score=1.0,
                    passed=True,
                )
            )
            scores.append(1.0)
        elif structure.line_count >= 13:  # Allow some flexibility
            rubric.append(
                RubricItem(
                    criterion="Line count",
                    expected=str(expected_lines),
                    actual=str(structure.line_count),
                    score=0.8,
                    passed=False,
                )
            )
            scores.append(0.8)
        else:
            rubric.append(
                RubricItem(
                    criterion="Line count",
                    expected=str(expected_lines),
                    actual=str(structure.line_count),
                    score=structure.line_count / expected_lines,
                    passed=False,
                )
            )
            scores.append(structure.line_count / expected_lines)

        if structure.line_count < 3:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=rubric,
                constraint_name=self.name,
                constraint_type=self.constraint_type,
            )

        # Extract rentrement (first few words of line 1)
        first_line = structure.lines[0]
        rentrement = self._extract_rentrement(first_line)

        if rentrement:
            rubric.append(
                RubricItem(
                    criterion="Rentrement identified",
                    expected="opening phrase",
                    actual=f"'{rentrement}'",
                    score=1.0,
                    passed=True,
                )
            )
            scores.append(1.0)

        # Check rentrement positions (lines 9 and 15, 0-indexed: 8 and 14)
        rentrement_positions = [8, 14]
        rentrement_scores = []

        for pos in rentrement_positions:
            if pos < len(structure.lines):
                line = structure.lines[pos]
                similarity = self._check_rentrement(line, rentrement)
                passed = similarity >= self.refrain_threshold

                rubric.append(
                    RubricItem(
                        criterion=f"Rentrement at line {pos + 1}",
                        expected=f"contains '{rentrement}'",
                        actual=line[:40] if len(line) > 40 else line,
                        score=similarity,
                        passed=passed,
                    )
                )
                rentrement_scores.append(similarity)

        if rentrement_scores:
            scores.append(sum(rentrement_scores) / len(rentrement_scores))

        # Check rhyme scheme (AABBA AAB AABBA pattern, excluding rentrement)
        # Lines by rhyme: A=1,2,5,6,7,10,11,14 B=3,4,8,12,13
        end_words = extract_end_words(structure)

        # A rhymes (0-indexed: 0,1,4,5,6,9,10,13)
        a_positions = [0, 1, 4, 5, 6, 9, 10, 13]
        a_words = [end_words[i] for i in a_positions if i < len(end_words)]

        # B rhymes (0-indexed: 2,3,7,11,12)
        b_positions = [2, 3, 7, 11, 12]
        b_words = [end_words[i] for i in b_positions if i < len(end_words)]

        # Check A rhymes
        if len(a_words) >= 2:
            a_scores = []
            base_a = a_words[0]
            for word in a_words[1:]:
                score = rhyme_score(base_a, word)
                a_scores.append(score)

            avg_a = sum(a_scores) / len(a_scores)
            rubric.append(
                RubricItem(
                    criterion="A rhymes consistent",
                    expected=f"all rhyme with '{base_a}'",
                    actual=f"avg score: {avg_a:.2f}",
                    score=avg_a,
                    passed=avg_a >= self.rhyme_threshold,
                )
            )
            scores.append(avg_a)

        # Check B rhymes
        if len(b_words) >= 2:
            b_scores = []
            base_b = b_words[0]
            for word in b_words[1:]:
                score = rhyme_score(base_b, word)
                b_scores.append(score)

            avg_b = sum(b_scores) / len(b_scores)
            rubric.append(
                RubricItem(
                    criterion="B rhymes consistent",
                    expected=f"all rhyme with '{base_b}'",
                    actual=f"avg score: {avg_b:.2f}",
                    score=avg_b,
                    passed=avg_b >= self.rhyme_threshold,
                )
            )
            scores.append(avg_b)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric) if self.strict else overall_score >= 0.6

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"rentrement": rentrement},
        )

    def _extract_rentrement(self, line: str) -> str:
        """Extract rentrement (first phrase) from opening line."""
        words = line.split()
        # Typically 2-4 words
        if len(words) >= 4:
            return " ".join(words[:4])
        elif len(words) >= 2:
            return " ".join(words[:2])
        return line.strip()

    def _check_rentrement(self, line: str, rentrement: str) -> float:
        """Check if line matches or contains the rentrement."""
        line_lower = line.lower().strip()
        rentrement_lower = rentrement.lower().strip()

        # Exact match or contains
        if rentrement_lower in line_lower:
            return 1.0

        # Check if it's the whole line (short refrain line)
        if line_lower.startswith(rentrement_lower[:10]):
            return jaro_winkler_similarity(line_lower, rentrement_lower)

        return jaro_winkler_similarity(line_lower, rentrement_lower)

    def describe(self) -> str:
        return "Rondeau: 15 lines with AABBA AABR AABBAR rhyme and rentrement"
