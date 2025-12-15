"""
Ballade form template.

A 28-line poem with envoi and refrain.
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
    normalized_levenshtein,
    rhyme_score,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Ballade(Constraint):
    """
    Ballade: 28 lines with 3 octaves + envoi, refrain at stanza ends.

    Structure:
    - 3 stanzas of 8 lines (octaves) + 1 envoi of 4 lines
    - Rhyme scheme per octave: ABABBCBC
    - Envoi rhyme: BCBC
    - Last line of each stanza is refrain (same line)
    - Only 3 rhyme sounds used throughout (A, B, C)
    - Envoi traditionally addressed to "Prince"

    Famous example:
        "Ballade des Dames du Temps Jadis" by François Villon

    Examples:
        >>> ballade = Ballade()
        >>> result = ballade.verify(poem)
    """

    name = "Ballade"
    constraint_type = ConstraintType.COMPOSITE

    LINE_COUNT = 28
    OCTAVE_SCHEME = "ABABBCBC"
    ENVOI_SCHEME = "BCBC"

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        refrain_threshold: float = 0.9,
        strict: bool = False,
    ) -> None:
        """
        Initialize ballade constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum score for rhymes
            refrain_threshold: Minimum similarity for refrain lines
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

        # Check line count
        if structure.line_count == self.LINE_COUNT:
            rubric.append(
                RubricItem(
                    criterion="Line count",
                    expected=str(self.LINE_COUNT),
                    actual=str(structure.line_count),
                    score=1.0,
                    passed=True,
                )
            )
            scores.append(1.0)
        elif structure.line_count >= 24:
            rubric.append(
                RubricItem(
                    criterion="Line count",
                    expected=str(self.LINE_COUNT),
                    actual=str(structure.line_count),
                    score=0.8,
                    passed=False,
                )
            )
            scores.append(0.8)
        else:
            # Quadratic penalty for stricter GRPO training
            linear_score = structure.line_count / self.LINE_COUNT
            rubric.append(
                RubricItem(
                    criterion="Line count",
                    expected=str(self.LINE_COUNT),
                    actual=str(structure.line_count),
                    score=linear_score**2,
                    passed=False,
                )
            )
            scores.append(linear_score**2)

        if structure.line_count < 8:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=rubric,
                constraint_name=self.name,
                constraint_type=self.constraint_type,
            )

        # Check refrain (last line of each octave + envoi should be identical)
        # Lines 8, 16, 24, 28 (0-indexed: 7, 15, 23, 27)
        refrain_positions = [7, 15, 23, 27]
        refrain_lines = []

        for pos in refrain_positions:
            if pos < len(structure.lines):
                refrain_lines.append(structure.lines[pos])

        if len(refrain_lines) >= 2:
            refrain_scores = []
            base_refrain = refrain_lines[0]

            for i, line in enumerate(refrain_lines[1:], 1):
                similarity = self._line_similarity(base_refrain, line)
                passed = similarity >= self.refrain_threshold

                rubric.append(
                    RubricItem(
                        criterion=f"Refrain at position {refrain_positions[i] + 1}",
                        expected=base_refrain[:40],
                        actual=line[:40],
                        score=similarity,
                        passed=passed,
                    )
                )
                refrain_scores.append(similarity)

            if refrain_scores:
                scores.append(sum(refrain_scores) / len(refrain_scores))

        # Check rhyme scheme for octaves
        end_words = extract_end_words(structure)

        # Build expected rhyme groups from ABABBCBC pattern
        # Octave 1: lines 0-7, Octave 2: lines 8-15, Octave 3: lines 16-23
        # A positions in octave: 0, 2
        # B positions in octave: 1, 3, 4, 6
        # C positions in octave: 5, 7

        all_a_words = []
        all_b_words = []
        all_c_words = []

        for octave in range(3):
            base = octave * 8
            if base + 7 < len(end_words):
                all_a_words.extend([end_words[base], end_words[base + 2]])
                all_b_words.extend(
                    [
                        end_words[base + 1],
                        end_words[base + 3],
                        end_words[base + 4],
                        end_words[base + 6],
                    ]
                )
                all_c_words.extend([end_words[base + 5], end_words[base + 7]])

        # Add envoi rhymes (BCBC at lines 24-27)
        envoi_base = 24
        if envoi_base + 3 < len(end_words):
            all_b_words.extend([end_words[envoi_base], end_words[envoi_base + 2]])
            all_c_words.extend([end_words[envoi_base + 1], end_words[envoi_base + 3]])

        # Check A rhymes
        if len(all_a_words) >= 2:
            a_scores = self._check_rhyme_group(all_a_words)
            avg_a = sum(a_scores) / len(a_scores) if a_scores else 0.0
            rubric.append(
                RubricItem(
                    criterion="A rhymes consistent",
                    expected="all A positions rhyme",
                    actual=f"avg: {avg_a:.2f}, words: {all_a_words[:4]}",
                    score=avg_a,
                    passed=avg_a >= self.rhyme_threshold,
                )
            )
            scores.append(avg_a)

        # Check B rhymes
        if len(all_b_words) >= 2:
            b_scores = self._check_rhyme_group(all_b_words)
            avg_b = sum(b_scores) / len(b_scores) if b_scores else 0.0
            rubric.append(
                RubricItem(
                    criterion="B rhymes consistent",
                    expected="all B positions rhyme",
                    actual=f"avg: {avg_b:.2f}, words: {all_b_words[:4]}",
                    score=avg_b,
                    passed=avg_b >= self.rhyme_threshold,
                )
            )
            scores.append(avg_b)

        # Check C rhymes
        if len(all_c_words) >= 2:
            c_scores = self._check_rhyme_group(all_c_words)
            avg_c = sum(c_scores) / len(c_scores) if c_scores else 0.0
            rubric.append(
                RubricItem(
                    criterion="C rhymes consistent",
                    expected="all C positions rhyme",
                    actual=f"avg: {avg_c:.2f}, words: {all_c_words[:4]}",
                    score=avg_c,
                    passed=avg_c >= self.rhyme_threshold,
                )
            )
            scores.append(avg_c)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric) if self.strict else overall_score >= 0.6

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "refrain": refrain_lines[0] if refrain_lines else None,
            },
        )

    def _check_rhyme_group(self, words: list[str]) -> list[float]:
        """Check that all words in a group rhyme with each other."""
        if len(words) < 2:
            return []

        scores = []
        base = words[0]
        for word in words[1:]:
            scores.append(rhyme_score(base, word))
        return scores

    def _line_similarity(self, line1: str, line2: str) -> float:
        """Compute similarity between two lines."""
        l1 = line1.lower().strip()
        l2 = line2.lower().strip()

        if l1 == l2:
            return 1.0

        lev = normalized_levenshtein(l1, l2)
        jw = jaro_winkler_similarity(l1, l2)
        return (lev + jw) / 2

    def describe(self) -> str:
        return "Ballade: 28 lines (3 octaves + envoi) with ABABBCBC rhyme and refrain"
