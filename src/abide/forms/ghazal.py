"""
Ghazal form template.

A poem of couplets with shared radif (refrain) and qafiya (rhyme).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from abide.constraints import (
    Constraint,
    ConstraintType,
    RubricItem,
    VerificationResult,
)
from abide.primitives import rhyme_score

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Ghazal(Constraint):
    """
    Ghazal: couplets with radif (refrain) and qafiya (rhyme).

    Structure:
    - 5-15 couplets (typically 7-12)
    - First couplet (matla): both lines end with qafiya + radif
    - Subsequent couplets: only second line ends with qafiya + radif
    - Pattern: AA BA CA DA EA...

    Examples:
        >>> ghazal = Ghazal()
        >>> result = ghazal.verify(poem)
    """

    name = "Ghazal"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        weight: float = 1.0,
        min_couplets: int = 5,
        max_couplets: int = 15,
        refrain_threshold: float = 0.85,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize ghazal constraint.

        Args:
            weight: Relative weight for composition
            min_couplets: Minimum number of couplets required
            max_couplets: Maximum number of couplets
            refrain_threshold: Minimum similarity for radif matching
            rhyme_threshold: Minimum score for qafiya rhymes
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.min_couplets = min_couplets
        self.max_couplets = max_couplets
        self.refrain_threshold = refrain_threshold
        self.rhyme_threshold = rhyme_threshold
        self.strict = strict

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        has_complete_couplets = structure.line_count % 2 == 0
        rubric.append(
            RubricItem(
                criterion="Complete couplets",
                expected="an even number of lines",
                actual=str(structure.line_count),
                score=1.0 if has_complete_couplets else 0.0,
                passed=has_complete_couplets,
            )
        )
        scores.append(1.0 if has_complete_couplets else 0.0)

        # Check couplet count
        num_couplets = structure.line_count // 2
        couplet_count_passed = self.min_couplets <= num_couplets <= self.max_couplets
        if num_couplets < self.min_couplets:
            rubric.append(
                RubricItem(
                    criterion="Minimum couplets",
                    expected=f"at least {self.min_couplets}",
                    actual=str(num_couplets),
                    score=(num_couplets / self.min_couplets) ** 2,  # Quadratic
                    passed=False,
                )
            )
            scores.append((num_couplets / self.min_couplets) ** 2)
        elif num_couplets > self.max_couplets:
            rubric.append(
                RubricItem(
                    criterion="Maximum couplets",
                    expected=f"at most {self.max_couplets}",
                    actual=str(num_couplets),
                    score=0.8,  # Slight penalty for too many
                    passed=False,
                )
            )
            scores.append(0.8)
        else:
            rubric.append(
                RubricItem(
                    criterion="Couplet count",
                    expected=f"{self.min_couplets}-{self.max_couplets}",
                    actual=str(num_couplets),
                    score=1.0,
                    passed=True,
                )
            )
            scores.append(1.0)

        if num_couplets < 1:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=rubric,
                constraint_name=self.name,
                constraint_type=self.constraint_type,
            )

        # Extract radif from first couplet
        # Radif is the repeated word/phrase at the end of rhyming lines
        first_line = structure.lines[0]
        second_line = structure.lines[1]

        radif = self._extract_radif(first_line, second_line)
        has_radif = radif is not None
        if has_radif:
            rubric.append(
                RubricItem(
                    criterion="Radif detected",
                    expected="repeated ending phrase",
                    actual=f"'{radif}'",
                    score=1.0,
                    passed=True,
                )
            )
            scores.append(1.0)
        else:
            rubric.append(
                RubricItem(
                    criterion="Radif detected",
                    expected="repeated ending phrase",
                    actual="(none found)",
                    score=0.5,
                    passed=False,
                )
            )
            scores.append(0.5)

        # Check first couplet (matla) - both lines should have radif
        matla_passed = False
        if radif is not None:
            matla_passed = self._ends_with_phrase(first_line, radif) and self._ends_with_phrase(
                second_line,
                radif,
            )
        rubric.append(
            RubricItem(
                criterion="Matla (first couplet): both lines have radif",
                expected=f"ends with '{radif}'" if radif else "shared radif",
                actual="both lines match" if matla_passed else "matla radif not established",
                score=1.0 if matla_passed else 0.0,
                passed=matla_passed,
            )
        )
        scores.append(1.0 if matla_passed else 0.0)

        # Check subsequent couplets - second line should have radif
        radif_violations = 0
        radif_total = 0
        for i in range(1, num_couplets):
            line_idx = i * 2 + 1  # Second line of each couplet
            if line_idx < len(structure.lines):
                line = structure.lines[line_idx]
                if radif:
                    has_radif = self._ends_with_phrase(line, radif)
                    score = 1.0 if has_radif else 0.3
                    radif_total += 1
                    if not has_radif:
                        radif_violations += 1
                else:
                    score = 0.5  # No radif to check against

                rubric.append(
                    RubricItem(
                        criterion=f"Couplet {i + 1} L2 has radif",
                        expected=f"ends with '{radif}'" if radif else "radif",
                        actual=line[-30:] if len(line) > 30 else line,
                        score=score,
                        passed=score >= 0.8,
                    )
                )

        # Steep penalty for radif violations
        if radif_total > 0:
            radif_score_final = self._steep_penalty(radif_violations)
            scores.append(radif_score_final)

        # Check qafiya (rhyme before radif) - all rhyming lines should rhyme
        qafiya_words = []
        expected_qafiya_slots = num_couplets + 1
        for i in range(num_couplets):
            # First couplet: both lines
            # Other couplets: only second line
            if i == 0:
                for j in [0, 1]:
                    if j < len(structure.lines):
                        word = self._extract_qafiya(structure.lines[j], radif)
                        if word:
                            qafiya_words.append((i, j, word))
            else:
                line_idx = i * 2 + 1
                if line_idx < len(structure.lines):
                    word = self._extract_qafiya(structure.lines[line_idx], radif)
                    if word:
                        qafiya_words.append((i, 1, word))

        qafiya_presence_passed = len(qafiya_words) == expected_qafiya_slots
        qafiya_presence_score = (
            1.0
            if qafiya_presence_passed
            else self._steep_penalty(expected_qafiya_slots - len(qafiya_words))
        )
        rubric.append(
            RubricItem(
                criterion="Qafiya extracted",
                expected=f"{expected_qafiya_slots} pre-radif words",
                actual=str(len(qafiya_words)),
                score=qafiya_presence_score,
                passed=qafiya_presence_passed,
            )
        )
        scores.append(qafiya_presence_score)

        # Check that all qafiya words rhyme with first one
        qafiya_violations = 0
        qafiya_total = 0
        if len(qafiya_words) >= 2:
            base_word = qafiya_words[0][2]
            for couplet_i, _line_j, word in qafiya_words[1:]:
                score = rhyme_score(base_word, word)
                passed = score >= self.rhyme_threshold
                qafiya_total += 1
                if not passed:
                    qafiya_violations += 1
                rubric.append(
                    RubricItem(
                        criterion=f"Qafiya: couplet {couplet_i + 1}",
                        expected=f"rhymes with '{base_word}'",
                        actual=f"'{word}' = {score:.2f}",
                        score=score,
                        passed=passed,
                    )
                )

            # Steep penalty for qafiya violations
            if qafiya_total > 0:
                qafiya_score_final = self._steep_penalty(qafiya_violations)
                scores.append(qafiya_score_final)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        if not has_complete_couplets:
            # A dangling line breaks the defining couplet structure.
            overall_score *= 0.25
        if not qafiya_presence_passed:
            # Missing qafiya evidence means the poem has not established the
            # defining radif+qafiya structure, even if repeated lines inflate
            # the other heuristics.
            overall_score *= 0.2

        radif_pattern_passed = radif_total == 0 or radif_violations == 0
        qafiya_pattern_passed = qafiya_total > 0 and qafiya_violations == 0
        canonical_requirements_passed = (
            has_complete_couplets
            and couplet_count_passed
            and has_radif
            and matla_passed
            and qafiya_presence_passed
            and radif_pattern_passed
            and qafiya_pattern_passed
        )
        overall_passed = (
            all(r.passed for r in rubric) if self.strict else overall_score >= 0.6
        ) and canonical_requirements_passed

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "couplet_count": num_couplets,
                "radif": radif,
                "has_complete_couplets": has_complete_couplets,
                "canonical_requirements_passed": canonical_requirements_passed,
                "radif_violations": radif_violations,
                "qafiya_violations": qafiya_violations,
            },
        )

    def _steep_penalty(self, violations: int) -> float:
        """Map violations to the steep penalty ladder used for reward shaping."""
        if violations == 0:
            return 1.0
        if violations == 1:
            return 0.5
        if violations == 2:
            return 0.25
        return 0.05

    def _extract_radif(self, line1: str, line2: str) -> str | None:
        """Extract common ending phrase (radif) from two lines."""
        words1 = self._get_words(line1)
        words2 = self._get_words(line2)

        if not words1 or not words2:
            return None

        # Find longest common suffix
        common: list[str] = []
        for w1, w2 in zip(reversed(words1), reversed(words2), strict=False):
            if w1.lower() == w2.lower():
                common.insert(0, w1)
            else:
                break

        if common:
            return " ".join(common)
        return None

    def _ends_with_phrase(self, line: str, phrase: str) -> bool:
        """Check if line ends with the given phrase."""
        line_clean = self._normalize_line(line)
        phrase_clean = phrase.lower().strip()
        return line_clean.endswith(phrase_clean)

    def _extract_qafiya(self, line: str, radif: str | None) -> str | None:
        """Extract qafiya (rhyming word before radif)."""
        words = self._get_words(line)
        if not words:
            return None

        if radif:
            radif_words = radif.lower().split()
            # Remove radif words from end
            while words and radif_words and words[-1].lower() == radif_words[-1].lower():
                words.pop()
                radif_words.pop()

        return words[-1] if words else None

    def _get_words(self, line: str) -> list[str]:
        """Extract words from line."""
        line = re.sub(r"[^\w\s']", "", line)
        return line.split()

    def _normalize_line(self, line: str) -> str:
        """Normalize line for comparison."""
        line = line.lower().strip()
        line = re.sub(r"[^\w\s']", "", line)
        return " ".join(line.split())

    def describe(self) -> str:
        return f"Ghazal: {self.min_couplets}-{self.max_couplets} couplets with radif and qafiya"
