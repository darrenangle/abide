"""
Relational constraints for poem verification.

Rhyme schemes, refrains, end-word patterns, acrostics.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from abide.constraints.base import Constraint
from abide.constraints.types import (
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
    from collections.abc import Sequence

    from abide.primitives import PoemStructure


class RhymeScheme(Constraint):
    """
    Constraint on end-rhyme pattern.

    Uses letter notation: A=first rhyme, B=second, etc.
    Same letter means lines should rhyme.

    Examples:
        >>> constraint = RhymeScheme("ABABCDCDEFEFGG")  # Shakespearean sonnet
        >>> constraint = RhymeScheme("ABBAABBA CDECDE")  # Petrarchan sonnet
        >>> constraint = RhymeScheme("AABBA")  # Limerick
    """

    name = "Rhyme Scheme"
    constraint_type = ConstraintType.RELATIONAL

    def __init__(
        self,
        scheme: str,
        weight: float = 1.0,
        threshold: float = 0.7,
        allow_identical: bool = False,
        binary_scoring: bool = False,
    ) -> None:
        """
        Initialize rhyme scheme constraint.

        Args:
            scheme: Rhyme pattern (e.g., "ABAB"). Spaces/punctuation ignored.
            weight: Relative weight for composition
            threshold: Minimum rhyme_score to count as rhyming
            allow_identical: Whether identical words count as rhyming
            binary_scoring: If True, scores are 1.0 if >= threshold, else 0.0.
                This is useful for form inference where we want 100% pass when
                all rhymes meet the threshold.
        """
        super().__init__(weight)
        # Extract only letters, uppercase
        self.scheme = "".join(c.upper() for c in scheme if c.isalpha())
        self.threshold = threshold
        self.allow_identical = allow_identical
        self.binary_scoring = binary_scoring

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        end_words = extract_end_words(structure)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        # Group lines by rhyme letter
        rhyme_groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for i, letter in enumerate(self.scheme):
            if i < len(end_words):
                rhyme_groups[letter].append((i, end_words[i]))

        # Check each rhyme group
        for letter, group in sorted(rhyme_groups.items()):
            if len(group) < 2:
                # Single line in group, automatically passes
                scores.append(1.0)
                continue

            # Check all pairs in the group
            pair_scores: list[float] = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    line_i, word_i = group[i]
                    line_j, word_j = group[j]

                    if word_i.lower() == word_j.lower():
                        # Identical words: full credit if allowed, partial otherwise
                        raw_score = 1.0 if self.allow_identical else 0.5
                    else:
                        raw_score = rhyme_score(word_i, word_j)

                    passed = raw_score >= self.threshold
                    # For binary scoring, use 1.0 or 0.0 based on threshold
                    score = (1.0 if passed else 0.0) if self.binary_scoring else raw_score
                    pair_scores.append(score)

                    rubric.append(
                        RubricItem(
                            criterion=f"Rhyme {letter}: line {line_i + 1} / {line_j + 1}",
                            expected=f"rhyme (threshold {self.threshold})",
                            actual=f"'{word_i}' / '{word_j}' = {raw_score:.2f}",
                            score=score,
                            passed=passed,
                        )
                    )

            if pair_scores:
                scores.append(sum(pair_scores) / len(pair_scores))

        # Check line count matches scheme length
        if len(end_words) != len(self.scheme):
            rubric.insert(
                0,
                RubricItem(
                    criterion="Line count for scheme",
                    expected=str(len(self.scheme)),
                    actual=str(len(end_words)),
                    score=0.5,
                    passed=False,
                    explanation="Poem length doesn't match rhyme scheme",
                ),
            )
            scores.insert(0, 0.5)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"scheme": self.scheme, "end_words": list(end_words)},
        )

    def describe(self) -> str:
        return f"Has rhyme scheme {self.scheme}"

    def instruction(self) -> str:
        """Plain English instruction for LLM prompts."""
        # Format scheme with spaces for readability (e.g., ABAB CDCD -> "A B A B C D C D")
        " ".join(self.scheme)
        return f"Follow the rhyme scheme {self.scheme}, where lines with the same letter must rhyme (e.g., all A lines rhyme with each other)."


class Refrain(Constraint):
    """
    Constraint that specific lines repeat exactly (or nearly).

    Used for villanelles, pantoums, etc.

    Examples:
        >>> # Villanelle: line 1 repeats at 6, 12, 18
        >>> constraint = Refrain(reference_line=0, repeat_at=[5, 11, 17])
    """

    name = "Refrain"
    constraint_type = ConstraintType.RELATIONAL

    def __init__(
        self,
        reference_line: int,
        repeat_at: Sequence[int],
        weight: float = 1.0,
        threshold: float = 0.9,
    ) -> None:
        """
        Initialize refrain constraint.

        Args:
            reference_line: Index of the reference line (0-based)
            repeat_at: Indices where the line should repeat
            weight: Relative weight
            threshold: Minimum similarity for match (default 0.9 for near-exact)
        """
        super().__init__(weight)
        self.reference_line = reference_line
        self.repeat_at = tuple(repeat_at)
        self.threshold = threshold

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        if self.reference_line >= len(structure.lines):
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[
                    RubricItem(
                        criterion="Reference line exists",
                        expected=f"line {self.reference_line + 1}",
                        actual=f"only {len(structure.lines)} lines",
                        score=0.0,
                        passed=False,
                    )
                ],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
            )

        reference = structure.lines[self.reference_line]
        reference_normalized = self._normalize_line(reference)

        for pos in self.repeat_at:
            if pos >= len(structure.lines):
                rubric.append(
                    RubricItem(
                        criterion=f"Refrain at line {pos + 1}",
                        expected=reference[:50] + "..." if len(reference) > 50 else reference,
                        actual="(line missing)",
                        score=0.0,
                        passed=False,
                    )
                )
                scores.append(0.0)
                continue

            repeat = structure.lines[pos]
            repeat_normalized = self._normalize_line(repeat)

            # Exact match
            if reference_normalized == repeat_normalized:
                score = 1.0
            else:
                # Fuzzy match using combination of metrics
                lev_sim = normalized_levenshtein(reference_normalized, repeat_normalized)
                jw_sim = jaro_winkler_similarity(reference_normalized, repeat_normalized)
                score = (lev_sim + jw_sim) / 2

            passed = score >= self.threshold
            rubric.append(
                RubricItem(
                    criterion=f"Refrain at line {pos + 1}",
                    expected=reference[:40] + "..." if len(reference) > 40 else reference,
                    actual=repeat[:40] + "..." if len(repeat) > 40 else repeat,
                    score=score,
                    passed=passed,
                    explanation=f"similarity: {score:.2f}",
                )
            )
            scores.append(score)

        # Calculate overall score with steep exponential penalty
        # 0 violations: 1.0, 1 violation: 0.5, 2 violations: 0.25, 3+ violations: 0.05
        num_violations = sum(1 for r in rubric if not r.passed)
        if num_violations == 0:
            overall_score = 1.0
        elif num_violations == 1:
            overall_score = 0.5
        elif num_violations == 2:
            overall_score = 0.25
        else:
            overall_score = 0.05

        overall_passed = all(r.passed for r in rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
        )

    def _normalize_line(self, line: str) -> str:
        """Normalize line for comparison (lowercase, strip punctuation)."""
        line = line.lower().strip()
        # Remove punctuation but keep apostrophes in contractions
        line = re.sub(r"[^\w\s']", "", line)
        line = " ".join(line.split())
        return line

    def describe(self) -> str:
        positions = ", ".join(str(p + 1) for p in self.repeat_at)
        return f"Line {self.reference_line + 1} repeats at lines {positions}"

    def instruction(self) -> str:
        """Plain English instruction for LLM prompts."""
        positions = ", ".join(str(p + 1) for p in self.repeat_at)
        return f"Line {self.reference_line + 1} must be repeated exactly (as a refrain) at lines {positions}."


class EndWordPattern(Constraint):
    """
    Constraint on end-word patterns (like sestina rotation).

    Verifies that end words from the first stanza rotate through
    subsequent stanzas in a specific pattern.
    """

    name = "End Word Pattern"
    constraint_type = ConstraintType.RELATIONAL

    # Sestina rotation: position i in new stanza gets word from position ROTATION[i]
    SESTINA_ROTATION: ClassVar[list[int]] = [5, 0, 4, 1, 3, 2]

    def __init__(
        self,
        num_words: int = 6,
        num_stanzas: int = 6,
        rotation: Sequence[int] | None = None,
        weight: float = 1.0,
        threshold: float = 0.8,
    ) -> None:
        """
        Initialize end-word pattern constraint.

        Args:
            num_words: Number of end words per stanza
            num_stanzas: Number of stanzas to check
            rotation: Rotation pattern (default: sestina pattern)
            weight: Relative weight
            threshold: Similarity threshold for fuzzy matching
        """
        super().__init__(weight)
        self.num_words = num_words
        self.num_stanzas = num_stanzas
        self.rotation = tuple(rotation) if rotation else tuple(self.SESTINA_ROTATION)
        self.threshold = threshold

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        # Check we have enough stanzas
        if structure.stanza_count < self.num_stanzas:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[
                    RubricItem(
                        criterion="Stanza count",
                        expected=f"at least {self.num_stanzas}",
                        actual=str(structure.stanza_count),
                        score=0.0,
                        passed=False,
                    )
                ],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
            )

        # Extract end words from each stanza
        stanza_end_words: list[list[str]] = []
        for stanza in structure.stanzas[: self.num_stanzas]:
            words = [self._extract_end_word(line) for line in stanza[: self.num_words]]
            stanza_end_words.append(words)

        # Canonical words from first stanza
        canonical = stanza_end_words[0]

        # Check if first stanza has enough words
        if len(canonical) < self.num_words:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[
                    RubricItem(
                        criterion="End words in first stanza",
                        expected=f"at least {self.num_words}",
                        actual=str(len(canonical)),
                        score=0.0,
                        passed=False,
                    )
                ],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
            )

        # Generate expected patterns for each stanza
        expected_patterns = self._generate_patterns()

        # Check each stanza against expected pattern
        for stanza_idx in range(1, self.num_stanzas):
            expected_pattern = expected_patterns[stanza_idx]
            actual_words = stanza_end_words[stanza_idx]

            stanza_scores: list[float] = []
            for pos, expected_word_idx in enumerate(expected_pattern):
                if pos >= len(actual_words):
                    continue

                # Skip if canonical word index is out of range
                if expected_word_idx >= len(canonical):
                    continue

                expected_word = canonical[expected_word_idx]
                actual_word = actual_words[pos]

                score = self._word_similarity(expected_word, actual_word)
                passed = score >= self.threshold

                rubric.append(
                    RubricItem(
                        criterion=f"Stanza {stanza_idx + 1}, position {pos + 1}",
                        expected=expected_word,
                        actual=actual_word,
                        score=score,
                        passed=passed,
                    )
                )
                stanza_scores.append(score)

            if stanza_scores:
                scores.append(sum(stanza_scores) / len(stanza_scores))

        # Calculate overall score with steep exponential penalty
        # 0 violations: 1.0, 1 violation: 0.5, 2 violations: 0.25, 3+ violations: 0.05
        num_violations = sum(1 for r in rubric if not r.passed)
        if num_violations == 0:
            overall_score = 1.0
        elif num_violations == 1:
            overall_score = 0.5
        elif num_violations == 2:
            overall_score = 0.25
        else:
            overall_score = 0.05

        overall_passed = all(r.passed for r in rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"canonical_words": canonical},
        )

    def _generate_patterns(self) -> list[list[int]]:
        """Generate expected word positions for each stanza."""
        patterns: list[list[int]] = []
        current = list(range(self.num_words))
        patterns.append(current.copy())

        for _ in range(1, self.num_stanzas):
            current = [current[i] for i in self.rotation]
            patterns.append(current.copy())

        return patterns

    def _extract_end_word(self, line: str) -> str:
        """Extract end word from line."""
        from abide.primitives import extract_end_word

        return extract_end_word(line)

    def _word_similarity(self, word1: str, word2: str) -> float:
        """Compute similarity between two words."""
        if word1.lower() == word2.lower():
            return 1.0
        return jaro_winkler_similarity(word1.lower(), word2.lower())

    def describe(self) -> str:
        return (
            f"Has {self.num_words}-word end-word rotation pattern across {self.num_stanzas} stanzas"
        )

    def instruction(self) -> str:
        """Plain English instruction for LLM prompts."""
        return (
            f"Use exactly {self.num_words} end words in the first stanza, then rotate these same words "
            f"through all {self.num_stanzas} stanzas following the sestina pattern: each stanza uses "
            f"the same {self.num_words} words but in a different order, with the last word of each stanza "
            f"becoming the end word of the first line in the next stanza."
        )


class Acrostic(Constraint):
    """
    Constraint that first letters of lines spell a word/phrase.

    Examples:
        >>> constraint = Acrostic("LOVE")  # First letters spell LOVE
    """

    name = "Acrostic"
    constraint_type = ConstraintType.RELATIONAL

    def __init__(
        self,
        word: str,
        position: str = "first",  # "first", "last", or "middle"
        weight: float = 1.0,
        case_sensitive: bool = False,
    ) -> None:
        super().__init__(weight)
        self.word = word
        self.position = position
        self.case_sensitive = case_sensitive

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        rubric: list[RubricItem] = []
        scores: list[float] = []

        expected_word = self.word if self.case_sensitive else self.word.upper()

        for i, letter in enumerate(expected_word):
            if i >= len(structure.lines):
                rubric.append(
                    RubricItem(
                        criterion=f"Letter {i + 1}",
                        expected=letter,
                        actual="(line missing)",
                        score=0.0,
                        passed=False,
                    )
                )
                scores.append(0.0)
                continue

            line = structure.lines[i]
            actual_letter = self._get_letter(line)

            if not self.case_sensitive:
                actual_letter = actual_letter.upper()

            passed = actual_letter == letter
            score = 1.0 if passed else 0.0

            rubric.append(
                RubricItem(
                    criterion=f"Letter {i + 1}",
                    expected=letter,
                    actual=actual_letter,
                    score=score,
                    passed=passed,
                    explanation=line[:30] + "..." if len(line) > 30 else line,
                )
            )
            scores.append(score)

        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_passed = all(r.passed for r in rubric)

        return VerificationResult(
            score=overall_score,
            passed=overall_passed,
            rubric=rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
        )

    def _get_letter(self, line: str) -> str:
        """Extract the relevant letter from a line."""
        # Get first alphabetic character
        line = line.strip()
        if not line:
            return ""

        if self.position == "first":
            for char in line:
                if char.isalpha():
                    return char
        elif self.position == "last":
            for char in reversed(line):
                if char.isalpha():
                    return char
        # middle not implemented yet

        return ""

    def describe(self) -> str:
        return f"First letters spell '{self.word}'"

    def instruction(self) -> str:
        """Plain English instruction for LLM prompts."""
        return f"Write an acrostic where the first letter of each line spells out '{self.word}'."
