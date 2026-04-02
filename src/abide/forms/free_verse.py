"""
Free verse form template.

These verifiers check configurable structural bounds only. They do not attempt
to verify broader poetic qualities such as imagery, rhythm, or figurative
language.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    Constraint,
    ConstraintType,
    VerificationResult,
)
from abide.forms._validation import (
    require_nonnegative,
    require_ordered_bounds,
    require_positive,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class FreeVerse(Constraint):
    """
    Free verse with configurable structural bounds.

    This verifier checks only line/stanza counts and optional word-count bounds
    per line. It does not attempt to verify rhyme, meter, or other stylistic
    properties.

    Examples:
        >>> free = FreeVerse(min_lines=5, min_stanzas=2)
        >>> result = free.verify(poem)
    """

    name = "Free Verse"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        min_lines: int = 3,
        max_lines: int | None = None,
        min_stanzas: int = 1,
        max_stanzas: int | None = None,
        min_words_per_line: int = 1,
        max_words_per_line: int | None = None,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize free verse constraint.

        Args:
            min_lines: Minimum number of lines
            max_lines: Maximum number of lines (None for unlimited)
            min_stanzas: Minimum number of stanzas
            max_stanzas: Maximum number of stanzas (None for unlimited)
            min_words_per_line: Minimum words per line (for non-empty lines)
            max_words_per_line: Maximum words per line (None for unlimited)
            weight: Relative weight for composition
        """
        super().__init__(weight)
        require_positive(min_lines, "min_lines")
        require_positive(min_stanzas, "min_stanzas")
        require_nonnegative(min_words_per_line, "min_words_per_line")
        require_ordered_bounds("max_lines", max_lines, "min_lines", min_lines)
        require_ordered_bounds("max_stanzas", max_stanzas, "min_stanzas", min_stanzas)
        if max_words_per_line is not None:
            require_positive(max_words_per_line, "max_words_per_line")
            if max_words_per_line < min_words_per_line:
                raise ValueError(
                    "max_words_per_line must be greater than or equal to min_words_per_line"
                )
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.min_stanzas = min_stanzas
        self.max_stanzas = max_stanzas
        self.min_words_per_line = min_words_per_line
        self.max_words_per_line = max_words_per_line

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        score = 1.0
        issues = []
        word_bound_violations: list[str] = []

        # Check line count
        if structure.line_count < self.min_lines:
            score *= structure.line_count / self.min_lines
            issues.append(f"Too few lines ({structure.line_count} < {self.min_lines})")

        if self.max_lines and structure.line_count > self.max_lines:
            score *= self.max_lines / structure.line_count
            issues.append(f"Too many lines ({structure.line_count} > {self.max_lines})")

        # Check stanza count
        if structure.stanza_count < self.min_stanzas:
            score *= structure.stanza_count / self.min_stanzas
            issues.append(f"Too few stanzas ({structure.stanza_count} < {self.min_stanzas})")

        if self.max_stanzas and structure.stanza_count > self.max_stanzas:
            score *= self.max_stanzas / structure.stanza_count
            issues.append(f"Too many stanzas ({structure.stanza_count} > {self.max_stanzas})")

        # Check word counts per line
        if self.min_words_per_line or self.max_words_per_line:
            for i, line in enumerate(structure.lines):
                word_count = len(line.split())
                if word_count < self.min_words_per_line:
                    score *= 0.95  # Minor penalty
                    word_bound_violations.append(
                        f"Line {i + 1} has too few words ({word_count} < {self.min_words_per_line})"
                    )
                if self.max_words_per_line and word_count > self.max_words_per_line:
                    score *= 0.95
                    word_bound_violations.append(
                        f"Line {i + 1} has too many words ({word_count} > {self.max_words_per_line})"
                    )

        issues.extend(word_bound_violations)
        passed = score >= 0.8 and not issues

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "line_count": structure.line_count,
                "stanza_count": structure.stanza_count,
                "word_bound_violations": word_bound_violations,
                "issues": issues,
            },
        )

    def describe(self) -> str:
        parts = [f"Free Verse: {self.min_lines}+ lines"]
        if self.min_stanzas > 1:
            parts.append(f"{self.min_stanzas}+ stanzas")
        return ", ".join(parts)


class ProsePoem(Constraint):
    """
    Prose poem in paragraph form without line-broken verse.

    This verifier checks only paragraph count, approximate sentence count, and
    whether each paragraph is continuous prose rather than a stack of short
    verse lines.

    Examples:
        >>> prose = ProsePoem(min_paragraphs=2, max_paragraphs=5)
        >>> result = prose.verify(poem)
    """

    name = "Prose Poem"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        min_paragraphs: int = 1,
        max_paragraphs: int = 10,
        min_sentences: int = 3,
        max_sentences: int | None = None,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize prose poem constraint.

        Args:
            min_paragraphs: Minimum number of paragraphs
            max_paragraphs: Maximum number of paragraphs
            min_sentences: Minimum number of sentences total
            max_sentences: Maximum number of sentences (None for unlimited)
            weight: Relative weight for composition
        """
        super().__init__(weight)
        require_positive(min_paragraphs, "min_paragraphs")
        require_positive(max_paragraphs, "max_paragraphs")
        require_positive(min_sentences, "min_sentences")
        require_ordered_bounds(
            "max_paragraphs",
            max_paragraphs,
            "min_paragraphs",
            min_paragraphs,
        )
        require_ordered_bounds(
            "max_sentences",
            max_sentences,
            "min_sentences",
            min_sentences,
        )
        self.min_paragraphs = min_paragraphs
        self.max_paragraphs = max_paragraphs
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        # Count paragraphs (separated by double newlines)
        text = poem if isinstance(poem, str) else "\n".join(poem.lines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        para_count = len(paragraphs)
        line_broken_paragraphs = sum(
            1
            for paragraph in paragraphs
            if len([line for line in paragraph.splitlines() if line.strip()]) > 1
        )

        # Count sentences (rough approximation)
        sentence_count = sum(text.count(c) for c in ".!?")

        score = 1.0
        issues = []

        if para_count < self.min_paragraphs:
            score *= para_count / self.min_paragraphs
            issues.append(f"Too few paragraphs ({para_count} < {self.min_paragraphs})")

        if para_count > self.max_paragraphs:
            score *= self.max_paragraphs / para_count
            issues.append(f"Too many paragraphs ({para_count} > {self.max_paragraphs})")

        if sentence_count < self.min_sentences:
            score *= sentence_count / self.min_sentences
            issues.append(f"Too few sentences ({sentence_count} < {self.min_sentences})")

        if self.max_sentences and sentence_count > self.max_sentences:
            score *= self.max_sentences / sentence_count
            issues.append(f"Too many sentences ({sentence_count} > {self.max_sentences})")

        if line_broken_paragraphs:
            score *= 0.05
            issues.append(f"Contains line-broken verse in {line_broken_paragraphs} paragraph(s)")

        passed = score >= 0.8 and not issues

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "paragraph_count": para_count,
                "sentence_count": sentence_count,
                "line_broken_paragraphs": line_broken_paragraphs,
                "issues": issues,
            },
        )

    def describe(self) -> str:
        return f"Prose Poem: {self.min_paragraphs}-{self.max_paragraphs} paragraphs"
