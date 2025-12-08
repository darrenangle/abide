"""
Free verse form template.

Free verse has no fixed meter or rhyme scheme, but still has structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    Constraint,
    ConstraintType,
    LineCount,
    LineLengthRange,
    MeasureMode,
    StanzaCount,
    VerificationResult,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class FreeVerse(Constraint):
    """
    Free verse: Poetry without fixed meter or rhyme scheme.

    While "free," poems still have structure:
    - Line breaks create rhythm and emphasis
    - Stanza breaks create logical divisions
    - Length and pacing matter

    This constraint verifies minimal structural requirements.

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
                if self.max_words_per_line and word_count > self.max_words_per_line:
                    score *= 0.95

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
    Prose poem: Poetry in prose form without line breaks.

    Combines poetic language with prose format:
    - No line breaks (continuous paragraphs)
    - Uses poetic devices (imagery, rhythm, sound)
    - Short paragraphs

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
        self.min_paragraphs = min_paragraphs
        self.max_paragraphs = max_paragraphs
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        # Count paragraphs (separated by double newlines)
        text = poem if isinstance(poem, str) else "\n".join(poem.lines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        para_count = len(paragraphs)

        # Count sentences (rough approximation)
        sentence_count = sum(
            text.count(c) for c in ".!?"
        )

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
                "issues": issues,
            },
        )

    def describe(self) -> str:
        return f"Prose Poem: {self.min_paragraphs}-{self.max_paragraphs} paragraphs"
