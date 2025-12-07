"""
Text preprocessing and parsing for poetry analysis.

Provides robust normalization, line/stanza extraction, and tokenization
that handles the variety of whitespace and formatting found in poems.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class NormalizationMode(Enum):
    """Whitespace normalization strictness levels."""

    STRICT = "strict"  # Preserve original whitespace exactly
    LENIENT = "lenient"  # Normalize to standard format
    FUZZY = "fuzzy"  # Normalize + track quality penalty


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for text normalization."""

    mode: NormalizationMode = NormalizationMode.LENIENT
    tab_width: int = 4
    max_consecutive_blanks: int = 1
    strip_trailing: bool = True


@dataclass
class NormalizationResult:
    """Result of text normalization with quality metrics."""

    text: str
    quality_score: float  # 1.0 = no normalization needed, lower = more cleanup
    original_length: int
    changes_made: list[str]


def normalize_whitespace(
    text: str,
    config: NormalizationConfig | None = None,
) -> NormalizationResult:
    """
    Normalize whitespace in poem text.

    Handles:
    - CRLF/LF/CR line endings
    - Tabs to spaces
    - Multiple consecutive blank lines
    - Trailing whitespace

    Args:
        text: Raw poem text
        config: Normalization configuration

    Returns:
        NormalizationResult with normalized text and quality metrics

    Examples:
        >>> result = normalize_whitespace("hello\\r\\nworld")
        >>> result.text
        'hello\\nworld'
        >>> result = normalize_whitespace("line1\\n\\n\\n\\nline2")
        >>> result.text.count("\\n\\n")
        1
    """
    if config is None:
        config = NormalizationConfig()

    original_length = len(text)
    changes: list[str] = []
    result = text
    penalties = 0.0

    if config.mode == NormalizationMode.STRICT:
        return NormalizationResult(
            text=text,
            quality_score=1.0,
            original_length=original_length,
            changes_made=[],
        )

    # 1. Normalize line endings (CRLF -> LF, CR -> LF)
    crlf_count = result.count("\r\n")
    cr_count = result.count("\r") - crlf_count  # Standalone CRs
    if crlf_count > 0 or cr_count > 0:
        result = result.replace("\r\n", "\n").replace("\r", "\n")
        changes.append(f"normalized {crlf_count + cr_count} line endings")
        penalties += 0.01 * (crlf_count + cr_count)

    # 2. Convert tabs to spaces
    tab_count = result.count("\t")
    if tab_count > 0:
        result = result.replace("\t", " " * config.tab_width)
        changes.append(f"converted {tab_count} tabs")
        penalties += 0.005 * tab_count

    # 3. Normalize consecutive blank lines
    blank_pattern = r"\n{3,}"
    excessive_blanks = re.findall(blank_pattern, result)
    if excessive_blanks:
        max_newlines = config.max_consecutive_blanks + 1
        replacement = "\n" * max_newlines
        result = re.sub(blank_pattern, replacement, result)
        changes.append(f"normalized {len(excessive_blanks)} excessive blank sections")
        penalties += 0.02 * len(excessive_blanks)

    # 4. Strip trailing whitespace from lines
    if config.strip_trailing:
        lines = result.split("\n")
        stripped_count = 0
        new_lines = []
        for line in lines:
            stripped = line.rstrip()
            if len(stripped) < len(line):
                stripped_count += 1
            new_lines.append(stripped)
        if stripped_count > 0:
            result = "\n".join(new_lines)
            changes.append(f"stripped trailing whitespace from {stripped_count} lines")
            penalties += 0.001 * stripped_count

    # 5. Strip leading/trailing whitespace from whole text
    stripped_result = result.strip()
    if len(stripped_result) < len(result):
        changes.append("stripped leading/trailing whitespace from text")
        penalties += 0.01
    result = stripped_result

    quality_score = max(0.0, 1.0 - penalties)

    return NormalizationResult(
        text=result,
        quality_score=quality_score,
        original_length=original_length,
        changes_made=changes,
    )


@dataclass(frozen=True)
class PoemStructure:
    """Parsed structure of a poem."""

    lines: tuple[str, ...]
    stanzas: tuple[tuple[str, ...], ...]
    line_count: int
    stanza_count: int
    stanza_sizes: tuple[int, ...]

    @property
    def is_empty(self) -> bool:
        """Check if poem has no content."""
        return self.line_count == 0


def parse_structure(text: str, normalize: bool = True) -> PoemStructure:
    """
    Parse poem text into lines and stanzas.

    Stanza boundaries are detected by blank lines.

    Args:
        text: Poem text (optionally pre-normalized)
        normalize: Whether to normalize whitespace first

    Returns:
        PoemStructure with parsed lines and stanzas

    Examples:
        >>> structure = parse_structure("line1\\nline2\\n\\nline3\\nline4")
        >>> structure.line_count
        4
        >>> structure.stanza_count
        2
        >>> structure.stanza_sizes
        (2, 2)
    """
    if normalize:
        result = normalize_whitespace(text)
        text = result.text

    if not text.strip():
        return PoemStructure(
            lines=(),
            stanzas=(),
            line_count=0,
            stanza_count=0,
            stanza_sizes=(),
        )

    # Split into stanzas on blank lines
    raw_stanzas = re.split(r"\n\s*\n", text)

    stanzas: list[tuple[str, ...]] = []
    all_lines: list[str] = []

    for raw_stanza in raw_stanzas:
        raw_stanza = raw_stanza.strip()
        if not raw_stanza:
            continue

        stanza_lines = tuple(
            line for line in raw_stanza.split("\n") if line.strip()
        )
        if stanza_lines:
            stanzas.append(stanza_lines)
            all_lines.extend(stanza_lines)

    return PoemStructure(
        lines=tuple(all_lines),
        stanzas=tuple(stanzas),
        line_count=len(all_lines),
        stanza_count=len(stanzas),
        stanza_sizes=tuple(len(s) for s in stanzas),
    )


def extract_end_word(line: str) -> str:
    """
    Extract the final word from a line for rhyme/pattern analysis.

    Strips trailing punctuation and normalizes to lowercase.

    Args:
        line: A single line of text

    Returns:
        The final word, lowercase, without trailing punctuation

    Examples:
        >>> extract_end_word("To be or not to be!")
        'be'
        >>> extract_end_word("The end.")
        'end'
        >>> extract_end_word("'Hello,' she said")
        'said'
        >>> extract_end_word("")
        ''
    """
    line = line.strip()
    if not line:
        return ""

    # Remove trailing punctuation (preserving internal)
    # Handle quotes, parentheses, and standard punctuation
    while line and line[-1] in '.,;:!?\'")-]}>':
        line = line[:-1]

    line = line.strip()
    if not line:
        return ""

    # Split on whitespace and get last token
    words = line.split()
    if not words:
        return ""

    word = words[-1].lower()

    # Remove any leading punctuation from the word
    while word and word[0] in '\'"([{<':
        word = word[1:]

    return word


def extract_end_words(structure: PoemStructure) -> tuple[str, ...]:
    """
    Extract end words from all lines in a poem.

    Args:
        structure: Parsed poem structure

    Returns:
        Tuple of end words, one per line

    Examples:
        >>> s = parse_structure("Hello world!\\nGoodbye moon.")
        >>> extract_end_words(s)
        ('world', 'moon')
    """
    return tuple(extract_end_word(line) for line in structure.lines)


def tokenize_line(
    line: str,
    preserve_punctuation: bool = False,
    lowercase: bool = True,
) -> tuple[str, ...]:
    """
    Tokenize a line into words.

    Handles contractions, hyphenated words, and punctuation.

    Args:
        line: Line of text
        preserve_punctuation: Keep punctuation attached to words
        lowercase: Convert to lowercase

    Returns:
        Tuple of word tokens

    Examples:
        >>> tokenize_line("Don't stop!")
        ("don't", 'stop')
        >>> tokenize_line("well-known fact")
        ('well-known', 'fact')
        >>> tokenize_line("Hello, World!", preserve_punctuation=True)
        ('hello,', 'world!')
    """
    if not line.strip():
        return ()

    if lowercase:
        line = line.lower()

    if preserve_punctuation:
        # Simple split on whitespace
        return tuple(line.split())

    # More sophisticated tokenization:
    # - Keep contractions (don't, it's)
    # - Keep hyphenated words
    # - Remove standalone punctuation

    # Pattern: word characters, optionally with internal apostrophes/hyphens
    pattern = r"[a-zA-Z]+(?:[''-][a-zA-Z]+)*"
    tokens = re.findall(pattern, line, re.UNICODE)

    return tuple(tokens)


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Text to count words in

    Returns:
        Number of words

    Examples:
        >>> count_words("Hello world")
        2
        >>> count_words("")
        0
        >>> count_words("One, two, three!")
        3
    """
    structure = parse_structure(text)
    total = 0
    for line in structure.lines:
        total += len(tokenize_line(line))
    return total
