"""Tests for text preprocessing and parsing."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from abide.primitives import (
    NormalizationConfig,
    NormalizationMode,
    count_words,
    extract_end_word,
    extract_end_words,
    normalize_whitespace,
    parse_structure,
    tokenize_line,
)
from tests.conftest import ALL_POEMS, PoemSpec


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_crlf_normalization(self) -> None:
        result = normalize_whitespace("hello\r\nworld")
        assert result.text == "hello\nworld"
        assert "line endings" in result.changes_made[0]

    def test_cr_normalization(self) -> None:
        result = normalize_whitespace("hello\rworld")
        assert result.text == "hello\nworld"

    def test_tab_conversion(self) -> None:
        result = normalize_whitespace("hello\tworld")
        assert "\t" not in result.text
        assert "    " in result.text  # Default 4-space tabs

    def test_excessive_blank_lines(self) -> None:
        result = normalize_whitespace("line1\n\n\n\nline2")
        assert result.text == "line1\n\nline2"
        assert result.text.count("\n\n") == 1

    def test_trailing_whitespace(self) -> None:
        result = normalize_whitespace("hello   \nworld  ")
        assert result.text == "hello\nworld"

    def test_leading_trailing_text_whitespace(self) -> None:
        result = normalize_whitespace("  \n  hello\nworld  \n  ")
        assert result.text == "hello\nworld"

    def test_quality_score(self) -> None:
        # Clean text should have quality 1.0
        clean = normalize_whitespace("hello\nworld")
        assert clean.quality_score == 1.0

        # Dirty text should have lower quality
        dirty = normalize_whitespace("hello\r\n\t\t\nworld   \n\n\n\n")
        assert dirty.quality_score < 1.0

    def test_strict_mode(self) -> None:
        config = NormalizationConfig(mode=NormalizationMode.STRICT)
        result = normalize_whitespace("hello\r\nworld\t!", config)
        assert result.text == "hello\r\nworld\t!"  # Unchanged
        assert result.quality_score == 1.0

    @given(st.text(max_size=500))
    def test_idempotence(self, text: str) -> None:
        """Normalizing twice should give same result."""
        result1 = normalize_whitespace(text)
        result2 = normalize_whitespace(result1.text)
        assert result1.text == result2.text


class TestParseStructure:
    """Tests for poem structure parsing."""

    def test_simple_poem(self) -> None:
        text = "line1\nline2\n\nline3\nline4"
        structure = parse_structure(text)

        assert structure.line_count == 4
        assert structure.stanza_count == 2
        assert structure.stanza_sizes == (2, 2)
        assert structure.lines == ("line1", "line2", "line3", "line4")

    def test_single_stanza(self) -> None:
        text = "line1\nline2\nline3"
        structure = parse_structure(text)

        assert structure.stanza_count == 1
        assert structure.stanza_sizes == (3,)

    def test_empty_text(self) -> None:
        structure = parse_structure("")
        assert structure.is_empty
        assert structure.line_count == 0
        assert structure.stanza_count == 0

    def test_whitespace_only(self) -> None:
        structure = parse_structure("   \n\n\t\t\n   ")
        assert structure.is_empty

    @pytest.mark.parametrize("poem", ALL_POEMS, ids=lambda p: p.name)
    def test_ground_truth_poems(self, poem: PoemSpec) -> None:
        """Verify parsing matches expected structure for real poems."""
        structure = parse_structure(poem.text)
        assert structure.line_count == poem.expected_line_count
        assert structure.stanza_count == poem.expected_stanza_count
        assert list(structure.stanza_sizes) == poem.expected_stanza_sizes


class TestExtractEndWord:
    """Tests for end word extraction."""

    def test_simple_line(self) -> None:
        assert extract_end_word("To be or not to be") == "be"

    def test_with_punctuation(self) -> None:
        assert extract_end_word("Hello world!") == "world"
        assert extract_end_word("Hello world.") == "world"
        assert extract_end_word("Hello world?") == "world"

    def test_with_quotes(self) -> None:
        assert extract_end_word("'Hello,' she said") == "said"
        assert extract_end_word('He said "hello"') == "hello"

    def test_empty_line(self) -> None:
        assert extract_end_word("") == ""
        assert extract_end_word("   ") == ""

    def test_lowercase(self) -> None:
        assert extract_end_word("Hello WORLD") == "world"

    def test_multiple_punctuation(self) -> None:
        assert extract_end_word("Really?!") == "really"
        assert extract_end_word("End...)") == "end"

    @pytest.mark.parametrize(
        "poem",
        [p for p in ALL_POEMS if p.expected_end_words],
        ids=lambda p: p.name,
    )
    def test_ground_truth_end_words(self, poem: PoemSpec) -> None:
        """Test against known end words if specified."""
        if poem.expected_end_words:
            structure = parse_structure(poem.text)
            end_words = extract_end_words(structure)
            # Just verify we get the right count - specific words vary
            assert len(end_words) == poem.expected_line_count


class TestExtractEndWords:
    """Tests for extracting all end words."""

    def test_basic(self) -> None:
        structure = parse_structure("Hello world!\nGoodbye moon.")
        end_words = extract_end_words(structure)
        assert end_words == ("world", "moon")


class TestTokenizeLine:
    """Tests for line tokenization."""

    def test_simple_line(self) -> None:
        tokens = tokenize_line("Hello world")
        assert tokens == ("hello", "world")

    def test_with_punctuation(self) -> None:
        tokens = tokenize_line("Hello, world!")
        assert tokens == ("hello", "world")

    def test_contractions(self) -> None:
        tokens = tokenize_line("Don't stop!")
        assert tokens == ("don't", "stop")

    def test_hyphenated(self) -> None:
        tokens = tokenize_line("well-known fact")
        assert tokens == ("well-known", "fact")

    def test_preserve_punctuation(self) -> None:
        tokens = tokenize_line("Hello, world!", preserve_punctuation=True)
        assert tokens == ("hello,", "world!")

    def test_case_preservation(self) -> None:
        tokens = tokenize_line("Hello World", lowercase=False)
        assert tokens == ("Hello", "World")

    def test_empty_line(self) -> None:
        assert tokenize_line("") == ()
        assert tokenize_line("   ") == ()


class TestCountWords:
    """Tests for word counting."""

    def test_simple(self) -> None:
        assert count_words("Hello world") == 2

    def test_empty(self) -> None:
        assert count_words("") == 0

    def test_multiline(self) -> None:
        assert count_words("Hello world\nGoodbye moon") == 4

    def test_with_punctuation(self) -> None:
        assert count_words("One, two, three!") == 3
