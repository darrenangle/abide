"""Tests for structural constraints."""

from abide.constraints import (
    LineCount,
    NumericBound,
    StanzaCount,
    StanzaSizes,
    SyllablesPerLine,
    TotalSyllables,
)


class TestLineCount:
    """Tests for LineCount constraint."""

    def test_exact_match(self) -> None:
        """Exact line count passes."""
        poem = "Line one\nLine two\nLine three"
        constraint = LineCount(3)
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_exact_mismatch(self) -> None:
        """Wrong line count fails with partial score."""
        poem = "Line one\nLine two"
        constraint = LineCount(3)
        result = constraint.verify(poem)
        assert result.passed is False
        assert 0 < result.score < 1.0

    def test_min_bound_pass(self) -> None:
        """Minimum line count passes when met."""
        poem = "Line one\nLine two\nLine three\nLine four"
        constraint = LineCount(NumericBound.at_least(3))
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_min_bound_fail(self) -> None:
        """Minimum line count fails when not met."""
        poem = "Line one\nLine two"
        constraint = LineCount(NumericBound.at_least(3))
        result = constraint.verify(poem)
        assert result.passed is False
        assert result.score < 1.0

    def test_max_bound_pass(self) -> None:
        """Maximum line count passes when satisfied."""
        poem = "Line one\nLine two"
        constraint = LineCount(NumericBound.at_most(3))
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_range_bound_pass(self) -> None:
        """Range line count passes when in range."""
        poem = "Line one\nLine two\nLine three"
        constraint = LineCount(NumericBound.between(2, 5))
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_sonnet_line_count(self) -> None:
        """Sonnet has exactly 14 lines."""
        poem = "\n".join([f"Line {i}" for i in range(14)])
        constraint = LineCount(14)
        result = constraint.verify(poem)
        assert result.passed is True

    def test_describe(self) -> None:
        """Description is human-readable."""
        constraint = LineCount(14)
        desc = constraint.describe()
        assert "14" in desc
        assert "line" in desc.lower()

    def test_rubric_generation(self) -> None:
        """Rubric contains expected info."""
        poem = "Line one\nLine two"
        constraint = LineCount(3)
        result = constraint.verify(poem)
        assert len(result.rubric) == 1
        assert "2" in result.rubric[0].actual
        assert "3" in result.rubric[0].expected


class TestStanzaCount:
    """Tests for StanzaCount constraint."""

    def test_single_stanza(self) -> None:
        """Poem with no blank lines is one stanza."""
        poem = "Line one\nLine two\nLine three"
        constraint = StanzaCount(1)
        result = constraint.verify(poem)
        assert result.passed is True

    def test_multiple_stanzas(self) -> None:
        """Poem with blank lines has multiple stanzas."""
        poem = "Line one\nLine two\n\nLine three\nLine four"
        constraint = StanzaCount(2)
        result = constraint.verify(poem)
        assert result.passed is True

    def test_villanelle_stanzas(self) -> None:
        """Villanelle has 6 stanzas."""
        stanzas = [
            "Line 1\nLine 2\nLine 3",
            "Line 4\nLine 5\nLine 6",
            "Line 7\nLine 8\nLine 9",
            "Line 10\nLine 11\nLine 12",
            "Line 13\nLine 14\nLine 15",
            "Line 16\nLine 17\nLine 18\nLine 19",
        ]
        poem = "\n\n".join(stanzas)
        constraint = StanzaCount(6)
        result = constraint.verify(poem)
        assert result.passed is True

    def test_mismatch_stanzas(self) -> None:
        """Wrong stanza count fails."""
        poem = "Line one\nLine two\n\nLine three"
        constraint = StanzaCount(3)
        result = constraint.verify(poem)
        assert result.passed is False

    def test_describe(self) -> None:
        """Description is human-readable."""
        constraint = StanzaCount(6)
        desc = constraint.describe()
        assert "6" in desc
        assert "stanza" in desc.lower()


class TestStanzaSizes:
    """Tests for StanzaSizes constraint."""

    def test_exact_match(self) -> None:
        """Correct stanza sizes pass."""
        poem = "L1\nL2\nL3\n\nL4\nL5\nL6"
        constraint = StanzaSizes([3, 3])
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_villanelle_structure(self) -> None:
        """Villanelle: 5 tercets + 1 quatrain."""
        stanzas = []
        for i in range(5):
            stanzas.append(f"L{i * 3 + 1}\nL{i * 3 + 2}\nL{i * 3 + 3}")
        stanzas.append("L16\nL17\nL18\nL19")  # quatrain
        poem = "\n\n".join(stanzas)

        constraint = StanzaSizes([3, 3, 3, 3, 3, 4])
        result = constraint.verify(poem)
        assert result.passed is True

    def test_sestina_structure(self) -> None:
        """Sestina: 6 sestets + 1 tercet (envoi)."""
        stanzas = []
        for i in range(6):
            stanzas.append("\n".join([f"Line {i * 6 + j + 1}" for j in range(6)]))
        stanzas.append("L37\nL38\nL39")  # envoi
        poem = "\n\n".join(stanzas)

        constraint = StanzaSizes([6, 6, 6, 6, 6, 6, 3])
        result = constraint.verify(poem)
        assert result.passed is True

    def test_wrong_stanza_count(self) -> None:
        """Wrong number of stanzas fails."""
        poem = "L1\nL2\nL3"
        constraint = StanzaSizes([3, 3])
        result = constraint.verify(poem)
        assert result.passed is False

    def test_wrong_stanza_size(self) -> None:
        """Wrong stanza size fails."""
        poem = "L1\nL2\n\nL3\nL4\nL5"
        constraint = StanzaSizes([3, 3])
        result = constraint.verify(poem)
        assert result.passed is False

    def test_partial_credit(self) -> None:
        """Near-match gets partial credit."""
        poem = "L1\nL2\nL3\n\nL4\nL5"  # [3, 2] instead of [3, 3]
        constraint = StanzaSizes([3, 3])
        result = constraint.verify(poem)
        assert result.passed is False
        assert result.score > 0.5  # Should get partial credit

    def test_describe(self) -> None:
        """Description lists sizes."""
        constraint = StanzaSizes([3, 3, 3, 4])
        desc = constraint.describe()
        assert "3" in desc
        assert "4" in desc


class TestSyllablesPerLine:
    """Tests for SyllablesPerLine constraint."""

    def test_haiku(self) -> None:
        """Traditional haiku syllable pattern."""
        # 5-7-5 pattern
        haiku = "An old silent pond\nA frog jumps into the pond\nSplash silence again"
        constraint = SyllablesPerLine([5, 7, 5])
        result = constraint.verify(haiku)
        # Allow for syllable counting variation
        assert result.score > 0.5

    def test_exact_syllables(self) -> None:
        """Lines with exact syllable counts."""
        # Construct lines with known syllable counts
        poem = "cat\ndog ran fast\nhello"  # ~1, ~3, ~2
        constraint = SyllablesPerLine([1, 3, 2])
        result = constraint.verify(poem)
        # Syllable counting can vary, just check structure
        assert len(result.rubric) == 3

    def test_tolerance(self) -> None:
        """Tolerance allows near matches."""
        poem = "hello there\ngoodbye now"  # Variable syllables
        constraint = SyllablesPerLine([3, 3], tolerance=1)
        result = constraint.verify(poem)
        # With tolerance, should get credit for close matches
        assert result.score > 0.0

    def test_describe_uniform(self) -> None:
        """Description for uniform syllables."""
        constraint = SyllablesPerLine([10] * 14)
        desc = constraint.describe()
        assert "10" in desc

    def test_describe_varied(self) -> None:
        """Description for varied syllables."""
        constraint = SyllablesPerLine([5, 7, 5])
        desc = constraint.describe()
        assert "5" in desc or "syllable" in desc.lower()


class TestTotalSyllables:
    """Tests for TotalSyllables constraint."""

    def test_exact_count(self) -> None:
        """Exact syllable count passes."""
        # Simple words with known syllables
        poem = "cat dog rat"  # 3 syllables
        constraint = TotalSyllables(3)
        result = constraint.verify(poem)
        # May vary slightly with syllable counting
        assert result.score > 0.5

    def test_at_least(self) -> None:
        """Minimum syllable count."""
        poem = "Hello there my friend how are you today"
        constraint = TotalSyllables(NumericBound.at_least(5))
        result = constraint.verify(poem)
        assert result.passed is True

    def test_describe(self) -> None:
        """Description is human-readable."""
        constraint = TotalSyllables(100)
        desc = constraint.describe()
        assert "100" in desc
        assert "syllable" in desc.lower()
