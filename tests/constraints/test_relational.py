"""Tests for relational constraints."""

import pytest

from abide.constraints import (
    Acrostic,
    EndWordPattern,
    Refrain,
    RhymeScheme,
)


class TestRhymeScheme:
    """Tests for RhymeScheme constraint."""

    def test_aabb_rhyme(self) -> None:
        """AABB rhyme scheme (couplets)."""
        poem = "The cat sat on the mat\nI really like my hat\nThe dog ran down the lane\nIt started to rain"
        constraint = RhymeScheme("AABB")
        result = constraint.verify(poem)
        # mat/hat rhyme, lane/rain rhyme
        assert result.score > 0.5

    def test_abab_rhyme(self) -> None:
        """ABAB rhyme scheme (alternate)."""
        poem = "Roses are red\nViolets are blue\nSugar is sweet\nAnd so are you"
        constraint = RhymeScheme("ABAB")
        result = constraint.verify(poem)
        # red/sweet don't rhyme, blue/you do
        assert len(result.rubric) > 0

    def test_shakespearean_sonnet_scheme(self) -> None:
        """Shakespearean sonnet rhyme scheme."""
        constraint = RhymeScheme("ABABCDCDEFEFGG")
        desc = constraint.describe()
        assert "ABABCDCDEFEFGG" in desc

    def test_petrarchan_sonnet_scheme(self) -> None:
        """Petrarchan sonnet rhyme scheme."""
        # ABBAABBA CDECDE pattern
        constraint = RhymeScheme("ABBAABBACDECDE")
        assert constraint.scheme == "ABBAABBACDECDE"

    def test_limerick_scheme(self) -> None:
        """Limerick rhyme scheme AABBA."""
        limerick = """There once was a man from Nantucket
Who kept all his cash in a bucket
His daughter named Nan
Ran away with a man
And as for the bucket, Nantucket"""
        constraint = RhymeScheme("AABBA")
        result = constraint.verify(limerick)
        # Nantucket/bucket/Nantucket should rhyme (A)
        # Nan/man should rhyme (B)
        assert result.score > 0.3  # May vary with rhyme detection

    def test_scheme_normalizes_spaces(self) -> None:
        """Spaces and punctuation in scheme are ignored."""
        constraint = RhymeScheme("ABBA ABBA CDC DCD")
        assert constraint.scheme == "ABBAABBACDCDCD"

    def test_line_count_mismatch(self) -> None:
        """Wrong line count affects score."""
        poem = "Line one\nLine two\nLine three"
        constraint = RhymeScheme("AABB")  # Expects 4 lines
        result = constraint.verify(poem)
        assert result.passed is False
        # Should have rubric item about line count
        line_count_items = [r for r in result.rubric if "line count" in r.criterion.lower()]
        assert len(line_count_items) > 0

    def test_identical_words_partial_credit(self) -> None:
        """Identical words get partial credit by default."""
        poem = "The cat\nThe cat\nThe dog\nThe dog"
        constraint = RhymeScheme("AABB", allow_identical=False)
        result = constraint.verify(poem)
        # cat/cat and dog/dog are identical, not true rhymes
        assert 0 < result.score < 1.0

    def test_identical_words_allowed(self) -> None:
        """Identical words pass with allow_identical=True."""
        poem = "The cat\nThe cat"
        constraint = RhymeScheme("AA", allow_identical=True)
        result = constraint.verify(poem)
        assert result.score == 1.0

    def test_threshold_affects_passing(self) -> None:
        """Threshold controls what counts as rhyming."""
        poem = "The cat\nThe bat"  # Should rhyme well
        high_threshold = RhymeScheme("AA", threshold=0.95)
        low_threshold = RhymeScheme("AA", threshold=0.3)

        high_result = high_threshold.verify(poem)
        low_result = low_threshold.verify(poem)

        # Low threshold should be easier to pass
        assert low_result.score >= high_result.score

    def test_describe(self) -> None:
        """Description includes scheme."""
        constraint = RhymeScheme("ABAB")
        desc = constraint.describe()
        assert "ABAB" in desc
        assert "rhyme" in desc.lower()


class TestRefrain:
    """Tests for Refrain constraint."""

    def test_exact_refrain(self) -> None:
        """Exact line repetition passes."""
        poem = "Do not go gentle into that good night\nLine two\nLine three\nDo not go gentle into that good night"
        constraint = Refrain(reference_line=0, repeat_at=[3])
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_near_refrain(self) -> None:
        """Near-match gets partial credit."""
        poem = "Do not go gentle into that good night\nLine two\nLine three\nDo not go gently into that good night"
        constraint = Refrain(reference_line=0, repeat_at=[3], threshold=0.8)
        result = constraint.verify(poem)
        # "gentle" vs "gently" - high similarity
        assert result.score > 0.8

    def test_villanelle_refrains(self) -> None:
        """Villanelle has two alternating refrains."""
        # Line 0 repeats at 5, 11, 17
        # Line 2 repeats at 8, 14, 18
        lines = [
            "Do not go gentle into that good night",  # 0 - refrain A
            "Old age should burn and rave at close of day",  # 1
            "Rage, rage against the dying of the light",  # 2 - refrain B
            "Line 4",
            "Line 5",
            "Do not go gentle into that good night",  # 5 - refrain A
            "Line 7",
            "Line 8",
            "Rage, rage against the dying of the light",  # 8 - refrain B
        ]
        poem = "\n".join(lines)

        constraint_a = Refrain(reference_line=0, repeat_at=[5])
        constraint_b = Refrain(reference_line=2, repeat_at=[8])

        result_a = constraint_a.verify(poem)
        result_b = constraint_b.verify(poem)

        assert result_a.passed is True
        assert result_b.passed is True

    def test_missing_line(self) -> None:
        """Missing line in poem fails."""
        poem = "Line one\nLine two"
        constraint = Refrain(reference_line=0, repeat_at=[5])
        result = constraint.verify(poem)
        assert result.passed is False
        assert result.score == 0.0

    def test_reference_line_missing(self) -> None:
        """Reference line beyond poem fails."""
        poem = "Line one\nLine two"
        constraint = Refrain(reference_line=10, repeat_at=[15])
        result = constraint.verify(poem)
        assert result.passed is False
        assert result.score == 0.0

    def test_multiple_repeat_positions(self) -> None:
        """Multiple repeat positions checked."""
        lines = ["Refrain line"] * 5
        for i in [1, 2, 3]:
            lines[i] = f"Different line {i}"
        poem = "\n".join(lines)

        constraint = Refrain(reference_line=0, repeat_at=[4])
        result = constraint.verify(poem)
        assert result.passed is True

    def test_describe(self) -> None:
        """Description includes positions."""
        constraint = Refrain(reference_line=0, repeat_at=[5, 11, 17])
        desc = constraint.describe()
        assert "1" in desc  # 1-indexed reference
        assert "6" in desc or "12" in desc or "18" in desc


class TestEndWordPattern:
    """Tests for EndWordPattern constraint."""

    def test_sestina_default_rotation(self) -> None:
        """Default rotation is sestina pattern."""
        constraint = EndWordPattern()
        assert constraint.num_words == 6
        assert constraint.num_stanzas == 6
        assert constraint.rotation == (5, 0, 4, 1, 3, 2)

    def test_custom_rotation(self) -> None:
        """Custom rotation can be specified."""
        constraint = EndWordPattern(num_words=3, num_stanzas=3, rotation=[2, 0, 1])
        assert constraint.rotation == (2, 0, 1)

    def test_simple_pattern(self) -> None:
        """Simple end word pattern verification."""
        # Stanza 1: words A B C
        # Stanza 2: words C A B (rotation [2,0,1])
        poem = """First line ends with apple
Second line ends with banana
Third line ends with cherry

Now this line ends cherry
And this ends with apple
Finally comes banana"""
        constraint = EndWordPattern(num_words=3, num_stanzas=2, rotation=[2, 0, 1])
        result = constraint.verify(poem)
        # Should verify the pattern
        assert len(result.rubric) > 0

    def test_not_enough_stanzas(self) -> None:
        """Poem with too few stanzas fails."""
        poem = "Line 1\nLine 2\nLine 3"
        constraint = EndWordPattern(num_stanzas=6)
        result = constraint.verify(poem)
        assert result.passed is False
        assert result.score == 0.0

    def test_describe(self) -> None:
        """Description includes pattern info."""
        constraint = EndWordPattern()
        desc = constraint.describe()
        assert "6" in desc  # 6 words
        assert "rotation" in desc.lower() or "pattern" in desc.lower()


class TestAcrostic:
    """Tests for Acrostic constraint."""

    def test_simple_acrostic(self) -> None:
        """Simple acrostic spells word."""
        poem = """Love is patient and kind
Over all the earth it shines
Vast and wonderful and true
Every day it starts anew"""
        constraint = Acrostic("LOVE")
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_acrostic_case_insensitive(self) -> None:
        """Acrostic is case insensitive by default."""
        poem = "love starts here\nover the hill\nvalley below\neternally"
        constraint = Acrostic("LOVE", case_sensitive=False)
        result = constraint.verify(poem)
        assert result.passed is True

    def test_acrostic_case_sensitive(self) -> None:
        """Case sensitive acrostic."""
        poem = "love starts here\nOver the hill\nValley below\nEternally"
        constraint = Acrostic("lOVE", case_sensitive=True)
        result = constraint.verify(poem)
        assert result.passed is True

    def test_acrostic_wrong_letter(self) -> None:
        """Wrong letter fails."""
        poem = "Xove starts here\nOver the hill\nValley below\nEternally"
        constraint = Acrostic("LOVE")
        result = constraint.verify(poem)
        assert result.passed is False
        assert result.score == 0.75  # 3 out of 4 correct

    def test_acrostic_short_poem(self) -> None:
        """Poem shorter than word fails."""
        poem = "Line one\nLine two"
        constraint = Acrostic("LOVE")
        result = constraint.verify(poem)
        assert result.passed is False
        # L matches first line "Line...", O doesn't match "Line..."
        # Then lines 3-4 are missing (score 0 each)
        # So we get partial credit (1 out of 4 = 0.25)
        assert result.score < 0.5

    def test_acrostic_skips_punctuation(self) -> None:
        """Acrostic skips leading punctuation."""
        poem = '"Love is here\n"Over there\n"Very nice\n"Eternal"'
        constraint = Acrostic("LOVE")
        result = constraint.verify(poem)
        # Should find L, O, V, E after the quotes
        assert result.score > 0.5

    def test_describe(self) -> None:
        """Description includes the word."""
        constraint = Acrostic("LOVE")
        desc = constraint.describe()
        assert "LOVE" in desc
        assert "first" in desc.lower() or "letter" in desc.lower()

    def test_rubric_shows_each_letter(self) -> None:
        """Rubric has item for each letter."""
        poem = "Love\nOver\nVery\nEternal"
        constraint = Acrostic("LOVE")
        result = constraint.verify(poem)
        assert len(result.rubric) == 4
        for i, item in enumerate(result.rubric):
            assert f"Letter {i + 1}" in item.criterion
