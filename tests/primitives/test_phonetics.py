"""Tests for phonetic encoding and analysis."""

import pytest

from abide.primitives import (
    count_line_syllables,
    count_syllables,
    get_phonemes,
    get_rhyme_part,
    get_stress_pattern,
    metaphone,
    phonetic_similarity,
    rhyme_score,
    soundex,
    strip_stress,
    words_rhyme,
)
from tests.conftest import HAIKU_POEMS, PoemSpec


class TestSoundex:
    """Tests for Soundex encoding."""

    def test_basic_encoding(self) -> None:
        assert soundex("Robert") == "R163"
        assert soundex("Rupert") == "R163"

    def test_empty_string(self) -> None:
        assert soundex("") == "0000"

    def test_same_sound_different_spelling(self) -> None:
        # These should have similar (not necessarily identical) codes
        assert soundex("Smith")[0] == soundex("Smyth")[0]


class TestMetaphone:
    """Tests for Metaphone encoding."""

    def test_basic_encoding(self) -> None:
        # Night and knight should have same metaphone
        assert metaphone("night") == metaphone("knight")

    def test_empty_string(self) -> None:
        assert metaphone("") == ""


class TestPhoneticSimilarity:
    """Tests for combined phonetic similarity."""

    def test_identical_words(self) -> None:
        assert phonetic_similarity("hello", "hello") == 1.0

    def test_similar_sounding(self) -> None:
        # Night and knight should be very similar
        sim = phonetic_similarity("night", "knight")
        assert sim >= 0.5

    def test_different_words(self) -> None:
        sim = phonetic_similarity("cat", "dog")
        assert sim < 0.5

    def test_empty_strings(self) -> None:
        assert phonetic_similarity("", "hello") == 0.0
        assert phonetic_similarity("", "") == 0.0


class TestGetPhonemes:
    """Tests for CMU dictionary phoneme lookup."""

    def test_known_word(self) -> None:
        phonemes = get_phonemes("hello")
        assert len(phonemes) > 0
        # Should contain HH and L
        flat = [p for variant in phonemes for p in variant]
        assert any("HH" in p for p in flat)

    def test_unknown_word(self) -> None:
        phonemes = get_phonemes("xyzzyplugh")
        assert phonemes == ()

    def test_empty_string(self) -> None:
        assert get_phonemes("") == ()

    def test_caching(self) -> None:
        # Should be cached, so second call is fast
        p1 = get_phonemes("test")
        p2 = get_phonemes("test")
        assert p1 == p2


class TestGetRhymePart:
    """Tests for rhyme part extraction."""

    def test_basic_extraction(self) -> None:
        # "night" = N AY1 T -> rhyme part should include AY1 T
        phonemes = ("N", "AY1", "T")
        rhyme = get_rhyme_part(phonemes)
        assert "AY1" in rhyme
        assert "T" in rhyme

    def test_empty_phonemes(self) -> None:
        assert get_rhyme_part(()) == ()


class TestStripStress:
    """Tests for stress marker removal."""

    def test_removes_digits(self) -> None:
        result = strip_stress(("AY1", "T"))
        assert result == ("AY", "T")

    def test_handles_no_stress(self) -> None:
        result = strip_stress(("N", "T"))
        assert result == ("N", "T")


class TestWordsRhyme:
    """Tests for rhyme detection."""

    def test_perfect_rhymes(self) -> None:
        assert words_rhyme("night", "light")
        assert words_rhyme("cat", "hat")
        assert words_rhyme("moon", "June")

    def test_non_rhymes(self) -> None:
        assert not words_rhyme("cat", "dog")
        assert not words_rhyme("hello", "world")

    def test_same_word_doesnt_rhyme(self) -> None:
        # By convention, same word doesn't rhyme with itself
        assert not words_rhyme("night", "night")

    def test_empty_strings(self) -> None:
        assert not words_rhyme("", "night")
        assert not words_rhyme("night", "")


class TestRhymeScore:
    """Tests for rhyme scoring."""

    def test_perfect_rhyme(self) -> None:
        score = rhyme_score("night", "light")
        assert score > 0.9

    def test_good_rhyme(self) -> None:
        score = rhyme_score("cat", "hat")
        assert score > 0.8

    def test_no_rhyme(self) -> None:
        score = rhyme_score("cat", "dog")
        assert score < 0.3

    def test_same_word(self) -> None:
        score = rhyme_score("night", "night")
        assert score == 0.0

    def test_range(self) -> None:
        """All scores should be in [0, 1]."""
        pairs = [
            ("night", "light"),
            ("cat", "dog"),
            ("hello", "world"),
            ("moon", "June"),
        ]
        for w1, w2 in pairs:
            score = rhyme_score(w1, w2)
            assert 0.0 <= score <= 1.0


class TestCountSyllables:
    """Tests for syllable counting."""

    def test_known_words(self) -> None:
        assert count_syllables("hello") == 2
        assert count_syllables("the") == 1
        assert count_syllables("beautiful") == 3

    def test_empty_string(self) -> None:
        assert count_syllables("") == 0

    def test_single_syllable(self) -> None:
        assert count_syllables("cat") == 1
        assert count_syllables("dog") == 1

    def test_multisyllable(self) -> None:
        assert count_syllables("computer") == 3
        assert count_syllables("university") == 5


class TestCountLineSyllables:
    """Tests for line syllable counting."""

    def test_simple_line(self) -> None:
        count = count_line_syllables("Hello world")
        assert count == 3  # hel-lo (2) + world (1)

    def test_empty_line(self) -> None:
        assert count_line_syllables("") == 0

    @pytest.mark.parametrize("poem", HAIKU_POEMS, ids=lambda p: p.name)
    def test_haiku_syllables(self, poem: PoemSpec) -> None:
        """Test syllable counting against haiku ground truth."""
        if poem.expected_syllables_per_line:
            from abide.primitives import parse_structure

            structure = parse_structure(poem.text)
            for i, line in enumerate(structure.lines):
                expected = poem.expected_syllables_per_line[i]
                actual = count_line_syllables(line)
                # Allow some tolerance for syllable counting
                assert abs(actual - expected) <= 2, (
                    f"Line {i}: expected ~{expected} syllables, got {actual}: {line}"
                )


class TestGetStressPattern:
    """Tests for stress pattern extraction."""

    def test_known_word(self) -> None:
        pattern = get_stress_pattern("hello")
        # Should be something like "01" (he-LLO)
        assert len(pattern) == 2

    def test_unknown_word(self) -> None:
        pattern = get_stress_pattern("xyzzy")
        assert pattern == ""  # Unknown word

    def test_empty_string(self) -> None:
        assert get_stress_pattern("") == ""
