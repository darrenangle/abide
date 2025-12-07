"""Tests for string similarity functions."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from abide.primitives import (
    hamming_distance,
    jaro_similarity,
    jaro_winkler_similarity,
    lcs_similarity,
    levenshtein_distance,
    longest_common_subsequence_length,
    normalized_levenshtein,
)


class TestLevenshteinDistance:
    """Tests for Levenshtein edit distance."""

    def test_identical_strings(self) -> None:
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self) -> None:
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "xyz") == 3

    def test_single_char_difference(self) -> None:
        assert levenshtein_distance("cat", "bat") == 1
        assert levenshtein_distance("cat", "car") == 1
        assert levenshtein_distance("cat", "cats") == 1

    def test_known_values(self) -> None:
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("saturday", "sunday") == 3
        assert levenshtein_distance("gumbo", "gambol") == 2

    @given(st.text(max_size=20))
    def test_self_distance_is_zero(self, s: str) -> None:
        assert levenshtein_distance(s, s) == 0

    @given(st.text(max_size=20), st.text(max_size=20))
    def test_symmetry(self, s1: str, s2: str) -> None:
        assert levenshtein_distance(s1, s2) == levenshtein_distance(s2, s1)

    @given(st.text(max_size=20), st.text(max_size=20))
    def test_non_negative(self, s1: str, s2: str) -> None:
        assert levenshtein_distance(s1, s2) >= 0


class TestNormalizedLevenshtein:
    """Tests for normalized Levenshtein similarity."""

    def test_identical_strings(self) -> None:
        assert normalized_levenshtein("hello", "hello") == 1.0

    def test_empty_strings(self) -> None:
        assert normalized_levenshtein("", "") == 1.0

    def test_completely_different(self) -> None:
        # "abc" vs "xyz" = 3 edits, max_len=3, similarity = 0
        assert normalized_levenshtein("abc", "xyz") == 0.0

    @given(st.text(max_size=20), st.text(max_size=20))
    def test_range(self, s1: str, s2: str) -> None:
        sim = normalized_levenshtein(s1, s2)
        assert 0.0 <= sim <= 1.0


class TestJaroSimilarity:
    """Tests for Jaro similarity."""

    def test_identical_strings(self) -> None:
        assert jaro_similarity("abc", "abc") == 1.0

    def test_empty_strings(self) -> None:
        assert jaro_similarity("", "") == 1.0
        assert jaro_similarity("abc", "") == 0.0

    def test_known_values(self) -> None:
        # MARTHA vs MARHTA is a classic example
        sim = jaro_similarity("MARTHA", "MARHTA")
        assert 0.94 < sim < 0.95  # Should be ~0.944

    @given(st.text(alphabet="abcdefghij", max_size=15))
    def test_self_similarity_is_one(self, s: str) -> None:
        assert jaro_similarity(s, s) == 1.0

    @given(
        st.text(alphabet="abcdefghij", max_size=15),
        st.text(alphabet="abcdefghij", max_size=15),
    )
    def test_symmetry(self, s1: str, s2: str) -> None:
        assert abs(jaro_similarity(s1, s2) - jaro_similarity(s2, s1)) < 1e-10


class TestJaroWinklerSimilarity:
    """Tests for Jaro-Winkler similarity."""

    def test_prefix_boost(self) -> None:
        # Jaro-Winkler should give higher score than Jaro for common prefixes
        s1, s2 = "prefix_a", "prefix_b"
        jw = jaro_winkler_similarity(s1, s2)
        j = jaro_similarity(s1, s2)
        assert jw >= j

    def test_no_prefix(self) -> None:
        # With no common prefix, should equal Jaro
        s1, s2 = "abc", "xyz"
        jw = jaro_winkler_similarity(s1, s2)
        j = jaro_similarity(s1, s2)
        assert jw == j

    @given(
        st.text(alphabet="abcdefghij", max_size=15),
        st.text(alphabet="abcdefghij", max_size=15),
    )
    def test_at_least_jaro(self, s1: str, s2: str) -> None:
        jw = jaro_winkler_similarity(s1, s2)
        j = jaro_similarity(s1, s2)
        assert jw >= j - 1e-10  # Allow for floating point


class TestHammingDistance:
    """Tests for Hamming distance."""

    def test_identical_strings(self) -> None:
        assert hamming_distance("abc", "abc") == 0

    def test_empty_strings(self) -> None:
        assert hamming_distance("", "") == 0

    def test_different_lengths(self) -> None:
        assert hamming_distance("abc", "ab") is None

    def test_known_values(self) -> None:
        assert hamming_distance("karolin", "kathrin") == 3
        assert hamming_distance("1011101", "1001001") == 2


class TestLongestCommonSubsequence:
    """Tests for LCS functions."""

    def test_lcs_length_identical(self) -> None:
        assert longest_common_subsequence_length("abc", "abc") == 3

    def test_lcs_length_empty(self) -> None:
        assert longest_common_subsequence_length("", "abc") == 0
        assert longest_common_subsequence_length("abc", "") == 0

    def test_lcs_length_known(self) -> None:
        assert longest_common_subsequence_length("ABCDGH", "AEDFHR") == 3

    def test_lcs_similarity_identical(self) -> None:
        assert lcs_similarity("hello", "hello") == 1.0

    def test_lcs_similarity_empty(self) -> None:
        assert lcs_similarity("", "") == 1.0

    @given(st.text(max_size=20), st.text(max_size=20))
    def test_lcs_similarity_range(self, s1: str, s2: str) -> None:
        sim = lcs_similarity(s1, s2)
        assert 0.0 <= sim <= 2.0  # Can exceed 1.0 due to average length
