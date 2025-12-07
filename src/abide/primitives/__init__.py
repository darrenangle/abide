"""
NLP primitives for poetic analysis.

This module provides foundational tools for analyzing text:
- String similarity metrics (Levenshtein, Jaro-Winkler)
- Phonetic encoders (Soundex, Metaphone)
- CMU dictionary integration for pronunciation
- Syllable counting
- Stress pattern detection
- Rhyme classification
"""

from abide.primitives.phonetics import (
    DEFAULT_RHYME_WEIGHTS,
    RhymeType,
    classify_rhyme,
    count_line_syllables,
    count_syllables,
    get_line_stress_pattern,
    get_phonemes,
    get_rhyme_part,
    get_stress_pattern,
    match_rating_codex,
    metaphone,
    nysiis,
    phonetic_similarity,
    rhyme_score,
    soundex,
    strip_stress,
    words_rhyme,
)
from abide.primitives.preprocessing import (
    NormalizationConfig,
    NormalizationMode,
    NormalizationResult,
    PoemStructure,
    count_words,
    extract_end_word,
    extract_end_words,
    normalize_whitespace,
    parse_structure,
    tokenize_line,
)
from abide.primitives.similarity import (
    hamming_distance,
    jaro_similarity,
    jaro_winkler_similarity,
    lcs_similarity,
    levenshtein_distance,
    longest_common_subsequence_length,
    normalized_levenshtein,
)

__all__ = [
    "DEFAULT_RHYME_WEIGHTS",
    "NormalizationConfig",
    "NormalizationMode",
    "NormalizationResult",
    "PoemStructure",
    "RhymeType",
    "classify_rhyme",
    "count_line_syllables",
    "count_syllables",
    "count_words",
    "extract_end_word",
    "extract_end_words",
    "get_line_stress_pattern",
    "get_phonemes",
    "get_rhyme_part",
    "get_stress_pattern",
    "hamming_distance",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "lcs_similarity",
    "levenshtein_distance",
    "longest_common_subsequence_length",
    "match_rating_codex",
    "metaphone",
    "normalize_whitespace",
    "normalized_levenshtein",
    "nysiis",
    "parse_structure",
    "phonetic_similarity",
    "rhyme_score",
    "soundex",
    "strip_stress",
    "tokenize_line",
    "words_rhyme",
]
