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
    # Similarity
    "levenshtein_distance",
    "normalized_levenshtein",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "hamming_distance",
    "longest_common_subsequence_length",
    "lcs_similarity",
    # Phonetics
    "soundex",
    "metaphone",
    "nysiis",
    "match_rating_codex",
    "phonetic_similarity",
    "get_phonemes",
    "get_rhyme_part",
    "strip_stress",
    "words_rhyme",
    "rhyme_score",
    "count_syllables",
    "count_line_syllables",
    "get_stress_pattern",
    "get_line_stress_pattern",
    # Preprocessing
    "NormalizationMode",
    "NormalizationConfig",
    "NormalizationResult",
    "normalize_whitespace",
    "PoemStructure",
    "parse_structure",
    "extract_end_word",
    "extract_end_words",
    "tokenize_line",
    "count_words",
]
