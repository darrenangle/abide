"""
Phonetic encoding and analysis for poetry.

Provides phonetic encoders (Soundex, Metaphone) and CMU dictionary
integration for pronunciation-based analysis including rhyme detection.
"""

from __future__ import annotations

from functools import lru_cache

import jellyfish
import pronouncing

# ============================================================================
# Phonetic Encoders (via jellyfish)
# ============================================================================


def soundex(word: str) -> str:
    """
    Compute Soundex encoding of a word.

    Soundex encodes words by their sound, useful for matching
    words that sound similar but are spelled differently.

    Args:
        word: Word to encode

    Returns:
        4-character Soundex code (letter + 3 digits)

    Examples:
        >>> soundex("night")
        'N230'
        >>> soundex("knight")
        'K523'
        >>> soundex("Robert")
        'R163'
        >>> soundex("Rupert")
        'R163'
    """
    if not word:
        return "0000"
    return jellyfish.soundex(word)


def metaphone(word: str) -> str:
    """
    Compute Metaphone encoding of a word.

    More accurate than Soundex for English pronunciation.

    Args:
        word: Word to encode

    Returns:
        Metaphone code (variable length)

    Examples:
        >>> metaphone("night")
        'NT'
        >>> metaphone("knight")
        'NT'
    """
    if not word:
        return ""
    return jellyfish.metaphone(word)


def nysiis(word: str) -> str:
    """
    Compute NYSIIS (New York State Identification and Intelligence System) encoding.

    Another phonetic algorithm, sometimes more accurate than Soundex.

    Args:
        word: Word to encode

    Returns:
        NYSIIS code

    Examples:
        >>> nysiis("night")
        'NAGT'
        >>> nysiis("knight")
        'KNAGT'
    """
    if not word:
        return ""
    return jellyfish.nysiis(word)


def match_rating_codex(word: str) -> str:
    """
    Compute Match Rating Approach codex.

    Args:
        word: Word to encode

    Returns:
        MRA codex

    Examples:
        >>> match_rating_codex("night")
        'NGHT'
    """
    if not word:
        return ""
    return jellyfish.match_rating_codex(word)


def phonetic_similarity(word1: str, word2: str) -> float:
    """
    Compute phonetic similarity between two words.

    Uses multiple phonetic algorithms and returns weighted score.

    Args:
        word1: First word
        word2: Second word

    Returns:
        Similarity score in [0, 1]

    Examples:
        >>> phonetic_similarity("night", "knight") > 0.5
        True
        >>> phonetic_similarity("cat", "dog") < 0.5
        True
    """
    if not word1 or not word2:
        return 0.0

    if word1.lower() == word2.lower():
        return 1.0

    scores: list[float] = []

    # Soundex comparison
    s1, s2 = soundex(word1), soundex(word2)
    if s1 == s2:
        scores.append(1.0)
    else:
        # Partial credit for partial match
        matching = sum(c1 == c2 for c1, c2 in zip(s1, s2, strict=False))
        scores.append(matching / 4)

    # Metaphone comparison
    m1, m2 = metaphone(word1), metaphone(word2)
    if m1 and m2:
        if m1 == m2:
            scores.append(1.0)
        else:
            # Use Levenshtein on metaphone codes
            max_len = max(len(m1), len(m2))
            dist = jellyfish.levenshtein_distance(m1, m2)
            scores.append(1.0 - dist / max_len)

    return sum(scores) / len(scores) if scores else 0.0


# ============================================================================
# CMU Pronouncing Dictionary Integration
# ============================================================================


@lru_cache(maxsize=10000)
def get_phonemes(word: str) -> tuple[tuple[str, ...], ...]:
    """
    Get phoneme sequences for a word from CMU dictionary.

    Returns all pronunciation variants. Each phoneme may include
    stress markers (0=no stress, 1=primary, 2=secondary).

    Args:
        word: Word to look up

    Returns:
        Tuple of phoneme tuples (one per pronunciation variant)
        Empty tuple if word not found

    Examples:
        >>> phonemes = get_phonemes("hello")
        >>> len(phonemes) > 0
        True
        >>> "HH" in phonemes[0]
        True
    """
    if not word:
        return ()

    phones_list = pronouncing.phones_for_word(word.lower())
    if not phones_list:
        return ()

    return tuple(tuple(phones.split()) for phones in phones_list)


def get_rhyme_part(phonemes: tuple[str, ...]) -> tuple[str, ...]:
    """
    Extract the rhyming portion from a phoneme sequence.

    The rhyme part starts from the last stressed vowel to the end.

    Args:
        phonemes: Sequence of phonemes with stress markers

    Returns:
        The rhyming portion of the phonemes

    Examples:
        >>> # "night" = N AY1 T -> rhyme part is AY1 T
        >>> phonemes = ("N", "AY1", "T")
        >>> get_rhyme_part(phonemes)
        ('AY1', 'T')
    """
    if not phonemes:
        return ()

    # Find last stressed vowel (stress marker 1 or 2)
    last_stressed_idx = -1
    for i, phoneme in enumerate(phonemes):
        if any(c in "12" for c in phoneme):
            last_stressed_idx = i

    if last_stressed_idx == -1:
        # No stress found, try finding any vowel (has digit)
        for i, phoneme in enumerate(phonemes):
            if any(c.isdigit() for c in phoneme):
                last_stressed_idx = i

    if last_stressed_idx == -1:
        # Still nothing, return last 2 phonemes
        return phonemes[-2:] if len(phonemes) >= 2 else phonemes

    return phonemes[last_stressed_idx:]


def strip_stress(phonemes: tuple[str, ...]) -> tuple[str, ...]:
    """
    Remove stress markers from phonemes for comparison.

    Args:
        phonemes: Phonemes with stress markers

    Returns:
        Phonemes without stress markers

    Examples:
        >>> strip_stress(("AY1", "T"))
        ('AY', 'T')
    """
    return tuple("".join(c for c in p if not c.isdigit()) for p in phonemes)


def words_rhyme(
    word1: str,
    word2: str,
    strict: bool = False,
) -> bool:
    """
    Check if two words rhyme.

    Args:
        word1: First word
        word2: Second word
        strict: If True, require exact rhyme part match

    Returns:
        True if words rhyme

    Examples:
        >>> words_rhyme("night", "light")
        True
        >>> words_rhyme("night", "bright")
        True
        >>> words_rhyme("cat", "dog")
        False
    """
    if not word1 or not word2:
        return False

    # Same word doesn't rhyme with itself (by convention)
    if word1.lower() == word2.lower():
        return False

    phonemes1 = get_phonemes(word1)
    phonemes2 = get_phonemes(word2)

    if not phonemes1 or not phonemes2:
        # Fall back to suffix matching if not in dictionary
        return _suffix_rhyme(word1, word2)

    # Check all pronunciation combinations
    for p1 in phonemes1:
        for p2 in phonemes2:
            rhyme1 = get_rhyme_part(p1)
            rhyme2 = get_rhyme_part(p2)

            if strict:
                if rhyme1 == rhyme2:
                    return True
            else:
                # Compare without stress markers
                if strip_stress(rhyme1) == strip_stress(rhyme2):
                    return True

    return False


def _suffix_rhyme(word1: str, word2: str, min_suffix: int = 2) -> bool:
    """Fallback rhyme detection using suffix matching."""
    w1, w2 = word1.lower(), word2.lower()

    # Find longest common suffix
    common = 0
    for i in range(1, min(len(w1), len(w2)) + 1):
        if w1[-i] == w2[-i]:
            common = i
        else:
            break

    return common >= min_suffix


def rhyme_score(word1: str, word2: str) -> float:
    """
    Compute a rhyme similarity score between two words.

    Args:
        word1: First word
        word2: Second word

    Returns:
        Score in [0, 1] where 1.0 = perfect rhyme

    Examples:
        >>> rhyme_score("night", "light") > 0.9
        True
        >>> rhyme_score("cat", "hat") > 0.8
        True
        >>> rhyme_score("cat", "dog") < 0.3
        True
    """
    if not word1 or not word2:
        return 0.0

    if word1.lower() == word2.lower():
        return 0.0  # Same word doesn't rhyme

    phonemes1 = get_phonemes(word1)
    phonemes2 = get_phonemes(word2)

    if not phonemes1 or not phonemes2:
        # Fallback to suffix matching
        w1, w2 = word1.lower(), word2.lower()
        common = 0
        for i in range(1, min(len(w1), len(w2)) + 1):
            if w1[-i] == w2[-i]:
                common = i
            else:
                break
        if common == 0:
            return 0.0
        avg_len = (len(w1) + len(w2)) / 2
        return min(1.0, common / (avg_len * 0.5))

    # Find best rhyme score across all pronunciation combinations
    best_score = 0.0

    for p1 in phonemes1:
        for p2 in phonemes2:
            rhyme1 = strip_stress(get_rhyme_part(p1))
            rhyme2 = strip_stress(get_rhyme_part(p2))

            if rhyme1 == rhyme2:
                best_score = max(best_score, 1.0)
            else:
                # Partial credit based on matching phonemes from end
                max_len = max(len(rhyme1), len(rhyme2))
                if max_len > 0:
                    matches = 0
                    for i in range(1, min(len(rhyme1), len(rhyme2)) + 1):
                        if rhyme1[-i] == rhyme2[-i]:
                            matches += 1
                        else:
                            break
                    partial = matches / max_len
                    best_score = max(best_score, partial)

    return best_score


# ============================================================================
# Syllable Counting
# ============================================================================


@lru_cache(maxsize=10000)
def count_syllables(word: str) -> int:
    """
    Count syllables in a word.

    Uses CMU dictionary when available, falls back to heuristics.

    Args:
        word: Word to count syllables in

    Returns:
        Number of syllables

    Examples:
        >>> count_syllables("hello")
        2
        >>> count_syllables("beautiful")
        3
        >>> count_syllables("the")
        1
    """
    if not word:
        return 0

    word = word.lower().strip()

    # Try CMU dictionary first
    phonemes = get_phonemes(word)
    if phonemes:
        # Count vowel phonemes (those with stress markers)
        # Use first pronunciation
        return sum(1 for p in phonemes[0] if any(c.isdigit() for c in p))

    # Fallback: heuristic syllable counting
    return _heuristic_syllables(word)


def _heuristic_syllables(word: str) -> int:
    """
    Estimate syllables using rules when dictionary lookup fails.

    Based on vowel counting with adjustments for common patterns.
    """
    word = word.lower()

    # Remove non-alphabetic
    word = "".join(c for c in word if c.isalpha())

    if not word:
        return 0

    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Adjustments
    # Silent e at end
    if word.endswith("e") and count > 1:
        count -= 1

    # -le at end usually adds syllable
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1

    # -ed endings
    if word.endswith("ed") and not word.endswith(("ted", "ded")):
        count = max(1, count - 1)

    return max(1, count)


def count_line_syllables(line: str) -> int:
    """
    Count total syllables in a line.

    Args:
        line: Line of text

    Returns:
        Total syllable count

    Examples:
        >>> count_line_syllables("Hello world")
        3
    """
    # Simple word tokenization
    import re

    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", line)
    return sum(count_syllables(w) for w in words)


# ============================================================================
# Stress Patterns
# ============================================================================


def get_stress_pattern(word: str) -> str:
    """
    Get the stress pattern for a word.

    Args:
        word: Word to analyze

    Returns:
        String of stress markers (0, 1, 2) for each syllable
        Empty string if word not found

    Examples:
        >>> get_stress_pattern("hello")
        '01'
        >>> get_stress_pattern("beautiful")
        '100'
    """
    phonemes = get_phonemes(word)
    if not phonemes:
        return ""

    # Extract stress markers from first pronunciation
    pattern = ""
    for phoneme in phonemes[0]:
        for char in phoneme:
            if char.isdigit():
                pattern += char
                break

    return pattern


def get_line_stress_pattern(line: str) -> str:
    """
    Get concatenated stress pattern for a line.

    Args:
        line: Line of text

    Returns:
        Combined stress pattern for all words

    Examples:
        >>> pattern = get_line_stress_pattern("To be or not to be")
        >>> len(pattern) > 0
        True
    """
    import re

    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", line)

    pattern = ""
    for word in words:
        word_pattern = get_stress_pattern(word)
        if word_pattern:
            pattern += word_pattern
        else:
            # Unknown word: assume one unstressed syllable per heuristic count
            pattern += "0" * _heuristic_syllables(word)

    return pattern
