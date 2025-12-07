"""
String similarity metrics for fuzzy matching in poetry analysis.

Provides various distance and similarity functions used for:
- End-word matching in sestinas
- Refrain comparison in villanelles
- Near-rhyme detection
"""

from __future__ import annotations


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    The minimum number of single-character edits (insertions, deletions,
    or substitutions) required to transform s1 into s2.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Integer edit distance (0 = identical)

    Examples:
        >>> levenshtein_distance("kitten", "sitting")
        3
        >>> levenshtein_distance("", "abc")
        3
        >>> levenshtein_distance("same", "same")
        0
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity in range [0, 1].

    Args:
        s1: First string
        s2: Second string

    Returns:
        Float similarity where 1.0 = identical, 0.0 = completely different

    Examples:
        >>> normalized_levenshtein("hello", "hello")
        1.0
        >>> normalized_levenshtein("", "")
        1.0
        >>> 0.0 <= normalized_levenshtein("abc", "xyz") <= 1.0
        True
    """
    if not s1 and not s2:
        return 1.0

    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))

    return 1.0 - (distance / max_len)


def jaro_similarity(s1: str, s2: str) -> float:
    """
    Compute Jaro similarity between two strings.

    Good for short strings like words. Based on:
    - Number of matching characters
    - Number of transpositions

    Args:
        s1: First string
        s2: Second string

    Returns:
        Float similarity in [0, 1]

    Examples:
        >>> jaro_similarity("MARTHA", "MARHTA")  # doctest: +ELLIPSIS
        0.944...
        >>> jaro_similarity("", "")
        1.0
        >>> jaro_similarity("abc", "abc")
        1.0
    """
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    # Match window: floor(max(len1, len2) / 2) - 1
    match_distance = max(len1, len2) // 2 - 1
    match_distance = max(0, match_distance)

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matches
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

    return jaro


def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """
    Compute Jaro-Winkler similarity, which boosts score for common prefixes.

    Better than Jaro for words with common stems.

    Args:
        s1: First string
        s2: Second string
        prefix_weight: Scaling factor for prefix bonus (standard: 0.1)

    Returns:
        Float similarity in [0, 1]

    Examples:
        >>> jaro_winkler_similarity("MARTHA", "MARHTA") > jaro_similarity("MARTHA", "MARHTA")
        True
        >>> jaro_winkler_similarity("prefix_a", "prefix_b") > jaro_winkler_similarity("a_suffix", "b_suffix")
        True
    """
    jaro = jaro_similarity(s1, s2)

    # Find common prefix length (up to 4 chars)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * prefix_weight * (1 - jaro)


def hamming_distance(s1: str, s2: str) -> int | None:
    """
    Compute Hamming distance between two strings of equal length.

    The number of positions where corresponding characters differ.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Integer distance, or None if strings have different lengths

    Examples:
        >>> hamming_distance("karolin", "kathrin")
        3
        >>> hamming_distance("abc", "ab")  # Different lengths
        >>> hamming_distance("", "")
        0
    """
    if len(s1) != len(s2):
        return None

    return sum(c1 != c2 for c1, c2 in zip(s1, s2, strict=True))


def longest_common_subsequence_length(s1: str, s2: str) -> int:
    """
    Compute length of longest common subsequence.

    A subsequence is a sequence that can be derived by deleting some
    (or no) elements without changing the order.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Length of LCS

    Examples:
        >>> longest_common_subsequence_length("ABCDGH", "AEDFHR")
        3
        >>> longest_common_subsequence_length("", "abc")
        0
    """
    m, n = len(s1), len(s2)

    if m == 0 or n == 0:
        return 0

    # Use space-optimized DP (only need previous row)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


def lcs_similarity(s1: str, s2: str) -> float:
    """
    Compute similarity based on longest common subsequence.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Float similarity in [0, 1], ratio of LCS to average length

    Examples:
        >>> lcs_similarity("hello", "hello")
        1.0
        >>> lcs_similarity("", "")
        1.0
    """
    if not s1 and not s2:
        return 1.0

    lcs_len = longest_common_subsequence_length(s1, s2)
    avg_len = (len(s1) + len(s2)) / 2

    if avg_len == 0:
        return 1.0

    return lcs_len / avg_len
