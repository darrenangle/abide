[](https://claude.ai/new)

let’s say i wanted to create a reward environment / reward producing test suite that used classical NLP algorithms and text algorithms to create verifiable or spectrum rewards for a poem’s adherence to strict formats like a sestina or villanelle, with some fuzziness for spaces vs tabs and newline variety, what algorithms might i use to create this environment? Give me extremely detailed examples using your ultrathink ability

What a fascinating project! Let me think through this deeply.

<thinking> This is a really interesting technical question about creating a reward environment for evaluating poetry adherence to strict forms. The user wants to use classical NLP/text algorithms to create verifiable rewards for forms like sestinas and villanelles.

Let me think through this systematically:

## What defines these poetic forms?

**Sestina:**

- 6 stanzas of 6 lines each, plus a 3-line envoi
- 6 end-words that rotate in a specific pattern
- Pattern: If stanza 1 ends with words 1,2,3,4,5,6, then:
    - Stanza 2: 6,1,5,2,4,3
    - Stanza 3: 3,6,4,1,2,5
    - Stanza 4: 5,3,2,6,1,4
    - Stanza 5: 4,5,1,3,6,2
    - Stanza 6: 2,4,6,5,3,1
    - Envoi: 2-5, 4-3, 6-1 (middle and end of each line)

**Villanelle:**

- 19 lines total
- 5 tercets (3-line stanzas) + 1 quatrain (4-line stanza)
- Two refrains (A1 and A2) and two rhymes (A and B)
- Pattern: A1bA2 abA1 abA2 abA1 abA2 abA1A2
- Lines 1, 6, 12, 18 are the same (A1 refrain)
- Lines 3, 9, 15, 19 are the same (A2 refrain)

## Algorithms to consider:

1. **Structural parsing:**
    - Line counting
    - Stanza detection (blank line detection with fuzzy matching for whitespace)
    - Word tokenization
2. **End-word extraction and matching:**
    - For sestinas: extract final word of each line, normalize, compare patterns
    - String similarity metrics for "close enough" matches:
        - Levenshtein distance
        - Jaro-Winkler similarity
        - Soundex/Metaphone for phonetic similarity
3. **Rhyme detection:**
    - Phonetic encoding (CMU Pronouncing Dictionary)
    - Suffix matching
    - Phoneme sequence alignment
4. **Refrain matching:**
    - Exact string matching
    - Edit distance for fuzzy matching
    - TF-IDF or cosine similarity for semantic similarity (if allowing paraphrase)
5. **Meter detection (if relevant):**
    - Syllable counting
    - Stress pattern detection using pronunciation dictionaries

Let me design a detailed reward system:

## Reward Components:

### 1. Structural Rewards

- Line count accuracy
- Stanza count and size accuracy
- Whitespace normalization

### 2. Pattern Adherence Rewards

- End-word rotation (sestina)
- Rhyme scheme (villanelle)
- Refrain repetition (villanelle)

### 3. Fuzzy Matching Approaches

- Normalized edit distance
- Phonetic similarity
- Semantic similarity (optional)

Let me write out detailed algorithms and code examples. </thinking>

# Reward Environment for Strict Poetic Forms

This is a rich problem that combines structural parsing, phonetic analysis, string matching, and pattern verification. Let me break down the algorithmic approaches for creating a robust reward system.

## 1. Core Architectural Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    POEM REWARD ENVIRONMENT                      │
├─────────────────────────────────────────────────────────────────┤
│  Input: Raw poem text                                           │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │ Preprocessor │ → │  Structural  │ → │ Pattern Matcher  │    │
│  │ (normalize)  │   │   Parser     │   │ (form-specific)  │    │
│  └──────────────┘   └──────────────┘   └──────────────────┘    │
│         │                  │                    │               │
│         ▼                  ▼                    ▼               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │  Whitespace  │   │    Line/     │   │  Rhyme/End-word  │    │
│  │   Rewards    │   │Stanza Rewards│   │     Rewards      │    │
│  └──────────────┘   └──────────────┘   └──────────────────┘    │
│         │                  │                    │               │
│         └──────────────────┴────────────────────┘               │
│                            │                                    │
│                            ▼                                    │
│                   ┌─────────────────┐                          │
│                   │ Reward Aggregator│                          │
│                   │  (weighted sum)  │                          │
│                   └─────────────────┘                          │
│                            │                                    │
│                            ▼                                    │
│                    Final Score ∈ [0, 1]                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Preprocessing & Normalization

The first challenge is handling whitespace variability (tabs vs spaces, `\n` vs `\r\n`, multiple blank lines).

python

```python
import re
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum

class WhitespaceNormalization(Enum):
    STRICT = "strict"      # Exact matching
    LENIENT = "lenient"    # Normalize all whitespace
    FUZZY = "fuzzy"        # Probabilistic penalties

@dataclass
class NormalizationConfig:
    mode: WhitespaceNormalization = WhitespaceNormalization.LENIENT
    tab_to_spaces: int = 4
    max_consecutive_blanks: int = 1
    strip_trailing: bool = True
    
def preprocess_poem(
    raw_text: str, 
    config: NormalizationConfig
) -> Tuple[str, float]:
    """
    Normalize poem text and return a preprocessing quality score.
    
    Returns:
        (normalized_text, quality_score) where quality_score ∈ [0, 1]
        penalizes excessive normalization needed.
    """
    original_length = len(raw_text)
    penalties = 0.0
    
    # 1. Normalize line endings (CRLF → LF)
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    crlf_count = raw_text.count('\r')
    penalties += 0.01 * crlf_count  # Minor penalty for non-standard endings
    
    # 2. Handle tabs
    if config.mode != WhitespaceNormalization.STRICT:
        tab_count = text.count('\t')
        text = text.replace('\t', ' ' * config.tab_to_spaces)
        penalties += 0.005 * tab_count  # Very minor tab penalty
    
    # 3. Normalize consecutive blank lines
    if config.mode != WhitespaceNormalization.STRICT:
        # Count excessive blank lines before normalizing
        blank_line_pattern = r'\n{3,}'
        excessive_blanks = len(re.findall(blank_line_pattern, text))
        penalties += 0.02 * excessive_blanks
        
        # Replace multiple blanks with standard stanza break
        max_newlines = config.max_consecutive_blanks + 1
        replacement = '\n' * max_newlines
        text = re.sub(r'\n{2,}', replacement, text)
    
    # 4. Strip trailing whitespace from lines
    if config.strip_trailing:
        lines = text.split('\n')
        stripped_chars = sum(len(line) - len(line.rstrip()) for line in lines)
        penalties += 0.001 * stripped_chars
        text = '\n'.join(line.rstrip() for line in lines)
    
    # 5. Strip leading/trailing whitespace from whole poem
    text = text.strip()
    
    # Calculate quality score (higher is better)
    quality_score = max(0.0, 1.0 - penalties)
    
    return text, quality_score
```

---

## 3. Structural Parsing

### 3.1 Line and Stanza Extraction

python

```python
@dataclass
class PoemStructure:
    lines: List[str]
    stanzas: List[List[str]]
    line_count: int
    stanza_count: int
    stanza_sizes: List[int]

def parse_structure(normalized_text: str) -> PoemStructure:
    """
    Parse poem into lines and stanzas.
    
    Stanza breaks are detected by blank lines.
    """
    lines = normalized_text.split('\n')
    
    # Remove empty lines but track stanza boundaries
    stanzas = []
    current_stanza = []
    
    for line in lines:
        if line.strip() == '':
            if current_stanza:  # End of stanza
                stanzas.append(current_stanza)
                current_stanza = []
        else:
            current_stanza.append(line)
    
    # Don't forget the last stanza
    if current_stanza:
        stanzas.append(current_stanza)
    
    # Flatten for line list
    all_lines = [line for stanza in stanzas for line in stanza]
    
    return PoemStructure(
        lines=all_lines,
        stanzas=stanzas,
        line_count=len(all_lines),
        stanza_count=len(stanzas),
        stanza_sizes=[len(s) for s in stanzas]
    )
```

### 3.2 Structural Reward Calculation

python

```python
def structural_reward_sestina(structure: PoemStructure) -> dict:
    """
    Calculate structural rewards for a sestina.
    
    Expected structure:
    - 7 stanzas total (6 sixains + 1 tercet envoi)
    - First 6 stanzas: 6 lines each
    - Final stanza (envoi): 3 lines
    - Total: 39 lines
    """
    rewards = {}
    
    # Line count reward (39 expected)
    expected_lines = 39
    line_diff = abs(structure.line_count - expected_lines)
    # Gaussian-like penalty: perfect = 1.0, degrades with distance
    rewards['line_count'] = math.exp(-0.5 * (line_diff / 2) ** 2)
    
    # Stanza count reward (7 expected)
    expected_stanzas = 7
    stanza_diff = abs(structure.stanza_count - expected_stanzas)
    rewards['stanza_count'] = math.exp(-0.5 * (stanza_diff / 1) ** 2)
    
    # Stanza size reward
    expected_sizes = [6, 6, 6, 6, 6, 6, 3]
    if structure.stanza_count == expected_stanzas:
        size_errors = sum(
            abs(actual - expected) 
            for actual, expected in zip(structure.stanza_sizes, expected_sizes)
        )
        rewards['stanza_sizes'] = math.exp(-0.3 * size_errors)
    else:
        # Partial credit: check how many stanzas match expected sizes
        matches = sum(
            1 for i, size in enumerate(structure.stanza_sizes)
            if i < len(expected_sizes) and size == expected_sizes[i]
        )
        rewards['stanza_sizes'] = matches / len(expected_sizes)
    
    return rewards


def structural_reward_villanelle(structure: PoemStructure) -> dict:
    """
    Calculate structural rewards for a villanelle.
    
    Expected structure:
    - 6 stanzas total (5 tercets + 1 quatrain)
    - First 5 stanzas: 3 lines each
    - Final stanza: 4 lines
    - Total: 19 lines
    """
    rewards = {}
    
    # Line count reward (19 expected)
    expected_lines = 19
    line_diff = abs(structure.line_count - expected_lines)
    rewards['line_count'] = math.exp(-0.5 * (line_diff / 2) ** 2)
    
    # Stanza count reward (6 expected)
    expected_stanzas = 6
    stanza_diff = abs(structure.stanza_count - expected_stanzas)
    rewards['stanza_count'] = math.exp(-0.5 * (stanza_diff / 1) ** 2)
    
    # Stanza size reward
    expected_sizes = [3, 3, 3, 3, 3, 4]
    if structure.stanza_count == expected_stanzas:
        size_errors = sum(
            abs(actual - expected) 
            for actual, expected in zip(structure.stanza_sizes, expected_sizes)
        )
        rewards['stanza_sizes'] = math.exp(-0.3 * size_errors)
    else:
        matches = sum(
            1 for i, size in enumerate(structure.stanza_sizes)
            if i < len(expected_sizes) and size == expected_sizes[i]
        )
        rewards['stanza_sizes'] = matches / len(expected_sizes)
    
    return rewards
```

---

## 4. End-Word Extraction & Matching (Sestina)

The sestina's defining feature is its end-word rotation pattern.

### 4.1 End-Word Extraction

python

```python
import string

def extract_end_word(line: str) -> str:
    """
    Extract the final word from a line, handling punctuation.
    
    Uses a multi-stage approach:
    1. Strip trailing whitespace
    2. Remove trailing punctuation
    3. Extract final token
    4. Normalize to lowercase
    """
    line = line.rstrip()
    
    # Remove trailing punctuation (but preserve internal punctuation)
    while line and line[-1] in string.punctuation:
        line = line[:-1]
    
    # Handle edge case of empty line
    if not line.strip():
        return ""
    
    # Split and get last word
    words = line.split()
    if not words:
        return ""
    
    return words[-1].lower()


def extract_all_end_words(structure: PoemStructure) -> List[str]:
    """Extract end words from all lines."""
    return [extract_end_word(line) for line in structure.lines]
```

### 4.2 Sestina Pattern Definition & Verification

python

```python
# The sestina end-word rotation pattern
# If stanza 1 has end-words indexed as [0,1,2,3,4,5],
# subsequent stanzas permute them according to this rule:
# Position i in new stanza gets word from position SESTINA_ROTATION[i] of previous stanza

SESTINA_ROTATION = [5, 0, 4, 1, 3, 2]  # The "retrogradatio cruciata"

def generate_sestina_pattern() -> List[List[int]]:
    """
    Generate the complete end-word pattern for all 6 stanzas.
    
    Returns list of 6 lists, each containing 6 indices indicating
    which original end-word (0-5) should appear at each position.
    """
    pattern = []
    current = list(range(6))  # [0, 1, 2, 3, 4, 5]
    
    for stanza_num in range(6):
        pattern.append(current.copy())
        # Apply rotation for next stanza
        current = [current[i] for i in SESTINA_ROTATION]
    
    return pattern

# Pre-computed for efficiency
SESTINA_STANZA_PATTERNS = generate_sestina_pattern()
# Result:
# Stanza 0: [0, 1, 2, 3, 4, 5]
# Stanza 1: [5, 0, 4, 1, 3, 2]
# Stanza 2: [2, 5, 3, 0, 1, 4]
# Stanza 3: [4, 2, 1, 5, 0, 3]
# Stanza 4: [3, 4, 0, 2, 5, 1]
# Stanza 5: [1, 3, 5, 4, 2, 0]

# Envoi pattern: words appear mid-line and end-line
# Traditional: (2,5), (4,3), (6,1) → using 0-indexing: (1,4), (3,2), (5,0)
SESTINA_ENVOI_PATTERN = [(1, 4), (3, 2), (5, 0)]
```

### 4.3 String Similarity Algorithms for Fuzzy Matching

python

```python
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Classic dynamic programming Levenshtein edit distance.
    
    Time: O(mn), Space: O(min(m,n))
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Normalized similarity score ∈ [0, 1].
    
    1.0 = identical strings
    0.0 = completely different
    """
    if not s1 and not s2:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    return 1.0 - (distance / max_len)


def jaro_similarity(s1: str, s2: str) -> float:
    """
    Jaro similarity - good for short strings like words.
    
    Based on:
    - Number of matching characters
    - Number of transpositions
    
    Returns score ∈ [0, 1]
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
    
    jaro = (
        matches / len1 +
        matches / len2 +
        (matches - transpositions / 2) / matches
    ) / 3
    
    return jaro


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Jaro-Winkler similarity - boosts score for common prefixes.
    
    p = scaling factor for prefix bonus (standard: 0.1)
    
    Better than Jaro for words with common stems.
    """
    jaro = jaro_similarity(s1, s2)
    
    # Find common prefix length (up to 4 chars)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    return jaro + prefix_len * p * (1 - jaro)
```

### 4.4 Phonetic Similarity (Soundex & Metaphone)

python

```python
def soundex(word: str) -> str:
    """
    Classic Soundex algorithm - encodes words by sound.
    
    Useful for matching words that sound similar but are spelled differently.
    E.g., "night" and "knight" both → "N230"
    """
    if not word:
        return "0000"
    
    word = word.upper()
    
    # Soundex coding
    codes = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
        # A, E, I, O, U, H, W, Y are ignored (coded as '')
    }
    
    # Keep first letter
    result = word[0]
    
    # Encode rest
    prev_code = codes.get(word[0], '')
    
    for char in word[1:]:
        code = codes.get(char, '')
        if code and code != prev_code:
            result += code
        prev_code = code if code else prev_code
    
    # Pad or truncate to 4 characters
    result = (result + '000')[:4]
    
    return result


def double_metaphone(word: str) -> Tuple[str, str]:
    """
    Double Metaphone algorithm - more accurate than Soundex.
    
    Returns two encodings (primary and alternate) to handle
    words with multiple pronunciations.
    
    This is a simplified version - full implementation is ~500 lines.
    """
    # For brevity, here's a simplified version
    # In production, use the `metaphone` library
    
    word = word.upper()
    vowels = 'AEIOU'
    
    primary = ''
    alternate = ''
    
    i = 0
    while i < len(word) and len(primary) < 4:
        char = word[i]
        
        if char in vowels:
            if i == 0:
                primary += 'A'
                alternate += 'A'
        elif char == 'B':
            primary += 'P'
            alternate += 'P'
        elif char == 'C':
            if i + 1 < len(word) and word[i + 1] in 'EIY':
                primary += 'S'
                alternate += 'S'
            else:
                primary += 'K'
                alternate += 'K'
        elif char == 'D':
            primary += 'T'
            alternate += 'T'
        elif char == 'G':
            if i + 1 < len(word) and word[i + 1] in 'EIY':
                primary += 'J'
                alternate += 'J'
            else:
                primary += 'K'
                alternate += 'K'
        elif char == 'K':
            primary += 'K'
            alternate += 'K'
        elif char == 'P':
            if i + 1 < len(word) and word[i + 1] == 'H':
                primary += 'F'
                alternate += 'F'
                i += 1
            else:
                primary += 'P'
                alternate += 'P'
        elif char in 'FJLMNRSTWXZ':
            primary += char
            alternate += char
        
        i += 1
    
    return primary[:4], alternate[:4]


def phonetic_similarity(word1: str, word2: str) -> float:
    """
    Combined phonetic similarity using multiple algorithms.
    
    Returns weighted average of:
    - Soundex match (binary)
    - Metaphone match (binary for each variant)
    """
    scores = []
    
    # Soundex comparison
    s1, s2 = soundex(word1), soundex(word2)
    soundex_score = 1.0 if s1 == s2 else 0.0
    # Partial credit for partial match
    if s1 != s2:
        matching_chars = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        soundex_score = matching_chars / 4
    scores.append(soundex_score)
    
    # Double Metaphone comparison
    m1_primary, m1_alt = double_metaphone(word1)
    m2_primary, m2_alt = double_metaphone(word2)
    
    metaphone_matches = [
        m1_primary == m2_primary,
        m1_primary == m2_alt,
        m1_alt == m2_primary,
        m1_alt == m2_alt
    ]
    metaphone_score = 1.0 if any(metaphone_matches) else 0.0
    scores.append(metaphone_score)
    
    return sum(scores) / len(scores)
```

### 4.5 Combined End-Word Similarity Function

python

```python
@dataclass
class SimilarityWeights:
    exact: float = 0.4
    levenshtein: float = 0.2
    jaro_winkler: float = 0.2
    phonetic: float = 0.2

def word_similarity(
    word1: str, 
    word2: str, 
    weights: SimilarityWeights = SimilarityWeights()
) -> float:
    """
    Compute overall similarity between two words using multiple algorithms.
    
    Returns weighted combination of different similarity metrics.
    """
    word1, word2 = word1.lower().strip(), word2.lower().strip()
    
    # Exact match gets perfect score
    if word1 == word2:
        return 1.0
    
    scores = {
        'exact': 0.0,  # Already checked above
        'levenshtein': normalized_levenshtein_similarity(word1, word2),
        'jaro_winkler': jaro_winkler_similarity(word1, word2),
        'phonetic': phonetic_similarity(word1, word2)
    }
    
    weighted_sum = (
        weights.exact * scores['exact'] +
        weights.levenshtein * scores['levenshtein'] +
        weights.jaro_winkler * scores['jaro_winkler'] +
        weights.phonetic * scores['phonetic']
    )
    
    total_weight = (
        weights.exact + weights.levenshtein + 
        weights.jaro_winkler + weights.phonetic
    )
    
    return weighted_sum / total_weight
```

### 4.6 Sestina End-Word Pattern Reward

python

```python
def sestina_endword_reward(
    structure: PoemStructure,
    similarity_threshold: float = 0.8
) -> dict:
    """
    Calculate reward for sestina end-word pattern adherence.
    
    This is the core algorithmic challenge: verifying that the same 6 words
    rotate through prescribed positions across 6 stanzas.
    """
    rewards = {}
    
    # Need at least 6 stanzas with 6 lines each
    if structure.stanza_count < 6:
        return {'endword_pattern': 0.0, 'endword_details': 'insufficient stanzas'}
    
    # Extract end words from first 6 stanzas
    stanza_endwords = []
    for i, stanza in enumerate(structure.stanzas[:6]):
        if len(stanza) < 6:
            return {
                'endword_pattern': 0.0, 
                'endword_details': f'stanza {i+1} has only {len(stanza)} lines'
            }
        endwords = [extract_end_word(line) for line in stanza[:6]]
        stanza_endwords.append(endwords)
    
    # The 6 canonical end-words are from stanza 1
    canonical_words = stanza_endwords[0]
    
    # Verify each subsequent stanza follows the rotation pattern
    pattern_scores = []
    
    for stanza_idx in range(1, 6):
        expected_pattern = SESTINA_STANZA_PATTERNS[stanza_idx]
        actual_words = stanza_endwords[stanza_idx]
        
        stanza_scores = []
        for position, expected_word_idx in enumerate(expected_pattern):
            expected_word = canonical_words[expected_word_idx]
            actual_word = actual_words[position]
            
            similarity = word_similarity(expected_word, actual_word)
            stanza_scores.append(similarity)
        
        pattern_scores.append(sum(stanza_scores) / len(stanza_scores))
    
    # Overall pattern adherence
    rewards['endword_pattern'] = sum(pattern_scores) / len(pattern_scores)
    
    # Detailed per-stanza scores
    rewards['endword_per_stanza'] = pattern_scores
    
    # Binary "strict" score (all above threshold)
    rewards['endword_strict'] = float(
        all(s >= similarity_threshold for s in pattern_scores)
    )
    
    return rewards
```

---

## 5. Rhyme Detection (Villanelle)

The villanelle requires a specific rhyme scheme: ABA for tercets, ABAA for the final quatrain.

### 5.1 Phoneme-Based Rhyme Detection (CMU Dict)

python

```python
# You would load this from the actual CMU Pronouncing Dictionary
# Here's a mock version showing the structure
CMU_DICT = {
    'night': [['N', 'AY1', 'T']],
    'knight': [['N', 'AY1', 'T']],
    'light': [['L', 'AY1', 'T']],
    'sight': [['S', 'AY1', 'T']],
    'bright': [['B', 'R', 'AY1', 'T']],
    'day': [['D', 'EY1']],
    'way': [['W', 'EY1']],
    'say': [['S', 'EY1']],
    'play': [['P', 'L', 'EY1']],
    'stay': [['S', 'T', 'EY1']],
    # ... thousands more entries
}

def load_cmu_dict(filepath: str = '/usr/share/dict/cmudict') -> dict:
    """Load CMU Pronouncing Dictionary."""
    cmu = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith(';;;'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0].lower()
                    # Handle entries like "WORD(1)" for alternate pronunciations
                    if '(' in word:
                        word = word.split('(')[0]
                    phonemes = parts[1:]
                    if word not in cmu:
                        cmu[word] = []
                    cmu[word].append(phonemes)
    except FileNotFoundError:
        pass
    return cmu


def get_rhyme_part(phonemes: List[str]) -> List[str]:
    """
    Extract the rhyming portion of a phoneme sequence.
    
    Rhyme = from the last stressed vowel to the end.
    Stressed vowels have digits 1 or 2 (primary/secondary stress).
    """
    # Find last stressed vowel
    last_stressed_idx = -1
    for i, phoneme in enumerate(phonemes):
        if any(char.isdigit() and char in '12' for char in phoneme):
            last_stressed_idx = i
    
    if last_stressed_idx == -1:
        # No stress found, use last vowel
        for i, phoneme in enumerate(phonemes):
            if any(char.isdigit() for char in phoneme):
                last_stressed_idx = i
    
    if last_stressed_idx == -1:
        # Still nothing, return last 2 phonemes
        return phonemes[-2:] if len(phonemes) >= 2 else phonemes
    
    return phonemes[last_stressed_idx:]


def phoneme_rhyme_score(word1: str, word2: str, cmu_dict: dict) -> float:
    """
    Calculate rhyme score based on phoneme matching.
    
    Perfect rhyme: identical phonemes from stressed vowel onward.
    Near rhyme: similar phonemes (partial credit).
    """
    word1, word2 = word1.lower(), word2.lower()
    
    # Get pronunciations
    pron1_list = cmu_dict.get(word1, [])
    pron2_list = cmu_dict.get(word2, [])
    
    if not pron1_list or not pron2_list:
        # Fall back to suffix matching
        return suffix_rhyme_score(word1, word2)
    
    # Try all pronunciation combinations
    best_score = 0.0
    
    for pron1 in pron1_list:
        for pron2 in pron2_list:
            rhyme1 = get_rhyme_part(pron1)
            rhyme2 = get_rhyme_part(pron2)
            
            # Remove stress markers for comparison
            clean1 = [''.join(c for c in p if not c.isdigit()) for p in rhyme1]
            clean2 = [''.join(c for c in p if not c.isdigit()) for p in rhyme2]
            
            if clean1 == clean2:
                best_score = max(best_score, 1.0)
            else:
                # Partial credit based on matching phonemes
                max_len = max(len(clean1), len(clean2))
                if max_len > 0:
                    # Align from the end
                    matches = 0
                    for i in range(1, min(len(clean1), len(clean2)) + 1):
                        if clean1[-i] == clean2[-i]:
                            matches += 1
                        else:
                            break
                    partial_score = matches / max_len
                    best_score = max(best_score, partial_score)
    
    return best_score


def suffix_rhyme_score(word1: str, word2: str) -> float:
    """
    Fallback rhyme detection using suffix matching.
    
    Useful when words aren't in the pronunciation dictionary.
    """
    word1, word2 = word1.lower(), word2.lower()
    
    # Find longest common suffix
    min_len = min(len(word1), len(word2))
    common_suffix_len = 0
    
    for i in range(1, min_len + 1):
        if word1[-i] == word2[-i]:
            common_suffix_len = i
        else:
            break
    
    if common_suffix_len == 0:
        return 0.0
    
    # Normalize by average word length
    avg_len = (len(word1) + len(word2)) / 2
    
    # Require at least 2 characters for meaningful rhyme
    if common_suffix_len < 2:
        return common_suffix_len * 0.2
    
    # Score based on proportion of word that rhymes
    return min(1.0, common_suffix_len / (avg_len * 0.5))
```

### 5.2 Villanelle Rhyme Scheme Verification

python

```python
# Villanelle rhyme scheme
# Lines are indexed 0-18
# A rhyme: lines 0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18
# B rhyme: lines 1, 4, 7, 10, 13, 16

VILLANELLE_A_LINES = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
VILLANELLE_B_LINES = [1, 4, 7, 10, 13, 16]

def villanelle_rhyme_reward(
    structure: PoemStructure,
    cmu_dict: dict
) -> dict:
    """
    Calculate reward for villanelle rhyme scheme adherence.
    """
    rewards = {}
    
    if structure.line_count < 19:
        return {
            'rhyme_scheme': 0.0, 
            'details': f'only {structure.line_count} lines, need 19'
        }
    
    end_words = [extract_end_word(line) for line in structure.lines[:19]]
    
    # Check A rhymes (all should rhyme with line 0)
    a_reference = end_words[0]
    a_scores = []
    for line_idx in VILLANELLE_A_LINES[1:]:  # Skip first (it's the reference)
        score = phoneme_rhyme_score(a_reference, end_words[line_idx], cmu_dict)
        a_scores.append(score)
    
    rewards['a_rhyme_avg'] = sum(a_scores) / len(a_scores) if a_scores else 0.0
    rewards['a_rhyme_scores'] = a_scores
    
    # Check B rhymes (all should rhyme with line 1)
    b_reference = end_words[1]
    b_scores = []
    for line_idx in VILLANELLE_B_LINES[1:]:  # Skip first
        score = phoneme_rhyme_score(b_reference, end_words[line_idx], cmu_dict)
        b_scores.append(score)
    
    rewards['b_rhyme_avg'] = sum(b_scores) / len(b_scores) if b_scores else 0.0
    rewards['b_rhyme_scores'] = b_scores
    
    # Combined rhyme scheme score
    all_scores = a_scores + b_scores
    rewards['rhyme_scheme'] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    return rewards
```

---

## 6. Refrain Detection (Villanelle)

The villanelle has two refrains that repeat exactly:

- A1 (line 1) repeats at lines 6, 12, 18
- A2 (line 3) repeats at lines 9, 15, 19

### 6.1 Refrain Matching Algorithms

python

```python
def normalize_line_for_comparison(line: str) -> str:
    """
    Normalize a line for refrain comparison.
    
    - Lowercase
    - Remove extra whitespace
    - Optionally remove punctuation (configurable)
    """
    line = line.lower().strip()
    line = ' '.join(line.split())  # Normalize whitespace
    return line


def refrain_exact_match(line1: str, line2: str) -> float:
    """Binary exact match after normalization."""
    return 1.0 if normalize_line_for_comparison(line1) == normalize_line_for_comparison(line2) else 0.0


def refrain_edit_distance_score(line1: str, line2: str) -> float:
    """
    Soft match using normalized edit distance.
    
    Allows for minor variations in refrain repetition.
    """
    norm1 = normalize_line_for_comparison(line1)
    norm2 = normalize_line_for_comparison(line2)
    
    return normalized_levenshtein_similarity(norm1, norm2)


def refrain_token_overlap(line1: str, line2: str) -> float:
    """
    Jaccard similarity based on word tokens.
    
    Good for detecting refrains with minor word changes.
    """
    tokens1 = set(normalize_line_for_comparison(line1).split())
    tokens2 = set(normalize_line_for_comparison(line2).split())
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union


def longest_common_subsequence_ratio(s1: str, s2: str) -> float:
    """
    LCS-based similarity - good for detecting refrains with insertions.
    
    Returns ratio of LCS length to average string length.
    """
    m, n = len(s1), len(s2)
    
    if m == 0 or n == 0:
        return 1.0 if m == n else 0.0
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    avg_length = (m + n) / 2
    
    return lcs_length / avg_length


def combined_refrain_score(
    line1: str, 
    line2: str,
    exact_weight: float = 0.5,
    edit_weight: float = 0.25,
    lcs_weight: float = 0.25
) -> float:
    """
    Combined refrain similarity using multiple algorithms.
    """
    exact = refrain_exact_match(line1, line2)
    
    # If exact match, return 1.0 immediately
    if exact == 1.0:
        return 1.0
    
    edit = refrain_edit_distance_score(line1, line2)
    lcs = longest_common_subsequence_ratio(
        normalize_line_for_comparison(line1),
        normalize_line_for_comparison(line2)
    )
    
    return (exact_weight * exact + edit_weight * edit + lcs_weight * lcs)
```

### 6.2 Villanelle Refrain Verification

python

```python
# Villanelle refrain positions (0-indexed)
VILLANELLE_A1_POSITIONS = [0, 5, 11, 17]  # Lines 1, 6, 12, 18
VILLANELLE_A2_POSITIONS = [2, 8, 14, 18]  # Lines 3, 9, 15, 19

def villanelle_refrain_reward(structure: PoemStructure) -> dict:
    """
    Calculate reward for villanelle refrain repetition.
    """
    rewards = {}
    
    if structure.line_count < 19:
        return {'refrain_score': 0.0, 'details': 'insufficient lines'}
    
    lines = structure.lines[:19]
    
    # A1 refrain check (line 0 should repeat at 5, 11, 17)
    a1_reference = lines[VILLANELLE_A1_POSITIONS[0]]
    a1_scores = []
    for pos in VILLANELLE_A1_POSITIONS[1:]:
        score = combined_refrain_score(a1_reference, lines[pos])
        a1_scores.append(score)
    
    rewards['a1_refrain_avg'] = sum(a1_scores) / len(a1_scores)
    rewards['a1_refrain_scores'] = a1_scores
    
    # A2 refrain check (line 2 should repeat at 8, 14, 18)
    a2_reference = lines[VILLANELLE_A2_POSITIONS[0]]
    a2_scores = []
    for pos in VILLANELLE_A2_POSITIONS[1:]:
        score = combined_refrain_score(a2_reference, lines[pos])
        a2_scores.append(score)
    
    rewards['a2_refrain_avg'] = sum(a2_scores) / len(a2_scores)
    rewards['a2_refrain_scores'] = a2_scores
    
    # Combined refrain score
    all_scores = a1_scores + a2_scores
    rewards['refrain_score'] = sum(all_scores) / len(all_scores)
    
    # Strict binary score (all refrains match)
    threshold = 0.95
    rewards['refrain_strict'] = float(all(s >= threshold for s in all_scores))
    
    return rewards
```

---

## 7. Complete Reward Environment

### 7.1 Full Implementation

python

```python
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from enum import Enum

class PoemForm(Enum):
    SESTINA = "sestina"
    VILLANELLE = "villanelle"
    
@dataclass
class RewardWeights:
    """Weights for different reward components."""
    preprocessing: float = 0.05
    structure: float = 0.30
    pattern: float = 0.65  # End-words for sestina, rhyme+refrain for villanelle
    
@dataclass
class RewardResult:
    """Complete reward computation result."""
    total_score: float
    component_scores: Dict[str, float]
    component_weights: Dict[str, float]
    details: Dict[str, any]
    normalized_text: str
    
class PoemRewardEnvironment:
    """
    Reward environment for evaluating poetry adherence to strict forms.
    """
    
    def __init__(
        self,
        form: PoemForm,
        weights: RewardWeights = RewardWeights(),
        normalization_config: NormalizationConfig = NormalizationConfig(),
        similarity_threshold: float = 0.8,
        cmu_dict_path: Optional[str] = None
    ):
        self.form = form
        self.weights = weights
        self.norm_config = normalization_config
        self.similarity_threshold = similarity_threshold
        
        # Load CMU dictionary for rhyme detection
        self.cmu_dict = {}
        if cmu_dict_path:
            self.cmu_dict = load_cmu_dict(cmu_dict_path)
        
    def compute_reward(self, raw_poem: str) -> RewardResult:
        """
        Compute the complete reward for a poem.
        
        Returns a RewardResult with total score and component breakdown.
        """
        # Step 1: Preprocessing
        normalized_text, preprocess_score = preprocess_poem(
            raw_poem, self.norm_config
        )
        
        # Step 2: Structural parsing
        structure = parse_structure(normalized_text)
        
        # Step 3: Form-specific rewards
        if self.form == PoemForm.SESTINA:
            return self._compute_sestina_reward(
                normalized_text, structure, preprocess_score
            )
        elif self.form == PoemForm.VILLANELLE:
            return self._compute_villanelle_reward(
                normalized_text, structure, preprocess_score
            )
        else:
            raise ValueError(f"Unknown form: {self.form}")
    
    def _compute_sestina_reward(
        self,
        normalized_text: str,
        structure: PoemStructure,
        preprocess_score: float
    ) -> RewardResult:
        """Compute reward for sestina form."""
        
        component_scores = {}
        details = {}
        
        # Preprocessing score
        component_scores['preprocessing'] = preprocess_score
        
        # Structural rewards
        struct_rewards = structural_reward_sestina(structure)
        struct_score = (
            struct_rewards['line_count'] * 0.3 +
            struct_rewards['stanza_count'] * 0.3 +
            struct_rewards['stanza_sizes'] * 0.4
        )
        component_scores['structure'] = struct_score
        details['structure'] = struct_rewards
        
        # End-word pattern rewards
        endword_rewards = sestina_endword_reward(
            structure, self.similarity_threshold
        )
        pattern_score = endword_rewards.get('endword_pattern', 0.0)
        component_scores['pattern'] = pattern_score
        details['endword'] = endword_rewards
        
        # Compute weighted total
        total_score = (
            self.weights.preprocessing * component_scores['preprocessing'] +
            self.weights.structure * component_scores['structure'] +
            self.weights.pattern * component_scores['pattern']
        )
        
        return RewardResult(
            total_score=total_score,
            component_scores=component_scores,
            component_weights={
                'preprocessing': self.weights.preprocessing,
                'structure': self.weights.structure,
                'pattern': self.weights.pattern
            },
            details=details,
            normalized_text=normalized_text
        )
    
    def _compute_villanelle_reward(
        self,
        normalized_text: str,
        structure: PoemStructure,
        preprocess_score: float
    ) -> RewardResult:
        """Compute reward for villanelle form."""
        
        component_scores = {}
        details = {}
        
        # Preprocessing score
        component_scores['preprocessing'] = preprocess_score
        
        # Structural rewards
        struct_rewards = structural_reward_villanelle(structure)
        struct_score = (
            struct_rewards['line_count'] * 0.3 +
            struct_rewards['stanza_count'] * 0.3 +
            struct_rewards['stanza_sizes'] * 0.4
        )
        component_scores['structure'] = struct_score
        details['structure'] = struct_rewards
        
        # Rhyme scheme rewards
        rhyme_rewards = villanelle_rhyme_reward(structure, self.cmu_dict)
        rhyme_score = rhyme_rewards.get('rhyme_scheme', 0.0)
        details['rhyme'] = rhyme_rewards
        
        # Refrain rewards
        refrain_rewards = villanelle_refrain_reward(structure)
        refrain_score = refrain_rewards.get('refrain_score', 0.0)
        details['refrain'] = refrain_rewards
        
        # Combined pattern score (rhyme + refrain)
        pattern_score = (rhyme_score * 0.4 + refrain_score * 0.6)
        component_scores['pattern'] = pattern_score
        
        # Compute weighted total
        total_score = (
            self.weights.preprocessing * component_scores['preprocessing'] +
            self.weights.structure * component_scores['structure'] +
            self.weights.pattern * component_scores['pattern']
        )
        
        return RewardResult(
            total_score=total_score,
            component_scores=component_scores,
            component_weights={
                'preprocessing': self.weights.preprocessing,
                'structure': self.weights.structure,
                'pattern': self.weights.pattern
            },
            details=details,
            normalized_text=normalized_text
        )
```

### 7.2 Usage Example

python

```python
# Example: Evaluate a villanelle

villanelle_example = """
Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light.

Though wise men at their end know dark is right,
Because their words had forked no lightning they
Do not go gentle into that good night.

Good men, the last wave by, crying how bright
Their frail deeds might have danced in a green bay,
Rage, rage against the dying of the light.

Wild men who caught and sang the sun in flight,
And learn, too late, they grieved it on its way,
Do not go gentle into that good night.

Grave men, near death, who see with blinding sight
Blind eyes could blaze like meteors and be gay,
Rage, rage against the dying of the light.

And you, my father, there on the sad height,
Curse, bless, me now with your fierce tears, I pray.
Do not go gentle into that good night.
Rage, rage against the dying of the light.
"""

# Create environment
env = PoemRewardEnvironment(
    form=PoemForm.VILLANELLE,
    weights=RewardWeights(
        preprocessing=0.05,
        structure=0.25,
        pattern=0.70
    ),
    cmu_dict_path='/path/to/cmudict'
)

# Compute reward
result = env.compute_reward(villanelle_example)

print(f"Total Score: {result.total_score:.3f}")
print(f"Components:")
for name, score in result.component_scores.items():
    weight = result.component_weights[name]
    print(f"  {name}: {score:.3f} (weight: {weight})")
```

---

## 8. Advanced: Spectrum Rewards with Confidence Intervals

For RL training, you might want continuous spectrum rewards rather than just point estimates.

python

```python
import numpy as np
from scipy import stats

@dataclass
class SpectrumReward:
    """Reward with uncertainty quantification."""
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    distribution: str  # 'beta', 'normal', 'truncated_normal'
    
def compute_spectrum_reward(
    component_scores: Dict[str, float],
    component_uncertainties: Dict[str, float],
    weights: Dict[str, float]
) -> SpectrumReward:
    """
    Compute reward with uncertainty propagation.
    
    Uses Monte Carlo sampling to propagate uncertainty through
    the weighted combination of components.
    """
    n_samples = 10000
    samples = np.zeros(n_samples)
    
    for name, weight in weights.items():
        score = component_scores[name]
        uncertainty = component_uncertainties.get(name, 0.1)
        
        # Sample from truncated normal (scores are bounded [0, 1])
        component_samples = stats.truncnorm(
            (0 - score) / uncertainty,
            (1 - score) / uncertainty,
            loc=score,
            scale=uncertainty
        ).rvs(n_samples)
        
        samples += weight * component_samples
    
    # Normalize by total weight
    total_weight = sum(weights.values())
    samples /= total_weight
    
    mean = np.mean(samples)
    std = np.std(samples)
    ci = (np.percentile(samples, 2.5), np.percentile(samples, 97.5))
    
    return SpectrumReward(
        mean=mean,
        std=std,
        confidence_interval=ci,
        distribution='truncated_normal'
    )
```

---

## 9. Algorithm Summary Table

|Component|Algorithm|Use Case|Fuzzy Handling|
|---|---|---|---|
|**Preprocessing**|Regex normalization|Whitespace, line endings|Penalty scoring for deviations|
|**Structure**|Line/stanza counting|Form validation|Gaussian decay from expected values|
|**End-word matching**|Levenshtein distance|Sestina rotation|Normalized similarity ∈ [0,1]|
|**End-word matching**|Jaro-Winkler|Short word comparison|Prefix boost for stems|
|**End-word matching**|Soundex/Metaphone|Phonetic similarity|Binary + partial match|
|**Rhyme detection**|CMU phoneme alignment|Villanelle A/B rhymes|Suffix-based fallback|
|**Refrain matching**|LCS ratio|Villanelle repetition|Handles insertions/deletions|
|**Refrain matching**|Jaccard (token)|Word-level similarity|Handles reordering|

This framework gives you a solid foundation. You can extend it with:

- **Meter detection** using syllable counting and stress patterns
- **Semantic similarity** using word embeddings for "conceptual rhyme"
- **N-gram analysis** for detecting consistent style/voice across stanzas

i’d also like to consider a slight abstraction here, where we have a system that can compose the instructions from a composable set of abstractions / restrictions, that way we can compose the instructions and test the outputs programmatically while teaching the model to generalize to known and unknown forms

This is a brilliant abstraction! You're essentially designing a **constraint composition system** that can both generate instructions for the model AND verify outputs against those same constraints. This creates a powerful feedback loop for generalization.

Let me think through this deeply.

<thinking> The user wants to create a composable system of poetic constraints that can: 1. Generate human-readable instructions from composed constraints 2. Verify outputs programmatically against those constraints 3. Enable generalization to both known forms (sestina, villanelle) and novel/unknown forms

This is really about building a DSL (Domain Specific Language) for poetic forms that can:

- Compose atomic constraints (line count, rhyme scheme, repetition patterns, etc.)
- Serialize to natural language instructions
- Compile to verification functions
- Support both exact and fuzzy matching

Let me think about the key abstractions:

## Core Constraint Types

1. **Structural Constraints**
    - LineCount(n) - poem has exactly n lines
    - StanzaCount(n) - poem has exactly n stanzas
    - StanzaPattern([sizes]) - stanzas have specific sizes
    - LineLength(min, max) - syllable or word count bounds
2. **Positional Constraints**
    - EndWordAt(position, word_id) - specific word at line end
    - WordAt(line, position, word_id) - word at specific position
    - LineAt(position, content_id) - entire line content
3. **Relational Constraints**
    - Rhymes(pos1, pos2) - two positions rhyme
    - RhymeScheme(pattern) - e.g., "ABAB"
    - Repeats(pos1, pos2) - exact repetition
    - Rotates(word_set, pattern) - sestina-style rotation
4. **Group Constraints**
    - AllRhyme(positions) - all listed positions rhyme together
    - AllRepeat(positions) - all listed positions are identical
    - CyclicPermutation(base_positions, rotation_rule)
5. **Meta Constraints**
    - And(c1, c2, ...) - all must hold
    - Or(c1, c2, ...) - at least one must hold
    - Optional(c, weight) - bonus if holds
    - Soft(c, decay_fn) - spectrum reward

## Instruction Generation

Each constraint needs to be able to:

- Serialize to natural language
- Combine with other constraints coherently
- Handle context (e.g., "the first stanza" vs "stanza 1")

## Verification Compilation

Each constraint compiles to:

- A verification function: (poem) -> score ∈ [0, 1]
- Dependencies on other constraints (e.g., RhymeScheme depends on LineCount)
- Fuzzy matching parameters

## Novel Form Generation

For teaching generalization, we want to:

- Randomly compose valid constraint sets
- Ensure consistency (no contradictory constraints)
- Generate both the instruction and the verifier
- Vary difficulty (number of constraints, strictness)

Let me design this system in detail. </thinking>

# Composable Constraint System for Poetic Forms

This is a powerful idea—essentially building a **bidirectional DSL** where the same constraint specification:

1. **Generates** natural language instructions for the model
2. **Compiles** to verification functions for reward computation
3. **Composes** to create both known and novel forms

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPOSABLE POETRY CONSTRAINT SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     CONSTRAINT PRIMITIVES                            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │Structural│ │Positional│ │Relational│ │ Phonetic │ │ Semantic │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     COMPOSITION OPERATORS                            │   │
│  │         And │ Or │ Optional │ Weighted │ Conditional │ Grouped       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                    ┌───────────────┼───────────────┐                       │
│                    ▼               ▼               ▼                       │
│  ┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐  │
│  │ INSTRUCTION COMPILER│ │VERIFIER COMPILER│ │ FORM TEMPLATE LIBRARY   │  │
│  │                     │ │                 │ │                         │  │
│  │ constraints →       │ │ constraints →   │ │ sestina = [...]         │  │
│  │ natural language    │ │ reward function │ │ villanelle = [...]      │  │
│  └─────────────────────┘ └─────────────────┘ │ novel = compose(...)    │  │
│            │                     │           └─────────────────────────┘  │
│            ▼                     ▼                                         │
│     ┌─────────────┐      ┌─────────────┐                                  │
│     │ INSTRUCTION │      │   REWARD    │                                  │
│     │   STRING    │      │  FUNCTION   │                                  │
│     └─────────────┘      └─────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Type System

python

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List, Dict, Tuple, Optional, Callable, Union, 
    Set, Any, TypeVar, Generic, Protocol
)
from enum import Enum, auto
import math
import re
from functools import reduce

# ============================================================================
# FOUNDATIONAL TYPES
# ============================================================================

class MatchMode(Enum):
    """How strictly to match constraints."""
    EXACT = auto()      # Binary: 0 or 1
    FUZZY = auto()      # Spectrum: [0, 1] with similarity
    SOFT = auto()        # Gaussian decay from target

@dataclass(frozen=True)
class Position:
    """
    A position reference in the poem.
    
    Can be absolute (line 5) or relative (last line of stanza 2).
    """
    line: Optional[int] = None           # 0-indexed line number
    stanza: Optional[int] = None         # 0-indexed stanza number
    line_in_stanza: Optional[int] = None # Line within stanza
    word: Optional[int] = None           # Word position in line (0=first, -1=last)
    
    def resolve(self, structure: 'PoemStructure') -> Tuple[int, Optional[int]]:
        """
        Resolve to (line_index, word_index).
        
        Returns absolute indices into the poem.
        """
        if self.line is not None:
            line_idx = self.line
        elif self.stanza is not None and self.line_in_stanza is not None:
            # Calculate absolute line from stanza-relative
            if self.stanza >= len(structure.stanzas):
                raise ValueError(f"Stanza {self.stanza} doesn't exist")
            lines_before = sum(len(s) for s in structure.stanzas[:self.stanza])
            line_idx = lines_before + self.line_in_stanza
        else:
            raise ValueError("Position must specify line or (stanza, line_in_stanza)")
        
        # Handle negative indexing
        if line_idx < 0:
            line_idx = structure.line_count + line_idx
            
        return (line_idx, self.word)
    
    def describe(self) -> str:
        """Human-readable description of this position."""
        if self.line is not None:
            line_desc = f"line {self.line + 1}"
        elif self.stanza is not None:
            stanza_ord = _ordinal(self.stanza + 1)
            if self.line_in_stanza == 0:
                line_desc = f"the first line of the {stanza_ord} stanza"
            elif self.line_in_stanza == -1:
                line_desc = f"the last line of the {stanza_ord} stanza"
            else:
                line_ord = _ordinal(self.line_in_stanza + 1)
                line_desc = f"the {line_ord} line of the {stanza_ord} stanza"
        else:
            line_desc = "unknown position"
            
        if self.word is not None:
            if self.word == -1:
                return f"the last word of {line_desc}"
            elif self.word == 0:
                return f"the first word of {line_desc}"
            else:
                return f"word {self.word + 1} of {line_desc}"
        return line_desc


def _ordinal(n: int) -> str:
    """Convert number to ordinal string."""
    if 11 <= n % 100 <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


@dataclass
class ConstraintContext:
    """
    Shared context for constraint evaluation.
    
    Holds references to extracted features that multiple constraints might need.
    """
    structure: 'PoemStructure'
    end_words: List[str]
    rhyme_groups: Dict[str, List[int]]  # Computed rhyme groupings
    phonetic_cache: Dict[str, List[str]]  # Word -> phonemes
    cmu_dict: Dict[str, List[List[str]]]
    
    @classmethod
    def from_poem(cls, text: str, cmu_dict: dict) -> 'ConstraintContext':
        """Build context from raw poem text."""
        normalized, _ = preprocess_poem(text, NormalizationConfig())
        structure = parse_structure(normalized)
        end_words = [extract_end_word(line) for line in structure.lines]
        
        return cls(
            structure=structure,
            end_words=end_words,
            rhyme_groups={},
            phonetic_cache={},
            cmu_dict=cmu_dict
        )


@dataclass 
class VerificationResult:
    """Result of verifying a single constraint."""
    score: float                          # [0, 1]
    passed: bool                          # Binary judgment
    constraint_name: str
    details: Dict[str, Any] = field(default_factory=dict)
    sub_results: List['VerificationResult'] = field(default_factory=list)

# ============================================================================
# ABSTRACT CONSTRAINT BASE
# ============================================================================

class Constraint(ABC):
    """
    Base class for all poetic constraints.
    
    A constraint is a bidirectional specification that can:
    1. Generate natural language instructions
    2. Verify a poem and produce a reward score
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this constraint type."""
        pass
    
    @abstractmethod
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        """
        Generate natural language instruction for this constraint.
        
        Args:
            context: Optional dict with info about other constraints
                     (useful for generating coherent combined instructions)
        """
        pass
    
    @abstractmethod
    def verify(
        self, 
        ctx: ConstraintContext,
        mode: MatchMode = MatchMode.FUZZY
    ) -> VerificationResult:
        """
        Verify this constraint against a poem.
        
        Args:
            ctx: Pre-computed context with poem structure and features
            mode: How strictly to evaluate (exact, fuzzy, soft)
            
        Returns:
            VerificationResult with score and details
        """
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Return names of constraints this one depends on.
        
        Used for ordering evaluation and instruction generation.
        """
        pass
    
    def __and__(self, other: 'Constraint') -> 'And':
        """Compose with AND: both must hold."""
        return And([self, other])
    
    def __or__(self, other: 'Constraint') -> 'Or':
        """Compose with OR: at least one must hold."""
        return Or([self, other])
    
    def weighted(self, weight: float) -> 'Weighted':
        """Wrap with explicit weight."""
        return Weighted(self, weight)
    
    def optional(self, bonus: float = 0.1) -> 'Optional_':
        """Make this constraint optional (bonus points)."""
        return Optional_(self, bonus)
```

---

## 3. Structural Constraints

python

```python
# ============================================================================
# STRUCTURAL CONSTRAINTS
# ============================================================================

@dataclass
class LineCount(Constraint):
    """Constraint on total number of lines."""
    count: int
    tolerance: int = 0  # For fuzzy matching
    
    @property
    def name(self) -> str:
        return f"line_count_{self.count}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if self.tolerance == 0:
            return f"The poem must have exactly {self.count} lines."
        else:
            return f"The poem should have approximately {self.count} lines (±{self.tolerance})."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        actual = ctx.structure.line_count
        diff = abs(actual - self.count)
        
        if mode == MatchMode.EXACT:
            score = 1.0 if diff == 0 else 0.0
        elif mode == MatchMode.FUZZY:
            # Linear decay within tolerance, then faster decay
            if diff <= self.tolerance:
                score = 1.0
            else:
                score = max(0, 1.0 - (diff - self.tolerance) * 0.1)
        else:  # SOFT
            score = math.exp(-0.5 * (diff / max(1, self.tolerance + 1)) ** 2)
        
        return VerificationResult(
            score=score,
            passed=diff <= self.tolerance,
            constraint_name=self.name,
            details={'expected': self.count, 'actual': actual, 'diff': diff}
        )
    
    def get_dependencies(self) -> List[str]:
        return []


@dataclass
class StanzaCount(Constraint):
    """Constraint on number of stanzas."""
    count: int
    tolerance: int = 0
    
    @property
    def name(self) -> str:
        return f"stanza_count_{self.count}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        stanza_word = "stanza" if self.count == 1 else "stanzas"
        if self.tolerance == 0:
            return f"The poem must have exactly {self.count} {stanza_word}."
        else:
            return f"The poem should have approximately {self.count} {stanza_word}."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        actual = ctx.structure.stanza_count
        diff = abs(actual - self.count)
        
        if mode == MatchMode.EXACT:
            score = 1.0 if diff == 0 else 0.0
        else:
            score = math.exp(-0.5 * (diff / max(1, self.tolerance + 1)) ** 2)
        
        return VerificationResult(
            score=score,
            passed=diff <= self.tolerance,
            constraint_name=self.name,
            details={'expected': self.count, 'actual': actual}
        )
    
    def get_dependencies(self) -> List[str]:
        return []


@dataclass
class StanzaStructure(Constraint):
    """Constraint on the size of each stanza."""
    sizes: List[int]  # [6, 6, 6, 3] means first 3 stanzas have 6 lines, last has 3
    
    @property
    def name(self) -> str:
        return f"stanza_structure_{'_'.join(map(str, self.sizes))}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if len(set(self.sizes[:-1])) == 1 and self.sizes[-1] != self.sizes[0]:
            # Pattern like [6,6,6,6,6,6,3] - regular stanzas + different final
            main_size = self.sizes[0]
            final_size = self.sizes[-1]
            main_count = len(self.sizes) - 1
            return (
                f"The poem should have {main_count} stanzas of {main_size} lines each, "
                f"followed by a final stanza of {final_size} lines."
            )
        elif len(set(self.sizes)) == 1:
            # All same size
            return f"Each stanza should have exactly {self.sizes[0]} lines."
        else:
            # Describe each
            parts = []
            for i, size in enumerate(self.sizes):
                parts.append(f"stanza {i+1}: {size} lines")
            return "The stanza structure should be: " + ", ".join(parts) + "."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        actual_sizes = ctx.structure.stanza_sizes
        
        # Handle length mismatch
        if len(actual_sizes) != len(self.sizes):
            length_penalty = abs(len(actual_sizes) - len(self.sizes)) * 0.2
            base_score = max(0, 1.0 - length_penalty)
        else:
            base_score = 1.0
        
        # Compare sizes that exist in both
        min_len = min(len(actual_sizes), len(self.sizes))
        size_scores = []
        for i in range(min_len):
            diff = abs(actual_sizes[i] - self.sizes[i])
            if mode == MatchMode.EXACT:
                size_scores.append(1.0 if diff == 0 else 0.0)
            else:
                size_scores.append(math.exp(-0.5 * diff ** 2))
        
        if size_scores:
            size_score = sum(size_scores) / len(size_scores)
        else:
            size_score = 0.0
        
        final_score = base_score * size_score
        
        return VerificationResult(
            score=final_score,
            passed=actual_sizes == self.sizes,
            constraint_name=self.name,
            details={'expected': self.sizes, 'actual': actual_sizes}
        )
    
    def get_dependencies(self) -> List[str]:
        return []
```

---

## 4. Relational Constraints (Rhyme, Repetition, Rotation)

python

```python
# ============================================================================
# RELATIONAL CONSTRAINTS
# ============================================================================

@dataclass
class Rhymes(Constraint):
    """Two positions must rhyme."""
    pos1: Position
    pos2: Position
    
    @property
    def name(self) -> str:
        return f"rhymes_{id(self)}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        desc1 = self.pos1.describe()
        desc2 = self.pos2.describe()
        return f"{desc1.capitalize()} must rhyme with {desc2}."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        try:
            idx1, word1 = self.pos1.resolve(ctx.structure)
            idx2, word2 = self.pos2.resolve(ctx.structure)
        except (ValueError, IndexError) as e:
            return VerificationResult(
                score=0.0, passed=False, constraint_name=self.name,
                details={'error': str(e)}
            )
        
        # Get words (default to end words)
        if word1 == -1 or word1 is None:
            w1 = ctx.end_words[idx1] if idx1 < len(ctx.end_words) else ""
        else:
            words = ctx.structure.lines[idx1].split()
            w1 = words[word1] if word1 < len(words) else ""
            
        if word2 == -1 or word2 is None:
            w2 = ctx.end_words[idx2] if idx2 < len(ctx.end_words) else ""
        else:
            words = ctx.structure.lines[idx2].split()
            w2 = words[word2] if word2 < len(words) else ""
        
        score = phoneme_rhyme_score(w1, w2, ctx.cmu_dict)
        
        if mode == MatchMode.EXACT:
            score = 1.0 if score > 0.9 else 0.0
        
        return VerificationResult(
            score=score,
            passed=score > 0.7,
            constraint_name=self.name,
            details={'word1': w1, 'word2': w2, 'positions': (idx1, idx2)}
        )
    
    def get_dependencies(self) -> List[str]:
        return []


@dataclass
class RhymeScheme(Constraint):
    """
    Define a rhyme scheme for the poem.
    
    Pattern like "ABAB CDCD EFEF GG" where same letters rhyme.
    Spaces indicate stanza breaks (optional, for readability).
    """
    pattern: str
    
    @property
    def name(self) -> str:
        return f"rhyme_scheme_{self.pattern.replace(' ', '_')}"
    
    def _parse_pattern(self) -> List[str]:
        """Extract rhyme labels, ignoring spaces."""
        return [c for c in self.pattern.upper() if c.isalpha()]
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        clean_pattern = self.pattern.upper()
        
        # Check for common patterns and give friendly descriptions
        if clean_pattern.replace(' ', '') == 'ABABCDCDEFEFGG':
            return "The poem should follow a Shakespearean sonnet rhyme scheme (ABAB CDCD EFEF GG)."
        elif clean_pattern.replace(' ', '') == 'ABBAABBACDECDE':
            return "The poem should follow a Petrarchan sonnet rhyme scheme (ABBA ABBA CDE CDE)."
        else:
            return f"The poem should follow the rhyme scheme: {clean_pattern}."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        labels = self._parse_pattern()
        
        if len(labels) > len(ctx.end_words):
            return VerificationResult(
                score=0.0, passed=False, constraint_name=self.name,
                details={'error': f'Pattern has {len(labels)} positions but poem has {len(ctx.end_words)} lines'}
            )
        
        # Group lines by rhyme label
        groups: Dict[str, List[int]] = {}
        for i, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(i)
        
        # Check rhymes within each group
        group_scores = []
        for label, indices in groups.items():
            if len(indices) < 2:
                continue  # Single occurrence, nothing to rhyme with
            
            # Check all pairs in group
            pair_scores = []
            reference_word = ctx.end_words[indices[0]]
            for idx in indices[1:]:
                word = ctx.end_words[idx]
                pair_scores.append(phoneme_rhyme_score(reference_word, word, ctx.cmu_dict))
            
            group_scores.append(sum(pair_scores) / len(pair_scores))
        
        if not group_scores:
            final_score = 1.0  # No rhymes required
        else:
            final_score = sum(group_scores) / len(group_scores)
        
        if mode == MatchMode.EXACT:
            final_score = 1.0 if final_score > 0.9 else 0.0
        
        return VerificationResult(
            score=final_score,
            passed=final_score > 0.7,
            constraint_name=self.name,
            details={'pattern': labels, 'groups': groups, 'group_scores': group_scores}
        )
    
    def get_dependencies(self) -> List[str]:
        return []


@dataclass
class Repeats(Constraint):
    """Two lines must be identical (refrain)."""
    pos1: Position
    pos2: Position
    
    @property
    def name(self) -> str:
        return f"repeats_{id(self)}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        desc1 = self.pos1.describe()
        desc2 = self.pos2.describe()
        return f"{desc1.capitalize()} must be repeated exactly as {desc2}."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        try:
            idx1, _ = self.pos1.resolve(ctx.structure)
            idx2, _ = self.pos2.resolve(ctx.structure)
            line1 = ctx.structure.lines[idx1]
            line2 = ctx.structure.lines[idx2]
        except (ValueError, IndexError) as e:
            return VerificationResult(
                score=0.0, passed=False, constraint_name=self.name,
                details={'error': str(e)}
            )
        
        score = combined_refrain_score(line1, line2)
        
        if mode == MatchMode.EXACT:
            normalized1 = normalize_line_for_comparison(line1)
            normalized2 = normalize_line_for_comparison(line2)
            score = 1.0 if normalized1 == normalized2 else 0.0
        
        return VerificationResult(
            score=score,
            passed=score > 0.95,
            constraint_name=self.name,
            details={'line1': line1, 'line2': line2, 'positions': (idx1, idx2)}
        )
    
    def get_dependencies(self) -> List[str]:
        return []


@dataclass
class RefrainGroup(Constraint):
    """A group of positions that must all be identical."""
    positions: List[Position]
    refrain_id: str  # Human-readable identifier like "A1" or "chorus"
    
    @property
    def name(self) -> str:
        return f"refrain_{self.refrain_id}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if len(self.positions) == 2:
            desc1 = self.positions[0].describe()
            desc2 = self.positions[1].describe()
            return f"{desc1.capitalize()} must be repeated as {desc2} (refrain {self.refrain_id})."
        else:
            descs = [p.describe() for p in self.positions]
            positions_str = ", ".join(descs[:-1]) + f", and {descs[-1]}"
            return f"The following lines must all be identical (refrain {self.refrain_id}): {positions_str}."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        try:
            indices = [p.resolve(ctx.structure)[0] for p in self.positions]
            lines = [ctx.structure.lines[i] for i in indices]
        except (ValueError, IndexError) as e:
            return VerificationResult(
                score=0.0, passed=False, constraint_name=self.name,
                details={'error': str(e)}
            )
        
        if len(lines) < 2:
            return VerificationResult(score=1.0, passed=True, constraint_name=self.name)
        
        # Compare all to first line
        reference = lines[0]
        scores = [combined_refrain_score(reference, line) for line in lines[1:]]
        avg_score = sum(scores) / len(scores)
        
        if mode == MatchMode.EXACT:
            avg_score = 1.0 if all(s > 0.99 for s in scores) else 0.0
        
        return VerificationResult(
            score=avg_score,
            passed=all(s > 0.9 for s in scores),
            constraint_name=self.name,
            details={'lines': lines, 'scores': scores}
        )
    
    def get_dependencies(self) -> List[str]:
        return []


@dataclass
class EndWordRotation(Constraint):
    """
    Sestina-style end-word rotation.
    
    Specifies a set of canonical words and their rotation pattern across stanzas.
    """
    word_count: int  # Number of rotating words (6 for sestina)
    rotation_rule: List[int]  # Permutation indices (e.g., [5,0,4,1,3,2])
    stanza_count: int  # How many stanzas use this rotation
    
    @property
    def name(self) -> str:
        return f"endword_rotation_{self.word_count}_{self.stanza_count}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        return (
            f"Choose {self.word_count} distinct end-words for the first stanza. "
            f"These same {self.word_count} words must appear as the end-words of each subsequent stanza, "
            f"but rotated according to the pattern where position i in each new stanza "
            f"takes the word from position {self.rotation_rule} of the previous stanza. "
            f"This rotation continues for {self.stanza_count} stanzas."
        )
    
    def _generate_full_pattern(self) -> List[List[int]]:
        """Generate the complete rotation pattern for all stanzas."""
        pattern = []
        current = list(range(self.word_count))
        
        for _ in range(self.stanza_count):
            pattern.append(current.copy())
            current = [current[i] for i in self.rotation_rule]
        
        return pattern
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        pattern = self._generate_full_pattern()
        
        # Check we have enough stanzas
        if ctx.structure.stanza_count < self.stanza_count:
            return VerificationResult(
                score=0.0, passed=False, constraint_name=self.name,
                details={'error': f'Need {self.stanza_count} stanzas, have {ctx.structure.stanza_count}'}
            )
        
        # Extract end words per stanza
        stanza_endwords = []
        line_idx = 0
        for stanza in ctx.structure.stanzas[:self.stanza_count]:
            if len(stanza) < self.word_count:
                return VerificationResult(
                    score=0.0, passed=False, constraint_name=self.name,
                    details={'error': f'Stanza has {len(stanza)} lines, need {self.word_count}'}
                )
            endwords = ctx.end_words[line_idx:line_idx + self.word_count]
            stanza_endwords.append(endwords)
            line_idx += len(stanza)
        
        # Canonical words from first stanza
        canonical = stanza_endwords[0]
        
        # Check each subsequent stanza
        stanza_scores = []
        for stanza_idx in range(1, self.stanza_count):
            expected_pattern = pattern[stanza_idx]
            actual_words = stanza_endwords[stanza_idx]
            
            position_scores = []
            for pos, expected_word_idx in enumerate(expected_pattern):
                expected_word = canonical[expected_word_idx]
                actual_word = actual_words[pos]
                position_scores.append(word_similarity(expected_word, actual_word))
            
            stanza_scores.append(sum(position_scores) / len(position_scores))
        
        if stanza_scores:
            final_score = sum(stanza_scores) / len(stanza_scores)
        else:
            final_score = 1.0
        
        if mode == MatchMode.EXACT:
            final_score = 1.0 if all(s > 0.95 for s in stanza_scores) else 0.0
        
        return VerificationResult(
            score=final_score,
            passed=final_score > 0.8,
            constraint_name=self.name,
            details={'canonical_words': canonical, 'stanza_scores': stanza_scores}
        )
    
    def get_dependencies(self) -> List[str]:
        return []
```

---

## 5. Composition Operators

python

```python
# ============================================================================
# COMPOSITION OPERATORS
# ============================================================================

@dataclass
class And(Constraint):
    """All sub-constraints must hold."""
    constraints: List[Constraint]
    
    @property
    def name(self) -> str:
        return f"and_{'_'.join(c.name for c in self.constraints)}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        instructions = [c.to_instruction(context) for c in self.constraints]
        return " ".join(instructions)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        results = [c.verify(ctx, mode) for c in self.constraints]
        scores = [r.score for r in results]
        
        if mode == MatchMode.EXACT:
            final_score = 1.0 if all(r.passed for r in results) else 0.0
        else:
            final_score = sum(scores) / len(scores) if scores else 1.0
        
        return VerificationResult(
            score=final_score,
            passed=all(r.passed for r in results),
            constraint_name=self.name,
            sub_results=results
        )
    
    def get_dependencies(self) -> List[str]:
        return list(set(d for c in self.constraints for d in c.get_dependencies()))
    
    def __and__(self, other: Constraint) -> 'And':
        if isinstance(other, And):
            return And(self.constraints + other.constraints)
        return And(self.constraints + [other])


@dataclass
class Or(Constraint):
    """At least one sub-constraint must hold."""
    constraints: List[Constraint]
    
    @property
    def name(self) -> str:
        return f"or_{'_'.join(c.name for c in self.constraints)}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        instructions = [c.to_instruction(context) for c in self.constraints]
        if len(instructions) == 2:
            return f"Either {instructions[0].lower()} Or, alternatively, {instructions[1].lower()}"
        else:
            return "One of the following must hold: " + "; OR ".join(instructions)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        results = [c.verify(ctx, mode) for c in self.constraints]
        scores = [r.score for r in results]
        
        # Take the best score
        final_score = max(scores) if scores else 0.0
        
        return VerificationResult(
            score=final_score,
            passed=any(r.passed for r in results),
            constraint_name=self.name,
            sub_results=results
        )
    
    def get_dependencies(self) -> List[str]:
        return list(set(d for c in self.constraints for d in c.get_dependencies()))


@dataclass
class Weighted(Constraint):
    """Wrap a constraint with explicit weight."""
    constraint: Constraint
    weight: float
    
    @property
    def name(self) -> str:
        return f"weighted_{self.constraint.name}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        base_instruction = self.constraint.to_instruction(context)
        if self.weight > 1.0:
            return f"[IMPORTANT] {base_instruction}"
        elif self.weight < 0.5:
            return f"[Optional] {base_instruction}"
        return base_instruction
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        result = self.constraint.verify(ctx, mode)
        # Weight is applied during aggregation, not here
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            constraint_name=self.name,
            details={'weight': self.weight},
            sub_results=[result]
        )
    
    def get_dependencies(self) -> List[str]:
        return self.constraint.get_dependencies()


@dataclass
class Optional_(Constraint):
    """Constraint is optional - provides bonus points if met."""
    constraint: Constraint
    bonus: float = 0.1  # Extra points if met
    
    @property
    def name(self) -> str:
        return f"optional_{self.constraint.name}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        base = self.constraint.to_instruction(context)
        return f"[Optional, bonus points] {base}"
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        result = self.constraint.verify(ctx, mode)
        # Optional constraints always "pass" but provide bonus
        return VerificationResult(
            score=result.score * self.bonus,
            passed=True,  # Always passes (it's optional)
            constraint_name=self.name,
            details={'bonus': self.bonus, 'underlying_score': result.score},
            sub_results=[result]
        )
    
    def get_dependencies(self) -> List[str]:
        return self.constraint.get_dependencies()


@dataclass  
class Conditional(Constraint):
    """Apply constraint only if condition constraint passes."""
    condition: Constraint
    then_constraint: Constraint
    
    @property
    def name(self) -> str:
        return f"if_{self.condition.name}_then_{self.then_constraint.name}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        cond = self.condition.to_instruction(context)
        then = self.then_constraint.to_instruction(context)
        return f"If {cond.lower()}, then {then.lower()}"
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        cond_result = self.condition.verify(ctx, mode)
        
        if not cond_result.passed:
            # Condition not met, constraint doesn't apply
            return VerificationResult(
                score=1.0,  # Vacuously true
                passed=True,
                constraint_name=self.name,
                details={'condition_met': False}
            )
        
        then_result = self.then_constraint.verify(ctx, mode)
        return VerificationResult(
            score=then_result.score,
            passed=then_result.passed,
            constraint_name=self.name,
            details={'condition_met': True},
            sub_results=[cond_result, then_result]
        )
    
    def get_dependencies(self) -> List[str]:
        return self.condition.get_dependencies() + self.then_constraint.get_dependencies()
```

---

## 6. Form Templates (Known Forms)

python

```python
# ============================================================================
# FORM TEMPLATES - KNOWN POETIC FORMS
# ============================================================================

class FormTemplate:
    """
    A template for a known poetic form.
    
    Combines constraints with metadata and instruction generation.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        constraints: List[Constraint],
        weights: Optional[Dict[str, float]] = None
    ):
        self.form_name = name
        self.description = description
        self.constraints = constraints
        self.weights = weights or {}
    
    def get_constraint_spec(self) -> Constraint:
        """Return the combined constraint specification."""
        if len(self.constraints) == 1:
            return self.constraints[0]
        return And(self.constraints)
    
    def to_full_instruction(self) -> str:
        """Generate complete instruction for this form."""
        lines = [
            f"Write a {self.form_name}.",
            "",
            self.description,
            "",
            "Specific requirements:",
        ]
        for i, c in enumerate(self.constraints, 1):
            lines.append(f"{i}. {c.to_instruction()}")
        
        return "\n".join(lines)
    
    def to_brief_instruction(self) -> str:
        """Generate brief instruction (form name + key points)."""
        return f"Write a {self.form_name}. {self.description}"


# Pre-defined form templates

def create_sestina_template() -> FormTemplate:
    """Create the sestina form template."""
    return FormTemplate(
        name="sestina",
        description=(
            "A sestina is a 39-line poem with six stanzas of six lines each, "
            "followed by a three-line envoi. The same six end-words rotate "
            "through all six stanzas in a specific pattern."
        ),
        constraints=[
            LineCount(39),
            StanzaCount(7),
            StanzaStructure([6, 6, 6, 6, 6, 6, 3]),
            EndWordRotation(
                word_count=6,
                rotation_rule=[5, 0, 4, 1, 3, 2],  # Retrogradatio cruciata
                stanza_count=6
            ),
            # Envoi constraint would go here
        ],
        weights={
            'line_count': 0.1,
            'stanza_count': 0.1,
            'stanza_structure': 0.15,
            'endword_rotation': 0.65
        }
    )


def create_villanelle_template() -> FormTemplate:
    """Create the villanelle form template."""
    
    # Position helpers
    def line(n: int) -> Position:
        return Position(line=n)
    
    return FormTemplate(
        name="villanelle",
        description=(
            "A villanelle is a 19-line poem with five tercets (3-line stanzas) "
            "followed by a quatrain (4-line stanza). It features two refrains "
            "and two repeating rhymes."
        ),
        constraints=[
            LineCount(19),
            StanzaCount(6),
            StanzaStructure([3, 3, 3, 3, 3, 4]),
            RhymeScheme("ABA ABA ABA ABA ABA ABAA"),
            # A1 refrain: lines 1, 6, 12, 18
            RefrainGroup([line(0), line(5), line(11), line(17)], "A1"),
            # A2 refrain: lines 3, 9, 15, 19
            RefrainGroup([line(2), line(8), line(14), line(18)], "A2"),
        ],
        weights={
            'line_count': 0.05,
            'stanza_count': 0.05,
            'stanza_structure': 0.1,
            'rhyme_scheme': 0.3,
            'refrain_A1': 0.25,
            'refrain_A2': 0.25
        }
    )


def create_sonnet_template(variant: str = "shakespearean") -> FormTemplate:
    """Create a sonnet template."""
    
    if variant == "shakespearean":
        return FormTemplate(
            name="Shakespearean sonnet",
            description=(
                "A 14-line poem in iambic pentameter with three quatrains "
                "and a final couplet."
            ),
            constraints=[
                LineCount(14),
                StanzaStructure([4, 4, 4, 2]),  # Or [14] for no stanza breaks
                RhymeScheme("ABAB CDCD EFEF GG"),
            ]
        )
    elif variant == "petrarchan":
        return FormTemplate(
            name="Petrarchan sonnet",
            description=(
                "A 14-line poem with an octave (8 lines) and a sestet (6 lines)."
            ),
            constraints=[
                LineCount(14),
                StanzaStructure([8, 6]),
                RhymeScheme("ABBAABBA CDECDE"),
            ]
        )
    else:
        raise ValueError(f"Unknown sonnet variant: {variant}")


def create_haiku_template() -> FormTemplate:
    """Create haiku template."""
    return FormTemplate(
        name="haiku",
        description="A 3-line poem with a 5-7-5 syllable structure.",
        constraints=[
            LineCount(3),
            StanzaCount(1),
            # Would need syllable counting constraint here
        ]
    )
```

---

## 7. Novel Form Generator

python

```python
# ============================================================================
# NOVEL FORM GENERATOR - For Teaching Generalization
# ============================================================================

import random
from typing import Tuple

class NovelFormGenerator:
    """
    Generates novel poetic forms by composing constraints.
    
    Used for training models to generalize to unseen forms.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        
    def generate_structural_constraints(
        self,
        min_lines: int = 8,
        max_lines: int = 30,
        min_stanzas: int = 2,
        max_stanzas: int = 8
    ) -> Tuple[List[Constraint], Dict]:
        """Generate random but coherent structural constraints."""
        
        num_stanzas = self.rng.randint(min_stanzas, max_stanzas)
        
        # Decide stanza size strategy
        strategy = self.rng.choice(['uniform', 'varied', 'final_different'])
        
        if strategy == 'uniform':
            size = self.rng.randint(2, 6)
            sizes = [size] * num_stanzas
        elif strategy == 'varied':
            sizes = [self.rng.randint(2, 6) for _ in range(num_stanzas)]
        else:  # final_different
            main_size = self.rng.randint(3, 6)
            final_size = self.rng.randint(2, 4)
            sizes = [main_size] * (num_stanzas - 1) + [final_size]
        
        total_lines = sum(sizes)
        
        constraints = [
            LineCount(total_lines),
            StanzaCount(num_stanzas),
            StanzaStructure(sizes)
        ]
        
        metadata = {
            'strategy': strategy,
            'sizes': sizes,
            'total_lines': total_lines
        }
        
        return constraints, metadata
    
    def generate_rhyme_constraints(
        self,
        num_lines: int,
        num_stanzas: int,
        stanza_sizes: List[int]
    ) -> Tuple[List[Constraint], Dict]:
        """Generate random rhyme scheme."""
        
        # Available strategies
        strategy = self.rng.choice([
            'couplets',      # AA BB CC ...
            'alternating',   # ABAB CDCD ...
            'enclosed',      # ABBA CDDC ...
            'terza_rima',    # ABA BCB CDC ...
            'random'         # Random assignment
        ])
        
        if strategy == 'couplets':
            pattern = ''.join(
                chr(65 + i//2) * 2 for i in range(0, num_lines, 2)
            )[:num_lines]
        elif strategy == 'alternating':
            pattern = ''
            label_offset = 0
            for size in stanza_sizes:
                for i in range(size):
                    pattern += chr(65 + label_offset + (i % 2))
                label_offset += 2
        elif strategy == 'enclosed':
            pattern = ''
            label_offset = 0
            for size in stanza_sizes:
                if size == 4:
                    pattern += chr(65 + label_offset)
                    pattern += chr(66 + label_offset)
                    pattern += chr(66 + label_offset)
                    pattern += chr(65 + label_offset)
                else:
                    for i in range(size):
                        pattern += chr(65 + label_offset + (i % 2))
                label_offset += 2
        elif strategy == 'terza_rima':
            pattern = 'ABA'
            label = 1  # B
            for _ in range(1, num_stanzas):
                pattern += chr(65 + label)
                pattern += chr(66 + label)
                pattern += chr(65 + label)
                label += 1
        else:  # random
            num_rhyme_groups = self.rng.randint(2, min(num_lines // 2, 6))
            pattern = ''.join(
                chr(65 + self.rng.randint(0, num_rhyme_groups - 1))
                for _ in range(num_lines)
            )
        
        # Format with spaces between stanzas
        formatted = ''
        idx = 0
        for size in stanza_sizes:
            formatted += pattern[idx:idx + size] + ' '
            idx += size
        formatted = formatted.strip()
        
        constraints = [RhymeScheme(formatted)]
        metadata = {'strategy': strategy, 'pattern': formatted}
        
        return constraints, metadata
    
    def generate_refrain_constraints(
        self,
        num_lines: int,
        stanza_sizes: List[int],
        num_refrains: int = None
    ) -> Tuple[List[Constraint], Dict]:
        """Generate random refrain patterns."""
        
        if num_refrains is None:
            num_refrains = self.rng.randint(0, 2)
        
        if num_refrains == 0:
            return [], {'num_refrains': 0}
        
        constraints = []
        refrain_info = []
        
        # Calculate line offsets for each stanza
        stanza_starts = [0]
        for size in stanza_sizes[:-1]:
            stanza_starts.append(stanza_starts[-1] + size)
        
        for refrain_idx in range(num_refrains):
            # Pick which line position within stanzas
            line_in_stanza = self.rng.choice([0, -1])  # First or last line
            
            # Pick which stanzas include this refrain
            num_occurrences = self.rng.randint(2, min(4, len(stanza_sizes)))
            stanza_indices = sorted(self.rng.sample(range(len(stanza_sizes)), num_occurrences))
            
            # Build positions
            positions = []
            for stanza_idx in stanza_indices:
                if line_in_stanza == -1:
                    # Last line of stanza
                    abs_line = stanza_starts[stanza_idx] + stanza_sizes[stanza_idx] - 1
                else:
                    abs_line = stanza_starts[stanza_idx] + line_in_stanza
                positions.append(Position(line=abs_line))
            
            refrain_id = chr(65 + refrain_idx) + "1"  # A1, B1, etc.
            constraints.append(RefrainGroup(positions, refrain_id))
            refrain_info.append({
                'id': refrain_id,
                'line_in_stanza': line_in_stanza,
                'stanzas': stanza_indices
            })
        
        return constraints, {'num_refrains': num_refrains, 'refrains': refrain_info}
    
    def generate_rotation_constraint(
        self,
        stanza_sizes: List[int]
    ) -> Tuple[List[Constraint], Dict]:
        """Optionally generate end-word rotation (sestina-like)."""
        
        # Only makes sense if stanzas are same size
        if len(set(stanza_sizes)) != 1:
            return [], {'rotation': False}
        
        # 50% chance to include rotation
        if self.rng.random() < 0.5:
            return [], {'rotation': False}
        
        word_count = stanza_sizes[0]
        stanza_count = len(stanza_sizes)
        
        # Generate a random permutation as the rotation rule
        rotation_rule = list(range(word_count))
        self.rng.shuffle(rotation_rule)
        
        constraint = EndWordRotation(
            word_count=word_count,
            rotation_rule=rotation_rule,
            stanza_count=stanza_count
        )
        
        return [constraint], {
            'rotation': True,
            'word_count': word_count,
            'rule': rotation_rule
        }
    
    def generate_form(
        self,
        difficulty: str = 'medium'  # 'easy', 'medium', 'hard'
    ) -> Tuple[FormTemplate, Dict]:
        """
        Generate a complete novel form.
        
        Returns a FormTemplate and metadata about the generation process.
        """
        
        # Difficulty parameters
        if difficulty == 'easy':
            line_range = (4, 12)
            stanza_range = (2, 4)
            max_refrains = 1
            include_rotation = False
        elif difficulty == 'medium':
            line_range = (12, 24)
            stanza_range = (3, 6)
            max_refrains = 2
            include_rotation = self.rng.random() < 0.3
        else:  # hard
            line_range = (20, 40)
            stanza_range = (4, 8)
            max_refrains = 3
            include_rotation = self.rng.random() < 0.5
        
        all_constraints = []
        all_metadata = {'difficulty': difficulty}
        
        # 1. Structural constraints
        struct_constraints, struct_meta = self.generate_structural_constraints(
            min_lines=line_range[0],
            max_lines=line_range[1],
            min_stanzas=stanza_range[0],
            max_stanzas=stanza_range[1]
        )
        all_constraints.extend(struct_constraints)
        all_metadata['structure'] = struct_meta
        
        # 2. Rhyme constraints
        rhyme_constraints, rhyme_meta = self.generate_rhyme_constraints(
            num_lines=struct_meta['total_lines'],
            num_stanzas=len(struct_meta['sizes']),
            stanza_sizes=struct_meta['sizes']
        )
        all_constraints.extend(rhyme_constraints)
        all_metadata['rhyme'] = rhyme_meta
        
        # 3. Refrain constraints
        num_refrains = self.rng.randint(0, max_refrains)
        refrain_constraints, refrain_meta = self.generate_refrain_constraints(
            num_lines=struct_meta['total_lines'],
            stanza_sizes=struct_meta['sizes'],
            num_refrains=num_refrains
        )
        all_constraints.extend(refrain_constraints)
        all_metadata['refrain'] = refrain_meta
        
        # 4. Rotation constraints (optional)
        if include_rotation:
            rotation_constraints, rotation_meta = self.generate_rotation_constraint(
                stanza_sizes=struct_meta['sizes']
            )
            all_constraints.extend(rotation_constraints)
            all_metadata['rotation'] = rotation_meta
        
        # Generate a name for this novel form
        form_name = self._generate_form_name()
        
        # Create template
        template = FormTemplate(
            name=form_name,
            description=f"A novel {difficulty} poetic form with {struct_meta['total_lines']} lines.",
            constraints=all_constraints
        )
        
        return template, all_metadata
    
    def _generate_form_name(self) -> str:
        """Generate a plausible-sounding form name."""
        prefixes = ['neo-', 'pseudo-', 'inverse ', 'modified ', 'spiral ', 'echo ']
        roots = ['sonnet', 'rondel', 'ballade', 'ghazal', 'pantoum', 'triolet', 
                 'canzone', 'envoi', 'ode', 'elegy']
        suffixes = ['ine', 'esque', ' variation', ' form', '']
        
        name = self.rng.choice(prefixes) + self.rng.choice(roots) + self.rng.choice(suffixes)
        return name.strip()
```

---

## 8. Complete Reward Environment with Composition

python

```python
# ============================================================================
# UNIFIED REWARD ENVIRONMENT
# ============================================================================

@dataclass
class CompositeRewardResult:
    """Result from evaluating a poem against composed constraints."""
    total_score: float
    weighted_score: float
    constraint_results: Dict[str, VerificationResult]
    instruction_used: str
    form_name: str
    metadata: Dict[str, Any]


class ComposableRewardEnvironment:
    """
    Reward environment using composable constraints.
    
    Can evaluate against:
    - Known forms (sestina, villanelle, etc.)
    - Novel/generated forms
    - Custom constraint compositions
    """
    
    def __init__(
        self,
        cmu_dict: Optional[Dict] = None,
        default_mode: MatchMode = MatchMode.FUZZY
    ):
        self.cmu_dict = cmu_dict or {}
        self.default_mode = default_mode
        
        # Load known forms
        self.known_forms = {
            'sestina': create_sestina_template(),
            'villanelle': create_villanelle_template(),
            'shakespearean_sonnet': create_sonnet_template('shakespearean'),
            'petrarchan_sonnet': create_sonnet_template('petrarchan'),
        }
        
        self.form_generator = NovelFormGenerator()
    
    def evaluate(
        self,
        poem: str,
        form: Union[str, FormTemplate, Constraint],
        mode: Optional[MatchMode] = None
    ) -> CompositeRewardResult:
        """
        Evaluate a poem against a form specification.
        
        Args:
            poem: Raw poem text
            form: Either a form name, FormTemplate, or raw Constraint
            mode: Match mode (defaults to environment default)
        """
        mode = mode or self.default_mode
        
        # Resolve form to template
        if isinstance(form, str):
            if form not in self.known_forms:
                raise ValueError(f"Unknown form: {form}")
            template = self.known_forms[form]
        elif isinstance(form, FormTemplate):
            template = form
        elif isinstance(form, Constraint):
            template = FormTemplate(
                name="custom",
                description="Custom constraint specification",
                constraints=[form] if not isinstance(form, And) else form.constraints
            )
        else:
            raise TypeError(f"form must be str, FormTemplate, or Constraint, not {type(form)}")
        
        # Build context
        ctx = ConstraintContext.from_poem(poem, self.cmu_dict)
        
        # Evaluate each constraint
        results = {}
        scores = []
        
        for constraint in template.constraints:
            result = constraint.verify(ctx, mode)
            results[constraint.name] = result
            
            # Get weight if specified
            weight = template.weights.get(constraint.name, 1.0)
            scores.append((result.score, weight))
        
        # Compute total and weighted scores
        if scores:
            total_score = sum(s for s, w in scores) / len(scores)
            total_weight = sum(w for s, w in scores)
            weighted_score = sum(s * w for s, w in scores) / total_weight
        else:
            total_score = 1.0
            weighted_score = 1.0
        
        return CompositeRewardResult(
            total_score=total_score,
            weighted_score=weighted_score,
            constraint_results=results,
            instruction_used=template.to_full_instruction(),
            form_name=template.form_name,
            metadata={'mode': mode.name, 'num_constraints': len(template.constraints)}
        )
    
    def generate_training_pair(
        self,
        difficulty: str = 'medium',
        include_instruction: bool = True
    ) -> Tuple[str, Callable[[str], float]]:
        """
        Generate an (instruction, reward_function) pair for training.
        
        Returns:
            instruction: Natural language instruction for the model
            reward_fn: Function that takes a poem string and returns score
        """
        template, metadata = self.form_generator.generate_form(difficulty)
        
        if include_instruction:
            instruction = template.to_full_instruction()
        else:
            instruction = template.to_brief_instruction()
        
        def reward_fn(poem: str) -> float:
            result = self.evaluate(poem, template)
            return result.weighted_score
        
        return instruction, reward_fn
    
    def create_test_suite(
        self,
        num_known: int = 4,
        num_novel_easy: int = 5,
        num_novel_medium: int = 5,
        num_novel_hard: int = 3
    ) -> List[Tuple[str, FormTemplate, Dict]]:
        """
        Create a test suite mixing known and novel forms.
        
        Returns list of (instruction, template, metadata) tuples.
        """
        suite = []
        
        # Known forms
        known_names = list(self.known_forms.keys())[:num_known]
        for name in known_names:
            template = self.known_forms[name]
            suite.append((
                template.to_full_instruction(),
                template,
                {'type': 'known', 'form': name}
            ))
        
        # Novel forms by difficulty
        for difficulty, count in [
            ('easy', num_novel_easy),
            ('medium', num_novel_medium),
            ('hard', num_novel_hard)
        ]:
            for _ in range(count):
                template, meta = self.form_generator.generate_form(difficulty)
                suite.append((
                    template.to_full_instruction(),
                    template,
                    {'type': 'novel', **meta}
                ))
        
        return suite
```

---

## 9. Usage Example: Training Loop

python

```python
def training_example():
    """Example of using the composable system for RL training."""
    
    # Initialize environment
    env = ComposableRewardEnvironment()
    
    # Generate test suite
    test_suite = env.create_test_suite(
        num_known=4,
        num_novel_easy=10,
        num_novel_medium=10,
        num_novel_hard=5
    )
    
    print(f"Generated {len(test_suite)} test cases")
    
    # Example: evaluate a known form
    dylan_thomas = """
Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light.

Though wise men at their end know dark is right,
Because their words had forked no lightning they
Do not go gentle into that good night.

Good men, the last wave by, crying how bright
Their frail deeds might have danced in a green bay,
Rage, rage against the dying of the light.

Wild men who caught and sang the sun in flight,
And learn, too late, they grieved it on its way,
Do not go gentle into that good night.

Grave men, near death, who see with blinding sight
Blind eyes could blaze like meteors and be gay,
Rage, rage against the dying of the light.

And you, my father, there on the sad height,
Curse, bless, me now with your fierce tears, I pray.
Do not go gentle into that good night.
Rage, rage against the dying of the light.
"""
    
    result = env.evaluate(dylan_thomas, 'villanelle')
    print(f"\nVillanelle evaluation:")
    print(f"  Total score: {result.total_score:.3f}")
    print(f"  Weighted score: {result.weighted_score:.3f}")
    print(f"\n  Constraint breakdown:")
    for name, res in result.constraint_results.items():
        print(f"    {name}: {res.score:.3f} ({'PASS' if res.passed else 'FAIL'})")
    
    # Example: generate a novel form and its instruction
    instruction, reward_fn = env.generate_training_pair(difficulty='medium')
    print(f"\n{'='*60}")
    print("NOVEL FORM INSTRUCTION:")
    print(instruction)
    print(f"\n(Reward function ready for evaluating model outputs)")
    
    # The reward_fn can be used directly in an RL loop:
    # score = reward_fn(model_output)


if __name__ == "__main__":
    training_example()
```

---

## 10. Summary: Key Design Principles

|Principle|Implementation|
|---|---|
|**Bidirectionality**|Each constraint generates both instructions AND verification|
|**Composability**|`And`, `Or`, `Weighted`, `Optional`, `Conditional` operators|
|**Fuzzy matching**|`MatchMode` controls exact vs. spectrum rewards|
|**Generalization**|`NovelFormGenerator` creates unseen forms from primitives|
|**Extensibility**|New constraints just implement the `Constraint` interface|
|**Interpretability**|`VerificationResult` provides detailed breakdown|

This system lets you:

1. **Define known forms** as constraint compositions
2. **Generate novel forms** for generalization testing
3. **Create instruction-reward pairs** for RL training
4. **Tune strictness** via `MatchMode` and weights
5. **Debug failures** with detailed constraint breakdowns

we should be able to express all well known forms with our abstraction, and we should be able to specify rewards to have them graded, for instance requiring a title, requiring the title is the first line of the poem in bracketed notation, or to specify the number or words in the title or any given line. Each of which should take an inverse if possible— line must not have 5 syllables, poem must be a sonnet without rhymes and with 15 lines instead of 14, title should be shorter than 5 words— all composable, all with an Auto Rubric

This is exactly the right direction—we need a **complete, invertible constraint algebra** that can express any poetic form (known or absurd) and automatically generate rubrics. Let me build this out comprehensively.

<thinking> The user wants: 1. **Completeness**: Express ALL known poetic forms 2. **Invertibility**: Every constraint has a `Not` version 3. **Granularity**: Word counts, syllable counts, title requirements, bracketed notation, etc. 4. **Auto Rubric**: Automatically generate human-readable grading rubrics from constraints 5. **Composability**: All constraints work together seamlessly

Let me think about the full constraint taxonomy:

## Constraint Categories

### 1. Structural

- LineCount, StanzaCount, StanzaStructure (already have)
- WordCount (per line, per stanza, total)
- SyllableCount (per line, per stanza, total)
- CharacterCount

### 2. Positional/Content

- Title (exists, format, length, position)
- LineContent (exact text, pattern, contains)
- WordAt (specific word at position)
- LineStartsWith, LineEndsWith

### 3. Relational

- Rhymes, RhymeScheme (already have)
- Repeats, RefrainGroup (already have)
- EndWordRotation (already have)
- Alliteration, Assonance
- Acrostic (first letters spell something)

### 4. Phonetic/Prosodic

- Meter (iambic, trochaic, etc.)
- StressPattern
- SyllablePattern (like haiku 5-7-5)

### 5. Semantic/Lexical

- WordFromSet (vocabulary restriction)
- NoWord (banned words)
- UniqueWords (no repetition)
- Sentiment (positive/negative tone)

### 6. Meta/Format

- HasTitle
- TitleFormat (bracketed, first line, separate)
- Indentation
- Punctuation requirements

### 7. Logical Operators

- Not (inversion)
- And, Or (already have)
- Exactly (exactly N of M constraints)
- AtLeast, AtMost
- Implies (if A then B)

## Invertibility

Every constraint C should support:

- `~C` or `Not(C)`: The negation
- Spectrum inversion: If C gives score s, Not(C) gives 1-s? Or different semantics?

For quantitative constraints, inversion is nuanced:

- `LineCount(14)` → `Not(LineCount(14))` means "any line count except 14"
- But we might want `LineCount(15)` or `LineCount(range=...)` instead

Better approach: Constraints have parameters that can express negation naturally:

- `LineCount(exactly=14)` vs `LineCount(not_equal=14)` vs `LineCount(min=10, max=20)`
- `HasTitle(required=True)` vs `HasTitle(required=False)`

Or we use explicit `Not()` wrapper that inverts the score.

## Auto Rubric

Each constraint should generate:

1. A criterion name
2. A description
3. Point value (from weight)
4. Pass/fail criteria
5. Partial credit rules

Let me design this comprehensively. </thinking>

# Complete Invertible Constraint Algebra with Auto-Rubric

## 1. Foundation: Enhanced Type System

python

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List, Dict, Tuple, Optional, Callable, Union, 
    Set, Any, TypeVar, Generic, Literal, Pattern
)
from enum import Enum, auto
import math
import re
import string
from functools import reduce, cached_property

# ============================================================================
# CORE ENUMS AND TYPES
# ============================================================================

class MatchMode(Enum):
    EXACT = auto()      # Binary: 0 or 1
    FUZZY = auto()      # Continuous: [0, 1] with similarity
    SOFT = auto()       # Gaussian decay from target

class ComparisonOp(Enum):
    """Comparison operators for numeric constraints."""
    EQ = "=="           # Equal
    NE = "!="           # Not equal
    LT = "<"            # Less than
    LE = "<="           # Less than or equal
    GT = ">"            # Greater than
    GE = ">="           # Greater than or equal
    IN_RANGE = "in"     # Within range [min, max]
    NOT_IN_RANGE = "not_in"  # Outside range

class PositionType(Enum):
    """Types of position references."""
    ABSOLUTE_LINE = auto()
    STANZA_RELATIVE = auto()
    WORD_IN_LINE = auto()
    CHARACTER = auto()

class TitleFormat(Enum):
    """How the title is formatted/positioned."""
    FIRST_LINE = auto()           # Title is the first line
    BRACKETED_FIRST = auto()      # [Title] on first line
    SEPARATE_HEADER = auto()      # Title above poem, separated by blank line
    INLINE_COLON = auto()         # "Title: first line..."
    NONE = auto()                 # No title

class MeterType(Enum):
    """Common metrical patterns."""
    IAMBIC = "iambic"             # unstressed-STRESSED (da-DUM)
    TROCHAIC = "trochaic"         # STRESSED-unstressed (DUM-da)
    ANAPESTIC = "anapestic"       # unstressed-unstressed-STRESSED (da-da-DUM)
    DACTYLIC = "dactylic"         # STRESSED-unstressed-unstressed (DUM-da-da)
    SPONDAIC = "spondaic"         # STRESSED-STRESSED (DUM-DUM)
    PYRRHIC = "pyrrhic"           # unstressed-unstressed (da-da)
    FREE = "free"                 # No fixed meter

# ============================================================================
# NUMERIC BOUND SPECIFICATION
# ============================================================================

@dataclass(frozen=True)
class NumericBound:
    """
    Flexible numeric constraint specification.
    
    Supports: exactly N, at least N, at most N, between N and M, not N, etc.
    """
    op: ComparisonOp
    value: Optional[int] = None
    min_val: Optional[int] = None
    max_val: Optional[int] = None
    
    @classmethod
    def exactly(cls, n: int) -> 'NumericBound':
        return cls(op=ComparisonOp.EQ, value=n)
    
    @classmethod
    def not_equal(cls, n: int) -> 'NumericBound':
        return cls(op=ComparisonOp.NE, value=n)
    
    @classmethod
    def at_least(cls, n: int) -> 'NumericBound':
        return cls(op=ComparisonOp.GE, value=n)
    
    @classmethod
    def at_most(cls, n: int) -> 'NumericBound':
        return cls(op=ComparisonOp.LE, value=n)
    
    @classmethod
    def less_than(cls, n: int) -> 'NumericBound':
        return cls(op=ComparisonOp.LT, value=n)
    
    @classmethod
    def greater_than(cls, n: int) -> 'NumericBound':
        return cls(op=ComparisonOp.GT, value=n)
    
    @classmethod
    def between(cls, min_val: int, max_val: int) -> 'NumericBound':
        return cls(op=ComparisonOp.IN_RANGE, min_val=min_val, max_val=max_val)
    
    @classmethod
    def outside(cls, min_val: int, max_val: int) -> 'NumericBound':
        return cls(op=ComparisonOp.NOT_IN_RANGE, min_val=min_val, max_val=max_val)
    
    def check(self, actual: int) -> bool:
        """Check if actual value satisfies this bound."""
        if self.op == ComparisonOp.EQ:
            return actual == self.value
        elif self.op == ComparisonOp.NE:
            return actual != self.value
        elif self.op == ComparisonOp.LT:
            return actual < self.value
        elif self.op == ComparisonOp.LE:
            return actual <= self.value
        elif self.op == ComparisonOp.GT:
            return actual > self.value
        elif self.op == ComparisonOp.GE:
            return actual >= self.value
        elif self.op == ComparisonOp.IN_RANGE:
            return self.min_val <= actual <= self.max_val
        elif self.op == ComparisonOp.NOT_IN_RANGE:
            return actual < self.min_val or actual > self.max_val
        return False
    
    def score(self, actual: int, mode: MatchMode = MatchMode.FUZZY) -> float:
        """
        Compute score for actual value against this bound.
        
        For EXACT mode: binary 0/1
        For FUZZY/SOFT mode: continuous based on distance from target
        """
        if mode == MatchMode.EXACT:
            return 1.0 if self.check(actual) else 0.0
        
        # Fuzzy scoring
        if self.op == ComparisonOp.EQ:
            diff = abs(actual - self.value)
            return math.exp(-0.5 * (diff / max(1, self.value * 0.1)) ** 2)
        
        elif self.op == ComparisonOp.NE:
            if actual == self.value:
                return 0.0
            diff = abs(actual - self.value)
            return min(1.0, diff / max(1, self.value * 0.2))
        
        elif self.op in (ComparisonOp.LT, ComparisonOp.LE):
            target = self.value if self.op == ComparisonOp.LE else self.value - 1
            if actual <= target:
                return 1.0
            overage = actual - target
            return math.exp(-0.3 * overage)
        
        elif self.op in (ComparisonOp.GT, ComparisonOp.GE):
            target = self.value if self.op == ComparisonOp.GE else self.value + 1
            if actual >= target:
                return 1.0
            underage = target - actual
            return math.exp(-0.3 * underage)
        
        elif self.op == ComparisonOp.IN_RANGE:
            if self.min_val <= actual <= self.max_val:
                return 1.0
            if actual < self.min_val:
                diff = self.min_val - actual
            else:
                diff = actual - self.max_val
            return math.exp(-0.3 * diff)
        
        elif self.op == ComparisonOp.NOT_IN_RANGE:
            if actual < self.min_val or actual > self.max_val:
                return 1.0
            # How far into the forbidden range?
            center = (self.min_val + self.max_val) / 2
            half_width = (self.max_val - self.min_val) / 2
            dist_from_center = abs(actual - center)
            return dist_from_center / half_width if half_width > 0 else 0.0
        
        return 0.0
    
    def invert(self) -> 'NumericBound':
        """Return the logical inverse of this bound."""
        inversions = {
            ComparisonOp.EQ: ComparisonOp.NE,
            ComparisonOp.NE: ComparisonOp.EQ,
            ComparisonOp.LT: ComparisonOp.GE,
            ComparisonOp.LE: ComparisonOp.GT,
            ComparisonOp.GT: ComparisonOp.LE,
            ComparisonOp.GE: ComparisonOp.LT,
            ComparisonOp.IN_RANGE: ComparisonOp.NOT_IN_RANGE,
            ComparisonOp.NOT_IN_RANGE: ComparisonOp.IN_RANGE,
        }
        return NumericBound(
            op=inversions[self.op],
            value=self.value,
            min_val=self.min_val,
            max_val=self.max_val
        )
    
    def describe(self) -> str:
        """Human-readable description."""
        if self.op == ComparisonOp.EQ:
            return f"exactly {self.value}"
        elif self.op == ComparisonOp.NE:
            return f"not {self.value}"
        elif self.op == ComparisonOp.LT:
            return f"fewer than {self.value}"
        elif self.op == ComparisonOp.LE:
            return f"at most {self.value}"
        elif self.op == ComparisonOp.GT:
            return f"more than {self.value}"
        elif self.op == ComparisonOp.GE:
            return f"at least {self.value}"
        elif self.op == ComparisonOp.IN_RANGE:
            return f"between {self.min_val} and {self.max_val}"
        elif self.op == ComparisonOp.NOT_IN_RANGE:
            return f"not between {self.min_val} and {self.max_val}"
        return "unknown"
```

---

## 2. Auto-Rubric System

python

```python
# ============================================================================
# AUTO-RUBRIC SYSTEM
# ============================================================================

@dataclass
class RubricCriterion:
    """A single criterion in the grading rubric."""
    name: str
    description: str
    max_points: float
    passing_threshold: float  # Score needed to "pass" this criterion
    partial_credit: bool      # Whether partial credit is awarded
    grading_notes: str        # Additional grading guidance
    
    def format_markdown(self) -> str:
        """Format as markdown for display."""
        lines = [
            f"### {self.name} ({self.max_points} points)",
            "",
            self.description,
            "",
            f"**Passing threshold:** {self.passing_threshold * 100:.0f}%",
            f"**Partial credit:** {'Yes' if self.partial_credit else 'No'}",
        ]
        if self.grading_notes:
            lines.extend(["", f"*{self.grading_notes}*"])
        return "\n".join(lines)


@dataclass
class Rubric:
    """Complete grading rubric generated from constraints."""
    title: str
    description: str
    criteria: List[RubricCriterion]
    total_points: float
    passing_score: float  # Minimum total to pass
    
    @classmethod
    def from_constraints(
        cls,
        constraints: List['Constraint'],
        weights: Optional[Dict[str, float]] = None,
        total_points: float = 100.0,
        title: str = "Poetry Grading Rubric"
    ) -> 'Rubric':
        """Generate rubric from a list of constraints."""
        weights = weights or {}
        
        # Calculate weight sum for normalization
        raw_weights = []
        for c in constraints:
            w = weights.get(c.name, 1.0)
            raw_weights.append((c, w))
        
        total_weight = sum(w for _, w in raw_weights)
        
        # Generate criteria
        criteria = []
        for constraint, weight in raw_weights:
            normalized_weight = weight / total_weight
            points = total_points * normalized_weight
            
            criterion = RubricCriterion(
                name=constraint.rubric_name(),
                description=constraint.rubric_description(),
                max_points=round(points, 1),
                passing_threshold=constraint.passing_threshold(),
                partial_credit=constraint.allows_partial_credit(),
                grading_notes=constraint.grading_notes()
            )
            criteria.append(criterion)
        
        return cls(
            title=title,
            description=f"Auto-generated rubric with {len(criteria)} criteria",
            criteria=criteria,
            total_points=total_points,
            passing_score=total_points * 0.7  # Default 70% to pass
        )
    
    def format_markdown(self) -> str:
        """Format complete rubric as markdown."""
        lines = [
            f"# {self.title}",
            "",
            self.description,
            "",
            f"**Total points:** {self.total_points}",
            f"**Passing score:** {self.passing_score}",
            "",
            "---",
            ""
        ]
        
        for criterion in self.criteria:
            lines.append(criterion.format_markdown())
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def grade(self, results: Dict[str, 'VerificationResult']) -> 'GradeReport':
        """Grade a poem using verification results."""
        earned_points = 0.0
        criterion_grades = []
        
        for criterion in self.criteria:
            # Find matching result
            result = None
            for name, r in results.items():
                if criterion.name in name or name in criterion.name:
                    result = r
                    break
            
            if result:
                points = criterion.max_points * result.score
                passed = result.score >= criterion.passing_threshold
            else:
                points = 0.0
                passed = False
            
            earned_points += points
            criterion_grades.append({
                'criterion': criterion.name,
                'max_points': criterion.max_points,
                'earned_points': round(points, 1),
                'percentage': result.score if result else 0.0,
                'passed': passed
            })
        
        return GradeReport(
            total_earned=round(earned_points, 1),
            total_possible=self.total_points,
            percentage=earned_points / self.total_points,
            passed=earned_points >= self.passing_score,
            criterion_grades=criterion_grades
        )


@dataclass
class GradeReport:
    """Report from grading a poem."""
    total_earned: float
    total_possible: float
    percentage: float
    passed: bool
    criterion_grades: List[Dict[str, Any]]
    
    def format_markdown(self) -> str:
        """Format grade report as markdown."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        
        lines = [
            f"# Grade Report",
            "",
            f"**Score:** {self.total_earned}/{self.total_possible} ({self.percentage*100:.1f}%)",
            f"**Status:** {status}",
            "",
            "## Criterion Breakdown",
            "",
            "| Criterion | Earned | Max | % | Status |",
            "|-----------|--------|-----|---|--------|",
        ]
        
        for cg in self.criterion_grades:
            status_emoji = "✅" if cg['passed'] else "❌"
            lines.append(
                f"| {cg['criterion']} | {cg['earned_points']} | {cg['max_points']} | "
                f"{cg['percentage']*100:.0f}% | {status_emoji} |"
            )
        
        return "\n".join(lines)
```

---

## 3. Enhanced Base Constraint with Rubric Support

python

```python
# ============================================================================
# ENHANCED BASE CONSTRAINT
# ============================================================================

class Constraint(ABC):
    """
    Base class for all poetic constraints.
    
    Every constraint is:
    - Invertible (supports Not/negation)
    - Self-describing (generates instructions AND rubric criteria)
    - Scorable (produces continuous rewards)
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # CORE ABSTRACT METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this constraint."""
        pass
    
    @abstractmethod
    def verify(
        self, 
        ctx: 'ConstraintContext',
        mode: MatchMode = MatchMode.FUZZY
    ) -> 'VerificationResult':
        """Verify this constraint and return score."""
        pass
    
    @abstractmethod
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        """Generate natural language instruction."""
        pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # RUBRIC METHODS (with defaults)
    # ─────────────────────────────────────────────────────────────────────────
    
    def rubric_name(self) -> str:
        """Name for rubric criterion. Override for custom names."""
        return self.name.replace('_', ' ').title()
    
    def rubric_description(self) -> str:
        """Description for rubric. Defaults to instruction."""
        return self.to_instruction()
    
    def passing_threshold(self) -> float:
        """Score threshold to pass this criterion. Default 0.7."""
        return 0.7
    
    def allows_partial_credit(self) -> bool:
        """Whether partial credit is possible. Default True."""
        return True
    
    def grading_notes(self) -> str:
        """Additional notes for graders. Default empty."""
        return ""
    
    # ─────────────────────────────────────────────────────────────────────────
    # INVERSION
    # ─────────────────────────────────────────────────────────────────────────
    
    def invert(self) -> 'Constraint':
        """
        Return the logical negation of this constraint.
        
        Default implementation wraps in Not(). Override for
        constraints with natural inversions.
        """
        return Not(self)
    
    def __invert__(self) -> 'Constraint':
        """Support ~constraint syntax for negation."""
        return self.invert()
    
    # ─────────────────────────────────────────────────────────────────────────
    # COMPOSITION OPERATORS
    # ─────────────────────────────────────────────────────────────────────────
    
    def __and__(self, other: 'Constraint') -> 'And':
        return And([self, other])
    
    def __or__(self, other: 'Constraint') -> 'Or':
        return Or([self, other])
    
    def weighted(self, weight: float) -> 'Weighted':
        return Weighted(self, weight)
    
    def optional(self, bonus: float = 0.1) -> 'Optional_':
        return Optional_(self, bonus)
    
    # ─────────────────────────────────────────────────────────────────────────
    # DEPENDENCIES
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_dependencies(self) -> List[str]:
        """Return names of constraints this depends on."""
        return []
```

---

## 4. Complete Structural Constraints

python

```python
# ============================================================================
# STRUCTURAL CONSTRAINTS (COMPLETE)
# ============================================================================

@dataclass
class LineCount(Constraint):
    """Constraint on total number of lines."""
    bound: NumericBound
    
    @classmethod
    def exactly(cls, n: int) -> 'LineCount':
        return cls(NumericBound.exactly(n))
    
    @classmethod
    def at_least(cls, n: int) -> 'LineCount':
        return cls(NumericBound.at_least(n))
    
    @classmethod
    def at_most(cls, n: int) -> 'LineCount':
        return cls(NumericBound.at_most(n))
    
    @classmethod
    def between(cls, min_n: int, max_n: int) -> 'LineCount':
        return cls(NumericBound.between(min_n, max_n))
    
    @classmethod
    def not_equal(cls, n: int) -> 'LineCount':
        return cls(NumericBound.not_equal(n))
    
    @property
    def name(self) -> str:
        return f"line_count_{self.bound.describe().replace(' ', '_')}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        return f"The poem must have {self.bound.describe()} lines."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        actual = ctx.structure.line_count
        score = self.bound.score(actual, mode)
        passed = self.bound.check(actual)
        
        return VerificationResult(
            score=score,
            passed=passed,
            constraint_name=self.name,
            details={'expected': self.bound.describe(), 'actual': actual}
        )
    
    def invert(self) -> 'LineCount':
        return LineCount(self.bound.invert())
    
    def rubric_name(self) -> str:
        return "Line Count"
    
    def grading_notes(self) -> str:
        return "Count all non-empty lines, excluding title if separate."


@dataclass
class WordCount(Constraint):
    """Constraint on word count (per line, per stanza, or total)."""
    bound: NumericBound
    scope: Literal['line', 'stanza', 'total'] = 'total'
    target_index: Optional[int] = None  # Which line/stanza (None = all/total)
    
    @classmethod
    def total_exactly(cls, n: int) -> 'WordCount':
        return cls(NumericBound.exactly(n), scope='total')
    
    @classmethod
    def per_line_exactly(cls, n: int, line: Optional[int] = None) -> 'WordCount':
        return cls(NumericBound.exactly(n), scope='line', target_index=line)
    
    @classmethod
    def per_line_at_most(cls, n: int, line: Optional[int] = None) -> 'WordCount':
        return cls(NumericBound.at_most(n), scope='line', target_index=line)
    
    @classmethod
    def per_line_at_least(cls, n: int, line: Optional[int] = None) -> 'WordCount':
        return cls(NumericBound.at_least(n), scope='line', target_index=line)
    
    @property
    def name(self) -> str:
        scope_str = f"{self.scope}"
        if self.target_index is not None:
            scope_str += f"_{self.target_index}"
        return f"word_count_{scope_str}_{self.bound.describe().replace(' ', '_')}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        bound_desc = self.bound.describe()
        
        if self.scope == 'total':
            return f"The poem must have {bound_desc} words in total."
        elif self.scope == 'line':
            if self.target_index is not None:
                return f"Line {self.target_index + 1} must have {bound_desc} words."
            else:
                return f"Each line must have {bound_desc} words."
        elif self.scope == 'stanza':
            if self.target_index is not None:
                return f"Stanza {self.target_index + 1} must have {bound_desc} words."
            else:
                return f"Each stanza must have {bound_desc} words."
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        if self.scope == 'total':
            total = sum(self._count_words(line) for line in ctx.structure.lines)
            score = self.bound.score(total, mode)
            passed = self.bound.check(total)
            details = {'total_words': total}
        
        elif self.scope == 'line':
            if self.target_index is not None:
                # Specific line
                if self.target_index >= len(ctx.structure.lines):
                    return VerificationResult(
                        score=0.0, passed=False, constraint_name=self.name,
                        details={'error': f'Line {self.target_index} does not exist'}
                    )
                count = self._count_words(ctx.structure.lines[self.target_index])
                score = self.bound.score(count, mode)
                passed = self.bound.check(count)
                details = {'line': self.target_index, 'words': count}
            else:
                # All lines
                scores = []
                passes = []
                line_counts = []
                for line in ctx.structure.lines:
                    count = self._count_words(line)
                    line_counts.append(count)
                    scores.append(self.bound.score(count, mode))
                    passes.append(self.bound.check(count))
                score = sum(scores) / len(scores) if scores else 1.0
                passed = all(passes)
                details = {'line_word_counts': line_counts}
        
        elif self.scope == 'stanza':
            if self.target_index is not None:
                if self.target_index >= len(ctx.structure.stanzas):
                    return VerificationResult(
                        score=0.0, passed=False, constraint_name=self.name,
                        details={'error': f'Stanza {self.target_index} does not exist'}
                    )
                stanza = ctx.structure.stanzas[self.target_index]
                count = sum(self._count_words(line) for line in stanza)
                score = self.bound.score(count, mode)
                passed = self.bound.check(count)
                details = {'stanza': self.target_index, 'words': count}
            else:
                scores = []
                passes = []
                stanza_counts = []
                for stanza in ctx.structure.stanzas:
                    count = sum(self._count_words(line) for line in stanza)
                    stanza_counts.append(count)
                    scores.append(self.bound.score(count, mode))
                    passes.append(self.bound.check(count))
                score = sum(scores) / len(scores) if scores else 1.0
                passed = all(passes)
                details = {'stanza_word_counts': stanza_counts}
        
        return VerificationResult(
            score=score, passed=passed, constraint_name=self.name, details=details
        )
    
    def invert(self) -> 'WordCount':
        return WordCount(self.bound.invert(), self.scope, self.target_index)


@dataclass
class SyllableCount(Constraint):
    """Constraint on syllable count."""
    bound: NumericBound
    scope: Literal['line', 'stanza', 'total'] = 'line'
    target_index: Optional[int] = None
    
    # Syllable counting is complex - we'll use a simple heuristic
    # In production, use CMU dict or syllable library
    
    @classmethod
    def per_line_exactly(cls, n: int, line: Optional[int] = None) -> 'SyllableCount':
        return cls(NumericBound.exactly(n), scope='line', target_index=line)
    
    @classmethod
    def per_line_pattern(cls, pattern: List[int]) -> 'SyllablePattern':
        """Create a pattern like [5, 7, 5] for haiku."""
        return SyllablePattern(pattern)
    
    @property
    def name(self) -> str:
        scope_str = f"{self.scope}"
        if self.target_index is not None:
            scope_str += f"_{self.target_index}"
        return f"syllable_count_{scope_str}_{self.bound.describe().replace(' ', '_')}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        bound_desc = self.bound.describe()
        
        if self.scope == 'line':
            if self.target_index is not None:
                return f"Line {self.target_index + 1} must have {bound_desc} syllables."
            else:
                return f"Each line must have {bound_desc} syllables."
        elif self.scope == 'total':
            return f"The poem must have {bound_desc} syllables in total."
        return f"Must have {bound_desc} syllables."
    
    def _count_syllables(self, word: str, cmu_dict: dict) -> int:
        """Count syllables in a word using CMU dict or heuristic."""
        word = word.lower().strip(string.punctuation)
        
        if word in cmu_dict:
            # Count vowel phonemes (they have digits for stress)
            phonemes = cmu_dict[word][0]
            return sum(1 for p in phonemes if any(c.isdigit() for c in p))
        
        # Fallback heuristic
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
    
    def _count_line_syllables(self, line: str, cmu_dict: dict) -> int:
        """Count syllables in a line."""
        words = re.findall(r"[a-zA-Z']+", line)
        return sum(self._count_syllables(w, cmu_dict) for w in words)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        cmu = ctx.cmu_dict
        
        if self.scope == 'line':
            if self.target_index is not None:
                if self.target_index >= len(ctx.structure.lines):
                    return VerificationResult(
                        score=0.0, passed=False, constraint_name=self.name,
                        details={'error': f'Line {self.target_index} does not exist'}
                    )
                count = self._count_line_syllables(ctx.structure.lines[self.target_index], cmu)
                score = self.bound.score(count, mode)
                passed = self.bound.check(count)
                details = {'line': self.target_index, 'syllables': count}
            else:
                scores, passes, counts = [], [], []
                for line in ctx.structure.lines:
                    count = self._count_line_syllables(line, cmu)
                    counts.append(count)
                    scores.append(self.bound.score(count, mode))
                    passes.append(self.bound.check(count))
                score = sum(scores) / len(scores) if scores else 1.0
                passed = all(passes)
                details = {'line_syllable_counts': counts}
        
        elif self.scope == 'total':
            total = sum(
                self._count_line_syllables(line, cmu) 
                for line in ctx.structure.lines
            )
            score = self.bound.score(total, mode)
            passed = self.bound.check(total)
            details = {'total_syllables': total}
        
        return VerificationResult(
            score=score, passed=passed, constraint_name=self.name, details=details
        )
    
    def invert(self) -> 'SyllableCount':
        return SyllableCount(self.bound.invert(), self.scope, self.target_index)


@dataclass
class SyllablePattern(Constraint):
    """Syllable pattern across lines (e.g., haiku 5-7-5)."""
    pattern: List[int]
    tolerance: int = 0  # Allowed deviation per line
    
    @property
    def name(self) -> str:
        return f"syllable_pattern_{'_'.join(map(str, self.pattern))}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        pattern_str = '-'.join(map(str, self.pattern))
        if self.tolerance > 0:
            return f"The poem must follow a {pattern_str} syllable pattern (±{self.tolerance} per line)."
        return f"The poem must follow a {pattern_str} syllable pattern."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        if len(ctx.structure.lines) != len(self.pattern):
            # Wrong number of lines
            line_penalty = abs(len(ctx.structure.lines) - len(self.pattern)) * 0.2
            return VerificationResult(
                score=max(0, 1.0 - line_penalty),
                passed=False,
                constraint_name=self.name,
                details={
                    'error': f'Expected {len(self.pattern)} lines, got {len(ctx.structure.lines)}'
                }
            )
        
        # Count syllables and compare
        cmu = ctx.cmu_dict
        scores = []
        details_per_line = []
        
        for i, (line, expected) in enumerate(zip(ctx.structure.lines, self.pattern)):
            # Reuse syllable counting logic
            words = re.findall(r"[a-zA-Z']+", line)
            actual = sum(self._count_syllables(w, cmu) for w in words)
            
            diff = abs(actual - expected)
            if diff <= self.tolerance:
                line_score = 1.0
            else:
                line_score = math.exp(-0.3 * (diff - self.tolerance))
            
            scores.append(line_score)
            details_per_line.append({
                'line': i, 'expected': expected, 'actual': actual, 'diff': diff
            })
        
        avg_score = sum(scores) / len(scores)
        all_passed = all(
            d['diff'] <= self.tolerance for d in details_per_line
        )
        
        return VerificationResult(
            score=avg_score,
            passed=all_passed,
            constraint_name=self.name,
            details={'pattern': self.pattern, 'lines': details_per_line}
        )
    
    def _count_syllables(self, word: str, cmu_dict: dict) -> int:
        """Count syllables using CMU dict or heuristic."""
        word = word.lower().strip(string.punctuation)
        
        if word in cmu_dict:
            phonemes = cmu_dict[word][0]
            return sum(1 for p in phonemes if any(c.isdigit() for c in p))
        
        # Heuristic fallback
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)
    
    def invert(self) -> 'Not':
        # Pattern inversion doesn't have a natural form, use Not wrapper
        return Not(self)
    
    def rubric_name(self) -> str:
        return f"Syllable Pattern ({'-'.join(map(str, self.pattern))})"
```

---

## 5. Title and Formatting Constraints

python

```python
# ============================================================================
# TITLE AND FORMATTING CONSTRAINTS
# ============================================================================

@dataclass
class HasTitle(Constraint):
    """Poem must (or must not) have a title."""
    required: bool = True
    format: TitleFormat = TitleFormat.FIRST_LINE
    
    @property
    def name(self) -> str:
        req = "required" if self.required else "forbidden"
        return f"title_{req}_{self.format.name.lower()}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if not self.required:
            return "The poem must NOT have a title."
        
        format_desc = {
            TitleFormat.FIRST_LINE: "The first line should be the title.",
            TitleFormat.BRACKETED_FIRST: "The title should appear in [brackets] on the first line.",
            TitleFormat.SEPARATE_HEADER: "The title should appear above the poem, separated by a blank line.",
            TitleFormat.INLINE_COLON: "The title should precede the first line, followed by a colon.",
        }
        return format_desc.get(self.format, "The poem must have a title.")
    
    def _extract_title(self, ctx: ConstraintContext) -> Tuple[Optional[str], float]:
        """
        Extract title based on format.
        Returns (title, confidence).
        """
        if not ctx.structure.lines:
            return None, 0.0
        
        first_line = ctx.structure.lines[0]
        
        if self.format == TitleFormat.BRACKETED_FIRST:
            match = re.match(r'^\[(.+)\]$', first_line.strip())
            if match:
                return match.group(1), 1.0
            # Partial match - has brackets but not exclusively
            if '[' in first_line and ']' in first_line:
                bracket_content = re.search(r'\[(.+?)\]', first_line)
                if bracket_content:
                    return bracket_content.group(1), 0.7
            return None, 0.0
        
        elif self.format == TitleFormat.FIRST_LINE:
            # Heuristic: title-like if short, capitalized, no end punctuation
            line = first_line.strip()
            is_title_like = (
                len(line.split()) <= 10 and
                line[0].isupper() if line else False and
                not line.endswith(('.', ',', ';', ':', '!', '?'))
            )
            return line if is_title_like else None, 0.8 if is_title_like else 0.3
        
        elif self.format == TitleFormat.INLINE_COLON:
            if ':' in first_line:
                parts = first_line.split(':', 1)
                return parts[0].strip(), 0.9
            return None, 0.0
        
        elif self.format == TitleFormat.SEPARATE_HEADER:
            # Would need to check original text for blank line separator
            # For now, assume first line is title if followed by stanza break
            if len(ctx.structure.stanzas) > 1 and len(ctx.structure.stanzas[0]) == 1:
                return first_line, 0.9
            return None, 0.3
        
        return None, 0.0
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        title, confidence = self._extract_title(ctx)
        has_title = title is not None and confidence > 0.5
        
        if self.required:
            score = confidence if has_title else 0.0
            passed = has_title
        else:
            score = 1.0 - confidence if not has_title else 0.0
            passed = not has_title
        
        return VerificationResult(
            score=score,
            passed=passed,
            constraint_name=self.name,
            details={
                'title': title,
                'confidence': confidence,
                'format': self.format.name
            }
        )
    
    def invert(self) -> 'HasTitle':
        return HasTitle(required=not self.required, format=self.format)
    
    def rubric_name(self) -> str:
        return "Title" if self.required else "No Title"


@dataclass
class TitleWordCount(Constraint):
    """Constraint on number of words in title."""
    bound: NumericBound
    title_format: TitleFormat = TitleFormat.FIRST_LINE
    
    @classmethod
    def at_most(cls, n: int, fmt: TitleFormat = TitleFormat.FIRST_LINE) -> 'TitleWordCount':
        return cls(NumericBound.at_most(n), fmt)
    
    @classmethod
    def exactly(cls, n: int, fmt: TitleFormat = TitleFormat.FIRST_LINE) -> 'TitleWordCount':
        return cls(NumericBound.exactly(n), fmt)
    
    @property
    def name(self) -> str:
        return f"title_word_count_{self.bound.describe().replace(' ', '_')}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        return f"The title must have {self.bound.describe()} words."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        # First extract title
        title_constraint = HasTitle(required=True, format=self.title_format)
        title, confidence = title_constraint._extract_title(ctx)
        
        if not title or confidence < 0.5:
            return VerificationResult(
                score=0.0,
                passed=False,
                constraint_name=self.name,
                details={'error': 'No title found'}
            )
        
        word_count = len(title.split())
        score = self.bound.score(word_count, mode)
        passed = self.bound.check(word_count)
        
        return VerificationResult(
            score=score,
            passed=passed,
            constraint_name=self.name,
            details={'title': title, 'word_count': word_count}
        )
    
    def invert(self) -> 'TitleWordCount':
        return TitleWordCount(self.bound.invert(), self.title_format)


@dataclass
class LineStartsWith(Constraint):
    """Line must start with specific text or pattern."""
    target: Union[str, 're.Pattern']
    line_index: Optional[int] = None  # None = all lines
    case_sensitive: bool = False
    
    @property
    def name(self) -> str:
        target_str = self.target if isinstance(self.target, str) else "pattern"
        line_str = f"line_{self.line_index}" if self.line_index is not None else "all_lines"
        return f"starts_with_{target_str}_{line_str}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        target_desc = f'"{self.target}"' if isinstance(self.target, str) else "the specified pattern"
        
        if self.line_index is not None:
            return f"Line {self.line_index + 1} must start with {target_desc}."
        return f"Each line must start with {target_desc}."
    
    def _check_line(self, line: str) -> Tuple[bool, float]:
        """Check if line starts with target. Returns (match, score)."""
        if isinstance(self.target, str):
            target = self.target if self.case_sensitive else self.target.lower()
            line_start = line if self.case_sensitive else line.lower()
            
            if line_start.startswith(target):
                return True, 1.0
            
            # Fuzzy: how close is the start?
            target_len = len(target)
            actual_start = line_start[:target_len]
            similarity = sum(a == b for a, b in zip(actual_start, target)) / target_len
            return False, similarity * 0.5
        else:
            # Regex pattern
            match = self.target.match(line)
            return match is not None, 1.0 if match else 0.0
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        if self.line_index is not None:
            if self.line_index >= len(ctx.structure.lines):
                return VerificationResult(
                    score=0.0, passed=False, constraint_name=self.name,
                    details={'error': f'Line {self.line_index} does not exist'}
                )
            matched, score = self._check_line(ctx.structure.lines[self.line_index])
            if mode == MatchMode.EXACT:
                score = 1.0 if matched else 0.0
            return VerificationResult(
                score=score, passed=matched, constraint_name=self.name,
                details={'line': self.line_index, 'matched': matched}
            )
        
        # All lines
        scores = []
        matches = []
        for line in ctx.structure.lines:
            matched, score = self._check_line(line)
            scores.append(score if mode != MatchMode.EXACT else (1.0 if matched else 0.0))
            matches.append(matched)
        
        avg_score = sum(scores) / len(scores) if scores else 1.0
        all_passed = all(matches)
        
        return VerificationResult(
            score=avg_score, passed=all_passed, constraint_name=self.name,
            details={'matches': matches, 'scores': scores}
        )
    
    def invert(self) -> 'LineDoesNotStartWith':
        return LineDoesNotStartWith(self.target, self.line_index, self.case_sensitive)


@dataclass
class LineDoesNotStartWith(Constraint):
    """Line must NOT start with specific text or pattern."""
    target: Union[str, 're.Pattern']
    line_index: Optional[int] = None
    case_sensitive: bool = False
    
    @property
    def name(self) -> str:
        target_str = self.target if isinstance(self.target, str) else "pattern"
        line_str = f"line_{self.line_index}" if self.line_index is not None else "all_lines"
        return f"not_starts_with_{target_str}_{line_str}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        target_desc = f'"{self.target}"' if isinstance(self.target, str) else "the specified pattern"
        
        if self.line_index is not None:
            return f"Line {self.line_index + 1} must NOT start with {target_desc}."
        return f"No line should start with {target_desc}."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        # Delegate to positive version and invert
        positive = LineStartsWith(self.target, self.line_index, self.case_sensitive)
        result = positive.verify(ctx, mode)
        
        return VerificationResult(
            score=1.0 - result.score,
            passed=not result.passed,
            constraint_name=self.name,
            details=result.details
        )
    
    def invert(self) -> 'LineStartsWith':
        return LineStartsWith(self.target, self.line_index, self.case_sensitive)
```

---

## 6. Acrostic and Pattern Constraints

python

```python
# ============================================================================
# ACROSTIC AND LETTER PATTERN CONSTRAINTS
# ============================================================================

@dataclass
class Acrostic(Constraint):
    """
    First letters of lines spell out a word or phrase.
    
    Can specify:
    - Exact word to spell
    - Pattern (regex)
    - Length constraint
    """
    target: Optional[str] = None  # Word to spell (None = any valid word)
    case_sensitive: bool = False
    position: Literal['first', 'last'] = 'first'  # First or last letter of each line
    
    @classmethod
    def spelling(cls, word: str) -> 'Acrostic':
        return cls(target=word)
    
    @classmethod
    def any_word(cls) -> 'Acrostic':
        return cls(target=None)
    
    @property
    def name(self) -> str:
        if self.target:
            return f"acrostic_{self.target.lower()}"
        return f"acrostic_{self.position}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        pos_desc = "first" if self.position == 'first' else "last"
        
        if self.target:
            return f'The {pos_desc} letter of each line must spell out "{self.target}".'
        return f"The {pos_desc} letters of each line should spell out a word or phrase."
    
    def _extract_letters(self, ctx: ConstraintContext) -> str:
        """Extract acrostic letters from lines."""
        letters = []
        for line in ctx.structure.lines:
            # Get first word
            words = line.split()
            if not words:
                letters.append('')
                continue
            
            word = words[0] if self.position == 'first' else words[-1]
            # Strip punctuation
            clean = ''.join(c for c in word if c.isalpha())
            
            if clean:
                letter = clean[0] if self.position == 'first' else clean[-1]
                letters.append(letter)
            else:
                letters.append('')
        
        return ''.join(letters)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        extracted = self._extract_letters(ctx)
        
        if self.target:
            target = self.target if self.case_sensitive else self.target.lower()
            actual = extracted if self.case_sensitive else extracted.lower()
            
            if actual == target:
                score = 1.0
                passed = True
            else:
                # Fuzzy: character-by-character comparison
                if len(actual) == len(target):
                    matches = sum(a == b for a, b in zip(actual, target))
                    score = matches / len(target)
                else:
                    # Length mismatch
                    score = max(0, 1.0 - abs(len(actual) - len(target)) * 0.1)
                    # Also check overlap
                    min_len = min(len(actual), len(target))
                    if min_len > 0:
                        matches = sum(a == b for a, b in zip(actual[:min_len], target[:min_len]))
                        score = max(score, matches / len(target) * 0.8)
                passed = False
            
            if mode == MatchMode.EXACT:
                score = 1.0 if passed else 0.0
        else:
            # Any valid word - just check it's non-empty
            score = 1.0 if extracted else 0.0
            passed = bool(extracted)
        
        return VerificationResult(
            score=score,
            passed=passed,
            constraint_name=self.name,
            details={
                'extracted': extracted,
                'target': self.target,
                'position': self.position
            }
        )
    
    def invert(self) -> 'NotAcrostic':
        return NotAcrostic(self.target, self.case_sensitive, self.position)
    
    def rubric_name(self) -> str:
        if self.target:
            return f"Acrostic: {self.target}"
        return "Acrostic Pattern"


@dataclass
class NotAcrostic(Constraint):
    """Lines must NOT form an acrostic."""
    forbidden_word: Optional[str] = None
    case_sensitive: bool = False
    position: Literal['first', 'last'] = 'first'
    
    @property
    def name(self) -> str:
        if self.forbidden_word:
            return f"not_acrostic_{self.forbidden_word.lower()}"
        return f"not_acrostic_{self.position}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if self.forbidden_word:
            return f'The first letters must NOT spell out "{self.forbidden_word}".'
        return "The first letters should not form any recognizable word."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        positive = Acrostic(self.forbidden_word, self.case_sensitive, self.position)
        result = positive.verify(ctx, mode)
        
        return VerificationResult(
            score=1.0 - result.score,
            passed=not result.passed,
            constraint_name=self.name,
            details=result.details
        )
    
    def invert(self) -> 'Acrostic':
        return Acrostic(self.forbidden_word, self.case_sensitive, self.position)
```

---

## 7. Negation Wrapper and Logical Operators

python

```python
# ============================================================================
# LOGICAL OPERATORS
# ============================================================================

@dataclass
class Not(Constraint):
    """
    Negation wrapper - inverts any constraint.
    
    Score inversion: score(Not(C)) = 1 - score(C)
    Pass inversion: passed(Not(C)) = not passed(C)
    """
    constraint: Constraint
    
    @property
    def name(self) -> str:
        return f"not_{self.constraint.name}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        base = self.constraint.to_instruction(context)
        # Transform positive to negative
        # Simple approach: prepend "NOT: " or transform common patterns
        
        replacements = [
            ("must have", "must NOT have"),
            ("must be", "must NOT be"),
            ("should have", "should NOT have"),
            ("should be", "should NOT be"),
            ("must follow", "must NOT follow"),
            ("must contain", "must NOT contain"),
        ]
        
        result = base
        for old, new in replacements:
            if old in result.lower():
                # Case-insensitive replacement
                import re
                result = re.sub(re.escape(old), new, result, flags=re.IGNORECASE)
                break
        else:
            # No pattern matched, prepend NOT
            result = f"NOT: {base}"
        
        return result
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        result = self.constraint.verify(ctx, mode)
        
        return VerificationResult(
            score=1.0 - result.score,
            passed=not result.passed,
            constraint_name=self.name,
            details={'inverted_from': result.constraint_name},
            sub_results=[result]
        )
    
    def invert(self) -> Constraint:
        """Double negation returns original."""
        return self.constraint
    
    def get_dependencies(self) -> List[str]:
        return self.constraint.get_dependencies()
    
    def rubric_name(self) -> str:
        return f"NOT: {self.constraint.rubric_name()}"
    
    def rubric_description(self) -> str:
        return self.to_instruction()


@dataclass
class And(Constraint):
    """All sub-constraints must hold."""
    constraints: List[Constraint]
    
    @property
    def name(self) -> str:
        return "and_" + "_".join(c.name for c in self.constraints[:3])
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        instructions = [c.to_instruction(context) for c in self.constraints]
        return " ".join(instructions)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        results = [c.verify(ctx, mode) for c in self.constraints]
        scores = [r.score for r in results]
        
        if mode == MatchMode.EXACT:
            final_score = 1.0 if all(r.passed for r in results) else 0.0
        else:
            final_score = sum(scores) / len(scores) if scores else 1.0
        
        return VerificationResult(
            score=final_score,
            passed=all(r.passed for r in results),
            constraint_name=self.name,
            sub_results=results
        )
    
    def invert(self) -> 'Or':
        """De Morgan's law: NOT(A AND B) = NOT(A) OR NOT(B)"""
        return Or([c.invert() for c in self.constraints])
    
    def __and__(self, other: Constraint) -> 'And':
        if isinstance(other, And):
            return And(self.constraints + other.constraints)
        return And(self.constraints + [other])
    
    def get_dependencies(self) -> List[str]:
        return list(set(d for c in self.constraints for d in c.get_dependencies()))


@dataclass
class Or(Constraint):
    """At least one sub-constraint must hold."""
    constraints: List[Constraint]
    
    @property
    def name(self) -> str:
        return "or_" + "_".join(c.name for c in self.constraints[:3])
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        instructions = [c.to_instruction(context) for c in self.constraints]
        if len(instructions) == 2:
            return f"Either: {instructions[0]} OR: {instructions[1]}"
        return "One of: " + " OR ".join(instructions)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        results = [c.verify(ctx, mode) for c in self.constraints]
        scores = [r.score for r in results]
        
        final_score = max(scores) if scores else 0.0
        
        return VerificationResult(
            score=final_score,
            passed=any(r.passed for r in results),
            constraint_name=self.name,
            sub_results=results
        )
    
    def invert(self) -> 'And':
        """De Morgan's law: NOT(A OR B) = NOT(A) AND NOT(B)"""
        return And([c.invert() for c in self.constraints])
    
    def get_dependencies(self) -> List[str]:
        return list(set(d for c in self.constraints for d in c.get_dependencies()))


@dataclass
class ExactlyN(Constraint):
    """Exactly N of the sub-constraints must hold."""
    constraints: List[Constraint]
    n: int
    
    @property
    def name(self) -> str:
        return f"exactly_{self.n}_of_{len(self.constraints)}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        instructions = [c.to_instruction(context) for c in self.constraints]
        return f"Exactly {self.n} of the following must hold: " + "; ".join(instructions)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        results = [c.verify(ctx, mode) for c in self.constraints]
        passed_count = sum(1 for r in results if r.passed)
        
        if mode == MatchMode.EXACT:
            score = 1.0 if passed_count == self.n else 0.0
        else:
            # Fuzzy: penalize distance from target count
            diff = abs(passed_count - self.n)
            score = math.exp(-0.5 * diff)
        
        return VerificationResult(
            score=score,
            passed=passed_count == self.n,
            constraint_name=self.name,
            details={'target': self.n, 'actual': passed_count},
            sub_results=results
        )
    
    def invert(self) -> 'NotExactlyN':
        return NotExactlyN(self.constraints, self.n)
    
    def get_dependencies(self) -> List[str]:
        return list(set(d for c in self.constraints for d in c.get_dependencies()))


@dataclass
class NotExactlyN(Constraint):
    """NOT exactly N of the sub-constraints hold (any other count is fine)."""
    constraints: List[Constraint]
    n: int
    
    @property
    def name(self) -> str:
        return f"not_exactly_{self.n}_of_{len(self.constraints)}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        return f"Do NOT satisfy exactly {self.n} of the following constraints (any other count is acceptable)."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        positive = ExactlyN(self.constraints, self.n)
        result = positive.verify(ctx, mode)
        
        return VerificationResult(
            score=1.0 - result.score,
            passed=not result.passed,
            constraint_name=self.name,
            details=result.details,
            sub_results=result.sub_results
        )
    
    def invert(self) -> 'ExactlyN':
        return ExactlyN(self.constraints, self.n)


@dataclass
class AtLeastN(Constraint):
    """At least N of the sub-constraints must hold."""
    constraints: List[Constraint]
    n: int
    
    @property
    def name(self) -> str:
        return f"at_least_{self.n}_of_{len(self.constraints)}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        instructions = [c.to_instruction(context) for c in self.constraints]
        return f"At least {self.n} of the following must hold: " + "; ".join(instructions)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        results = [c.verify(ctx, mode) for c in self.constraints]
        passed_count = sum(1 for r in results if r.passed)
        
        if mode == MatchMode.EXACT:
            score = 1.0 if passed_count >= self.n else 0.0
        else:
            if passed_count >= self.n:
                score = 1.0
            else:
                score = passed_count / self.n
        
        return VerificationResult(
            score=score,
            passed=passed_count >= self.n,
            constraint_name=self.name,
            details={'minimum': self.n, 'actual': passed_count},
            sub_results=results
        )
    
    def invert(self) -> 'AtMostN':
        """NOT(at least N) = at most N-1"""
        return AtMostN(self.constraints, self.n - 1)
    
    def get_dependencies(self) -> List[str]:
        return list(set(d for c in self.constraints for d in c.get_dependencies()))


@dataclass
class AtMostN(Constraint):
    """At most N of the sub-constraints can hold."""
    constraints: List[Constraint]
    n: int
    
    @property
    def name(self) -> str:
        return f"at_most_{self.n}_of_{len(self.constraints)}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        instructions = [c.to_instruction(context) for c in self.constraints]
        return f"At most {self.n} of the following can hold: " + "; ".join(instructions)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        results = [c.verify(ctx, mode) for c in self.constraints]
        passed_count = sum(1 for r in results if r.passed)
        
        if mode == MatchMode.EXACT:
            score = 1.0 if passed_count <= self.n else 0.0
        else:
            if passed_count <= self.n:
                score = 1.0
            else:
                overage = passed_count - self.n
                score = math.exp(-0.5 * overage)
        
        return VerificationResult(
            score=score,
            passed=passed_count <= self.n,
            constraint_name=self.name,
            details={'maximum': self.n, 'actual': passed_count},
            sub_results=results
        )
    
    def invert(self) -> 'AtLeastN':
        """NOT(at most N) = at least N+1"""
        return AtLeastN(self.constraints, self.n + 1)
    
    def get_dependencies(self) -> List[str]:
        return list(set(d for c in self.constraints for d in c.get_dependencies()))


@dataclass
class Implies(Constraint):
    """If condition holds, then consequence must hold."""
    condition: Constraint
    consequence: Constraint
    
    @property
    def name(self) -> str:
        return f"if_{self.condition.name}_then_{self.consequence.name}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        cond = self.condition.to_instruction(context)
        cons = self.consequence.to_instruction(context)
        return f"If {cond.lower()}, then {cons.lower()}"
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        cond_result = self.condition.verify(ctx, mode)
        
        if not cond_result.passed:
            # Condition false -> implication vacuously true
            return VerificationResult(
                score=1.0,
                passed=True,
                constraint_name=self.name,
                details={'condition_met': False, 'vacuously_true': True},
                sub_results=[cond_result]
            )
        
        cons_result = self.consequence.verify(ctx, mode)
        
        return VerificationResult(
            score=cons_result.score,
            passed=cons_result.passed,
            constraint_name=self.name,
            details={'condition_met': True},
            sub_results=[cond_result, cons_result]
        )
    
    def invert(self) -> 'And':
        """NOT(A -> B) = A AND NOT(B)"""
        return And([self.condition, self.consequence.invert()])
    
    def get_dependencies(self) -> List[str]:
        return self.condition.get_dependencies() + self.consequence.get_dependencies()
```

---

## 8. Complete Form Templates for Known Forms

python

```python
# ============================================================================
# COMPLETE FORM TEMPLATES
# ============================================================================

class FormLibrary:
    """Library of all known poetic forms expressed as constraints."""
    
    @staticmethod
    def sonnet_shakespearean() -> 'FormTemplate':
        """
        Shakespearean (English) Sonnet:
        - 14 lines
        - 3 quatrains + 1 couplet
        - Rhyme scheme: ABAB CDCD EFEF GG
        - Iambic pentameter (10 syllables per line)
        """
        return FormTemplate(
            name="Shakespearean Sonnet",
            description="A 14-line poem in iambic pentameter with three quatrains and a couplet.",
            constraints=[
                LineCount.exactly(14),
                RhymeScheme("ABAB CDCD EFEF GG"),
                SyllableCount.per_line_exactly(10),  # Iambic pentameter
            ],
            weights={
                'line_count': 0.15,
                'rhyme_scheme': 0.50,
                'syllable_count': 0.35
            }
        )
    
    @staticmethod
    def sonnet_petrarchan() -> 'FormTemplate':
        """
        Petrarchan (Italian) Sonnet:
        - 14 lines
        - Octave (8 lines) + Sestet (6 lines)
        - Rhyme scheme: ABBAABBA + CDECDE (or CDCDCD)
        """
        return FormTemplate(
            name="Petrarchan Sonnet",
            description="A 14-line poem with an octave and sestet, following Italian rhyme patterns.",
            constraints=[
                LineCount.exactly(14),
                StanzaStructure([8, 6]),
                RhymeScheme("ABBAABBA CDECDE"),
            ],
            weights={
                'line_count': 0.15,
                'stanza_structure': 0.15,
                'rhyme_scheme': 0.70
            }
        )
    
    @staticmethod
    def sonnet_without_rhymes(lines: int = 15) -> 'FormTemplate':
        """
        Unconventional: Sonnet structure without rhymes.
        Demonstrates constraint composition with inversions.
        """
        return FormTemplate(
            name="Unrhymed Sonnet Variant",
            description=f"A {lines}-line poem with sonnet-like structure but no rhyme scheme.",
            constraints=[
                LineCount.exactly(lines),
                ~RhymeScheme("ABAB CDCD EFEF GG"),  # NOT the standard scheme
                SyllableCount(NumericBound.between(8, 12), scope='line'),
            ]
        )
    
    @staticmethod
    def haiku() -> 'FormTemplate':
        """
        Haiku:
        - 3 lines
        - 5-7-5 syllable pattern
        - No title (traditionally)
        """
        return FormTemplate(
            name="Haiku",
            description="A 3-line Japanese form with 5-7-5 syllable structure.",
            constraints=[
                LineCount.exactly(3),
                StanzaCount.exactly(1),
                SyllablePattern([5, 7, 5], tolerance=0),
                HasTitle(required=False),
            ],
            weights={
                'line_count': 0.1,
                'stanza_count': 0.05,
                'syllable_pattern': 0.80,
                'title': 0.05
            }
        )
    
    @staticmethod
    def tanka() -> 'FormTemplate':
        """
        Tanka:
        - 5 lines
        - 5-7-5-7-7 syllable pattern
        """
        return FormTemplate(
            name="Tanka",
            description="A 5-line Japanese form with 5-7-5-7-7 syllable structure.",
            constraints=[
                LineCount.exactly(5),
                SyllablePattern([5, 7, 5, 7, 7], tolerance=0),
            ]
        )
    
    @staticmethod
    def villanelle() -> 'FormTemplate':
        """
        Villanelle:
        - 19 lines
        - 5 tercets + 1 quatrain
        - Two refrains (A1 and A2)
        - Rhyme scheme: ABA for tercets, ABAA for quatrain
        """
        def line(n: int) -> Position:
            return Position(line=n)
        
        return FormTemplate(
            name="Villanelle",
            description="A 19-line poem with two repeating refrains and intricate rhyme scheme.",
            constraints=[
                LineCount.exactly(19),
                StanzaCount.exactly(6),
                StanzaStructure([3, 3, 3, 3, 3, 4]),
                RhymeScheme("ABA ABA ABA ABA ABA ABAA"),
                # A1 refrain: lines 1, 6, 12, 18
                RefrainGroup([line(0), line(5), line(11), line(17)], "A1"),
                # A2 refrain: lines 3, 9, 15, 19
                RefrainGroup([line(2), line(8), line(14), line(18)], "A2"),
            ],
            weights={
                'line_count': 0.05,
                'stanza_count': 0.05,
                'stanza_structure': 0.10,
                'rhyme_scheme': 0.30,
                'refrain_A1': 0.25,
                'refrain_A2': 0.25
            }
        )
    
    @staticmethod
    def sestina() -> 'FormTemplate':
        """
        Sestina:
        - 39 lines (6 sixains + 1 tercet envoi)
        - 6 end-words rotate in specific pattern
        - No rhyme requirement
        """
        return FormTemplate(
            name="Sestina",
            description="A 39-line poem with six stanzas using rotating end-words.",
            constraints=[
                LineCount.exactly(39),
                StanzaCount.exactly(7),
                StanzaStructure([6, 6, 6, 6, 6, 6, 3]),
                EndWordRotation(
                    word_count=6,
                    rotation_rule=[5, 0, 4, 1, 3, 2],
                    stanza_count=6
                ),
            ],
            weights={
                'line_count': 0.10,
                'stanza_count': 0.10,
                'stanza_structure': 0.15,
                'endword_rotation': 0.65
            }
        )
    
    @staticmethod
    def limerick() -> 'FormTemplate':
        """
        Limerick:
        - 5 lines
        - AABBA rhyme scheme
        - Lines 1, 2, 5: ~8 syllables; Lines 3, 4: ~5 syllables
        """
        return FormTemplate(
            name="Limerick",
            description="A humorous 5-line poem with AABBA rhyme scheme.",
            constraints=[
                LineCount.exactly(5),
                RhymeScheme("AABBA"),
                # Syllable pattern (approximate)
                SyllableCount(NumericBound.between(7, 10), scope='line', target_index=0),
                SyllableCount(NumericBound.between(7, 10), scope='line', target_index=1),
                SyllableCount(NumericBound.between(4, 6), scope='line', target_index=2),
                SyllableCount(NumericBound.between(4, 6), scope='line', target_index=3),
                SyllableCount(NumericBound.between(7, 10), scope='line', target_index=4),
            ]
        )
    
    @staticmethod
    def ghazal() -> 'FormTemplate':
        """
        Ghazal:
        - 5-15 couplets
        - AA BA CA DA... rhyme scheme (all second lines rhyme)
        - Refrain (radif) at end of each couplet
        """
        return FormTemplate(
            name="Ghazal",
            description="An Arabic/Persian form of rhyming couplets with a refrain.",
            constraints=[
                LineCount(NumericBound.between(10, 30)),  # 5-15 couplets
                StanzaStructure([2] * 5),  # At least 5 couplets (can be more)
                # Even lines rhyme (AA BA CA DA pattern means lines 0,1 rhyme, then 3,5,7... rhyme with 1)
                RhymeScheme("AA BA CA DA EA"),  # For 5 couplets
            ]
        )
    
    @staticmethod
    def pantoum() -> 'FormTemplate':
        """
        Pantoum:
        - Quatrains with interlinked repetition
        - Lines 2 and 4 of each stanza become lines 1 and 3 of the next
        - ABAB rhyme scheme per stanza
        """
        def line(n: int) -> Position:
            return Position(line=n)
        
        # For a 4-stanza pantoum (16 lines)
        return FormTemplate(
            name="Pantoum",
            description="A Malayan form with interlinked repeating lines between stanzas.",
            constraints=[
                LineCount.exactly(16),
                StanzaCount.exactly(4),
                StanzaStructure([4, 4, 4, 4]),
                RhymeScheme("ABAB ABAB ABAB ABAB"),
                # Line 2 of stanza 1 = Line 1 of stanza 2
                Repeats(line(1), line(4)),
                # Line 4 of stanza 1 = Line 3 of stanza 2
                Repeats(line(3), line(6)),
                # Continue pattern...
                Repeats(line(5), line(8)),
                Repeats(line(7), line(10)),
                Repeats(line(9), line(12)),
                Repeats(line(11), line(14)),
            ]
        )
    
    @staticmethod
    def free_verse_with_constraints(
        min_lines: int = 10,
        max_lines: int = 30,
        require_title: bool = True,
        title_max_words: int = 5
    ) -> 'FormTemplate':
        """
        Free verse with optional constraints.
        Demonstrates how to build custom forms.
        """
        constraints = [
            LineCount.between(min_lines, max_lines),
        ]
        
        if require_title:
            constraints.append(HasTitle(required=True, format=TitleFormat.BRACKETED_FIRST))
            constraints.append(TitleWordCount.at_most(title_max_words, TitleFormat.BRACKETED_FIRST))
        
        return FormTemplate(
            name="Constrained Free Verse",
            description=f"Free verse with {min_lines}-{max_lines} lines and optional title constraints.",
            constraints=constraints
        )
    
    @staticmethod
    def acrostic_poem(word: str) -> 'FormTemplate':
        """
        Acrostic poem spelling a specific word.
        """
        return FormTemplate(
            name=f"Acrostic: {word.upper()}",
            description=f"A poem whose first letters spell '{word.upper()}'.",
            constraints=[
                LineCount.exactly(len(word)),
                Acrostic.spelling(word),
            ]
        )
    
    @staticmethod
    def double_acrostic(start_word: str, end_word: str) -> 'FormTemplate':
        """
        Double acrostic: first AND last letters spell words.
        """
        if len(start_word) != len(end_word):
            raise ValueError("Start and end words must have same length")
        
        return FormTemplate(
            name=f"Double Acrostic: {start_word.upper()}/{end_word.upper()}",
            description=f"First letters spell '{start_word}', last letters spell '{end_word}'.",
            constraints=[
                LineCount.exactly(len(start_word)),
                Acrostic(target=start_word, position='first'),
                Acrostic(target=end_word, position='last'),
            ]
        )
```

---

## 9. Complete Environment with Auto-Rubric

python

```python
# ============================================================================
# COMPLETE ENVIRONMENT
# ============================================================================

@dataclass
class FormTemplate:
    """A complete poetic form specification."""
    name: str
    description: str
    constraints: List[Constraint]
    weights: Optional[Dict[str, float]] = None
    
    def get_all_constraints(self) -> List[Constraint]:
        """Flatten all constraints including nested ones."""
        result = []
        for c in self.constraints:
            result.append(c)
            if hasattr(c, 'constraints'):
                result.extend(c.constraints)
        return result
    
    def to_instruction(self) -> str:
        """Generate complete natural language instruction."""
        lines = [
            f"Write a {self.name}.",
            "",
            self.description,
            "",
            "Requirements:",
        ]
        for i, c in enumerate(self.constraints, 1):
            lines.append(f"{i}. {c.to_instruction()}")
        return "\n".join(lines)
    
    def generate_rubric(self, total_points: float = 100.0) -> Rubric:
        """Generate grading rubric from constraints."""
        return Rubric.from_constraints(
            self.constraints,
            self.weights,
            total_points,
            title=f"{self.name} Grading Rubric"
        )


class ComposablePoetryEnvironment:
    """
    Complete reward environment with:
    - All known forms
    - Constraint composition
    - Auto-rubric generation
    - Novel form generation
    """
    
    def __init__(self, cmu_dict: Optional[Dict] = None):
        self.cmu_dict = cmu_dict or {}
        self.form_library = FormLibrary()
        self.form_generator = NovelFormGenerator()
    
    def evaluate(
        self,
        poem: str,
        form: Union[str, FormTemplate, Constraint],
        mode: MatchMode = MatchMode.FUZZY
    ) -> Tuple['CompositeRewardResult', Rubric, GradeReport]:
        """
        Evaluate poem and return reward, rubric, and grade report.
        """
        # Resolve form
        if isinstance(form, str):
            template = self._get_form_by_name(form)
        elif isinstance(form, Constraint):
            template = FormTemplate("Custom", "Custom constraints", [form])
        else:
            template = form
        
        # Build context
        ctx = ConstraintContext.from_poem(poem, self.cmu_dict)
        
        # Evaluate constraints
        results = {}
        for constraint in template.constraints:
            result = constraint.verify(ctx, mode)
            results[constraint.name] = result
        
        # Calculate scores
        weights = template.weights or {}
        total_weight = sum(weights.get(c.name, 1.0) for c in template.constraints)
        
        weighted_scores = []
        for c in template.constraints:
            w = weights.get(c.name, 1.0)
            r = results[c.name]
            weighted_scores.append((r.score * w, w))
        
        total_score = sum(s for s, _ in weighted_scores) / total_weight
        
        # Generate rubric and grade
        rubric = template.generate_rubric()
        grade = rubric.grade(results)
        
        reward_result = CompositeRewardResult(
            total_score=total_score,
            weighted_score=total_score,
            constraint_results=results,
            instruction_used=template.to_instruction(),
            form_name=template.name,
            metadata={'mode': mode.name}
        )
        
        return reward_result, rubric, grade
    
    def _get_form_by_name(self, name: str) -> FormTemplate:
        """Look up form by name."""
        forms = {
            'haiku': self.form_library.haiku,
            'tanka': self.form_library.tanka,
            'sonnet': self.form_library.sonnet_shakespearean,
            'shakespearean_sonnet': self.form_library.sonnet_shakespearean,
            'petrarchan_sonnet': self.form_library.sonnet_petrarchan,
            'villanelle': self.form_library.villanelle,
            'sestina': self.form_library.sestina,
            'limerick': self.form_library.limerick,
            'ghazal': self.form_library.ghazal,
            'pantoum': self.form_library.pantoum,
        }
        
        if name.lower() not in forms:
            raise ValueError(f"Unknown form: {name}. Available: {list(forms.keys())}")
        
        return forms[name.lower()]()
    
    def create_custom_form(
        self,
        name: str,
        constraints: List[Constraint],
        description: str = "",
        weights: Optional[Dict[str, float]] = None
    ) -> FormTemplate:
        """Create a custom form from constraints."""
        return FormTemplate(
            name=name,
            description=description or f"Custom form: {name}",
            constraints=constraints,
            weights=weights
        )
    
    def generate_training_example(
        self,
        difficulty: str = 'medium'
    ) -> Tuple[str, FormTemplate, Rubric, Callable[[str], float]]:
        """
        Generate a complete training example:
        - Instruction
        - Form template
        - Rubric
        - Reward function
        """
        template, metadata = self.form_generator.generate_form(difficulty)
        rubric = template.generate_rubric()
        instruction = template.to_instruction()
        
        def reward_fn(poem: str) -> float:
            result, _, _ = self.evaluate(poem, template)
            return result.total_score
        
        return instruction, template, rubric, reward_fn


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def demonstrate_system():
    """Demonstrate the complete system."""
    
    env = ComposablePoetryEnvironment()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Example 1: Standard haiku evaluation
    # ─────────────────────────────────────────────────────────────────────────
    
    haiku = """
An old silent pond
A frog jumps into the pond
Splash! Silence again
"""
    
    result, rubric, grade = env.evaluate(haiku, 'haiku')
    
    print("=" * 60)
    print("HAIKU EVALUATION")
    print("=" * 60)
    print(f"Score: {result.total_score:.2f}")
    print(grade.format_markdown())
    
    # ─────────────────────────────────────────────────────────────────────────
    # Example 2: Custom constraint composition
    # ─────────────────────────────────────────────────────────────────────────
    
    # "Write a poem that is NOT a haiku but has exactly 3 lines"
    anti_haiku = env.create_custom_form(
        name="Anti-Haiku",
        constraints=[
            LineCount.exactly(3),
            ~SyllablePattern([5, 7, 5]),  # NOT 5-7-5
            SyllableCount(NumericBound.at_least(20), scope='total'),  # More syllables
        ],
        description="A 3-line poem that deliberately violates haiku syllable rules."
    )
    
    test_poem = """
This is definitely not a haiku because it has way too many syllables
And this line is also very long with many many words
Breaking all the rules of traditional Japanese poetry forms
"""
    
    result2, rubric2, grade2 = env.evaluate(test_poem, anti_haiku)
    
    print("\n" + "=" * 60)
    print("ANTI-HAIKU EVALUATION")
    print("=" * 60)
    print(rubric2.format_markdown())
    print(grade2.format_markdown())
    
    # ─────────────────────────────────────────────────────────────────────────
    # Example 3: Complex constraint with title requirements
    # ─────────────────────────────────────────────────────────────────────────
    
    titled_form = env.create_custom_form(
        name="Titled Constrained Verse",
        constraints=[
            HasTitle(required=True, format=TitleFormat.BRACKETED_FIRST),
            TitleWordCount.at_most(3, TitleFormat.BRACKETED_FIRST),
            LineCount.between(8, 12),
            WordCount.per_line_at_least(5),
            ~LineStartsWith("The", line_index=None),  # No line starts with "The"
        ],
        description="A poem with bracketed title (≤3 words), 8-12 lines, no lines starting with 'The'."
    )
    
    print("\n" + "=" * 60)
    print("TITLED FORM RUBRIC")
    print("=" * 60)
    print(titled_form.generate_rubric().format_markdown())
    
    # ─────────────────────────────────────────────────────────────────────────
```

Continue

python

```python
    # ─────────────────────────────────────────────────────────────────────────
    # Example 4: Inverted/Subverted Forms
    # ─────────────────────────────────────────────────────────────────────────
    
    # "Sonnet without rhymes and with 15 lines instead of 14"
    broken_sonnet = env.create_custom_form(
        name="Broken Sonnet",
        constraints=[
            LineCount.exactly(15),  # Not 14
            LineCount.not_equal(14),  # Explicitly not 14 (redundant but demonstrates)
            ~RhymeScheme("ABAB CDCD EFEF GG"),  # No Shakespearean rhyme
            SyllableCount(NumericBound.between(8, 12), scope='line'),  # Loose pentameter
        ],
        description="A deliberately malformed sonnet: 15 lines, no traditional rhyme scheme.",
        weights={
            'line_count_exactly_15': 0.3,
            'line_count_not_equal_14': 0.1,
            'rhyme_scheme': 0.4,
            'syllable_count': 0.2
        }
    )
    
    print("\n" + "=" * 60)
    print("BROKEN SONNET SPECIFICATION")
    print("=" * 60)
    print(broken_sonnet.to_instruction())
    print("\n" + broken_sonnet.generate_rubric().format_markdown())
    
    # ─────────────────────────────────────────────────────────────────────────
    # Example 5: Conditional constraints
    # ─────────────────────────────────────────────────────────────────────────
    
    # "If the poem has a title, it must be exactly 2 words"
    conditional_form = env.create_custom_form(
        name="Conditional Title Form",
        constraints=[
            LineCount.between(5, 20),
            Implies(
                HasTitle(required=True, format=TitleFormat.BRACKETED_FIRST),
                TitleWordCount.exactly(2, TitleFormat.BRACKETED_FIRST)
            ),
        ],
        description="If poem has a bracketed title, it must be exactly 2 words."
    )
    
    print("\n" + "=" * 60)
    print("CONDITIONAL FORM")
    print("=" * 60)
    print(conditional_form.to_instruction())
    
    # ─────────────────────────────────────────────────────────────────────────
    # Example 6: "Exactly N of M" constraints
    # ─────────────────────────────────────────────────────────────────────────
    
    # "Satisfy exactly 2 of these 4 constraints"
    pick_two_form = env.create_custom_form(
        name="Choose Your Constraints",
        constraints=[
            ExactlyN([
                RhymeScheme("AABB"),
                SyllablePattern([8, 8, 8, 8]),
                Acrostic.spelling("POEM"),
                LineCount.exactly(4),
            ], n=2)
        ],
        description="A 4-line poem satisfying exactly 2 of: couplet rhymes, 8-syllable lines, POEM acrostic, or 4 lines."
    )
    
    print("\n" + "=" * 60)
    print("PICK-TWO FORM")
    print("=" * 60)
    print(pick_two_form.to_instruction())


def demonstrate_inversions():
    """Demonstrate constraint inversions and De Morgan's laws."""
    
    print("\n" + "=" * 60)
    print("CONSTRAINT INVERSIONS")
    print("=" * 60)
    
    # Basic inversions
    constraints = [
        LineCount.exactly(14),
        LineCount.at_least(10),
        LineCount.at_most(20),
        WordCount.per_line_exactly(5),
        HasTitle(required=True),
        RhymeScheme("ABAB"),
        Acrostic.spelling("HELLO"),
    ]
    
    print("\nOriginal → Inverted:")
    print("-" * 50)
    for c in constraints:
        inverted = ~c  # Using __invert__
        print(f"{c.to_instruction()}")
        print(f"  → {inverted.to_instruction()}")
        print()
    
    # De Morgan's laws
    print("\nDe Morgan's Laws:")
    print("-" * 50)
    
    a = LineCount.exactly(14)
    b = RhymeScheme("ABAB")
    
    # NOT(A AND B) = NOT(A) OR NOT(B)
    and_constraint = a & b
    inverted_and = ~and_constraint
    
    print(f"A AND B: {and_constraint.to_instruction()[:60]}...")
    print(f"NOT(A AND B): {inverted_and.to_instruction()[:60]}...")
    print(f"Type of NOT(A AND B): {type(inverted_and).__name__}")  # Should be Or
    
    # NOT(A OR B) = NOT(A) AND NOT(B)
    or_constraint = a | b
    inverted_or = ~or_constraint
    
    print(f"\nA OR B: {or_constraint.to_instruction()[:60]}...")
    print(f"NOT(A OR B): {inverted_or.to_instruction()[:60]}...")
    print(f"Type of NOT(A OR B): {type(inverted_or).__name__}")  # Should be And


if __name__ == "__main__":
    demonstrate_system()
    demonstrate_inversions()
```

---

## 10. Additional Constraints for Completeness

python

```python
# ============================================================================
# ADDITIONAL CONSTRAINTS FOR COMPLETENESS
# ============================================================================

@dataclass
class ContainsWord(Constraint):
    """Line or poem must contain specific word(s)."""
    words: List[str]
    scope: Literal['line', 'poem'] = 'poem'
    line_index: Optional[int] = None
    match_all: bool = False  # True = all words required, False = any word
    case_sensitive: bool = False
    
    @property
    def name(self) -> str:
        return f"contains_{'all' if self.match_all else 'any'}_{'_'.join(self.words[:3])}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        words_str = ', '.join(f'"{w}"' for w in self.words)
        quantifier = "all of" if self.match_all else "at least one of"
        
        if self.scope == 'poem':
            return f"The poem must contain {quantifier} these words: {words_str}."
        elif self.line_index is not None:
            return f"Line {self.line_index + 1} must contain {quantifier}: {words_str}."
        else:
            return f"Each line must contain {quantifier}: {words_str}."
    
    def _check_text(self, text: str) -> Tuple[bool, float]:
        """Check if text contains required words."""
        check_text = text if self.case_sensitive else text.lower()
        check_words = self.words if self.case_sensitive else [w.lower() for w in self.words]
        
        found = [w in check_text for w in check_words]
        found_count = sum(found)
        
        if self.match_all:
            passed = all(found)
            score = found_count / len(self.words)
        else:
            passed = any(found)
            score = 1.0 if passed else 0.0
        
        return passed, score
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        if self.scope == 'poem':
            full_text = ' '.join(ctx.structure.lines)
            passed, score = self._check_text(full_text)
        elif self.line_index is not None:
            if self.line_index >= len(ctx.structure.lines):
                return VerificationResult(score=0.0, passed=False, constraint_name=self.name,
                                         details={'error': 'Line does not exist'})
            passed, score = self._check_text(ctx.structure.lines[self.line_index])
        else:
            scores = []
            passes = []
            for line in ctx.structure.lines:
                p, s = self._check_text(line)
                passes.append(p)
                scores.append(s)
            passed = all(passes) if self.match_all else any(passes)
            score = sum(scores) / len(scores) if scores else 0.0
        
        if mode == MatchMode.EXACT:
            score = 1.0 if passed else 0.0
        
        return VerificationResult(score=score, passed=passed, constraint_name=self.name,
                                 details={'words': self.words})
    
    def invert(self) -> 'DoesNotContainWord':
        return DoesNotContainWord(self.words, self.scope, self.line_index, 
                                  not self.match_all, self.case_sensitive)


@dataclass
class DoesNotContainWord(Constraint):
    """Line or poem must NOT contain specific word(s)."""
    words: List[str]
    scope: Literal['line', 'poem'] = 'poem'
    line_index: Optional[int] = None
    forbid_all: bool = True  # True = none allowed, False = not all at once
    case_sensitive: bool = False
    
    @property
    def name(self) -> str:
        return f"not_contains_{'_'.join(self.words[:3])}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        words_str = ', '.join(f'"{w}"' for w in self.words)
        if self.forbid_all:
            return f"The poem must NOT contain any of these words: {words_str}."
        else:
            return f"The poem must not contain ALL of these words together: {words_str}."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        positive = ContainsWord(self.words, self.scope, self.line_index, 
                               self.forbid_all, self.case_sensitive)
        result = positive.verify(ctx, mode)
        
        return VerificationResult(
            score=1.0 - result.score,
            passed=not result.passed,
            constraint_name=self.name,
            details=result.details
        )
    
    def invert(self) -> 'ContainsWord':
        return ContainsWord(self.words, self.scope, self.line_index,
                           self.forbid_all, self.case_sensitive)


@dataclass
class UniqueWords(Constraint):
    """Words must be unique (no repetition) within scope."""
    scope: Literal['line', 'stanza', 'poem'] = 'poem'
    min_unique_ratio: float = 1.0  # 1.0 = all unique, 0.5 = at least half unique
    
    @property
    def name(self) -> str:
        return f"unique_words_{self.scope}_{self.min_unique_ratio}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if self.min_unique_ratio >= 1.0:
            return f"All words in the {self.scope} must be unique (no repetition)."
        else:
            pct = int(self.min_unique_ratio * 100)
            return f"At least {pct}% of words in the {self.scope} must be unique."
    
    def _compute_unique_ratio(self, text: str) -> float:
        words = text.lower().split()
        words = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation)]
        if not words:
            return 1.0
        unique = set(words)
        return len(unique) / len(words)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        if self.scope == 'poem':
            full_text = ' '.join(ctx.structure.lines)
            ratio = self._compute_unique_ratio(full_text)
            passed = ratio >= self.min_unique_ratio
            score = ratio / self.min_unique_ratio if self.min_unique_ratio > 0 else 1.0
            score = min(1.0, score)
        elif self.scope == 'line':
            ratios = [self._compute_unique_ratio(line) for line in ctx.structure.lines]
            passes = [r >= self.min_unique_ratio for r in ratios]
            passed = all(passes)
            score = sum(ratios) / len(ratios) if ratios else 1.0
        else:  # stanza
            ratios = []
            for stanza in ctx.structure.stanzas:
                stanza_text = ' '.join(stanza)
                ratios.append(self._compute_unique_ratio(stanza_text))
            passes = [r >= self.min_unique_ratio for r in ratios]
            passed = all(passes)
            score = sum(ratios) / len(ratios) if ratios else 1.0
        
        if mode == MatchMode.EXACT:
            score = 1.0 if passed else 0.0
        
        return VerificationResult(score=score, passed=passed, constraint_name=self.name,
                                 details={'scope': self.scope, 'min_ratio': self.min_unique_ratio})
    
    def invert(self) -> 'Not':
        return Not(self)


@dataclass
class Alliteration(Constraint):
    """Lines must have alliteration (repeated initial consonants)."""
    min_alliterative_words: int = 2
    scope: Literal['line', 'stanza'] = 'line'
    line_index: Optional[int] = None
    
    @property
    def name(self) -> str:
        return f"alliteration_{self.min_alliterative_words}_{self.scope}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if self.scope == 'line':
            if self.line_index is not None:
                return f"Line {self.line_index + 1} must have at least {self.min_alliterative_words} words starting with the same consonant."
            return f"Each line should have alliteration (at least {self.min_alliterative_words} words with same initial consonant)."
        return f"Each stanza should have alliteration throughout."
    
    def _count_alliteration(self, text: str) -> Tuple[int, str]:
        """Count max alliterative words and return the consonant."""
        words = text.lower().split()
        words = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation)]
        
        if not words:
            return 0, ''
        
        # Count initial consonants
        consonants = 'bcdfghjklmnpqrstvwxyz'
        initial_counts: Dict[str, int] = {}
        
        for word in words:
            if word and word[0] in consonants:
                initial = word[0]
                initial_counts[initial] = initial_counts.get(initial, 0) + 1
        
        if not initial_counts:
            return 0, ''
        
        max_consonant = max(initial_counts, key=initial_counts.get)
        return initial_counts[max_consonant], max_consonant
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        if self.scope == 'line':
            if self.line_index is not None:
                if self.line_index >= len(ctx.structure.lines):
                    return VerificationResult(score=0.0, passed=False, constraint_name=self.name)
                count, consonant = self._count_alliteration(ctx.structure.lines[self.line_index])
                passed = count >= self.min_alliterative_words
                score = min(1.0, count / self.min_alliterative_words)
            else:
                scores = []
                passes = []
                for line in ctx.structure.lines:
                    count, _ = self._count_alliteration(line)
                    passes.append(count >= self.min_alliterative_words)
                    scores.append(min(1.0, count / self.min_alliterative_words))
                passed = all(passes)
                score = sum(scores) / len(scores) if scores else 0.0
        else:  # stanza
            scores = []
            passes = []
            for stanza in ctx.structure.stanzas:
                stanza_text = ' '.join(stanza)
                count, _ = self._count_alliteration(stanza_text)
                passes.append(count >= self.min_alliterative_words)
                scores.append(min(1.0, count / self.min_alliterative_words))
            passed = all(passes)
            score = sum(scores) / len(scores) if scores else 0.0
        
        if mode == MatchMode.EXACT:
            score = 1.0 if passed else 0.0
        
        return VerificationResult(score=score, passed=passed, constraint_name=self.name)
    
    def invert(self) -> 'Not':
        return Not(self)


@dataclass
class Meter(Constraint):
    """
    Lines must follow a specific metrical pattern.
    
    Uses stress patterns: 'u' = unstressed, '/' = stressed
    """
    pattern: str  # e.g., "u/u/u/u/u/" for iambic pentameter
    scope: Literal['line', 'poem'] = 'line'
    line_index: Optional[int] = None
    tolerance: float = 0.2  # Allow 20% deviation
    
    # Common patterns
    IAMBIC_PENTAMETER = "u/u/u/u/u/"
    IAMBIC_TETRAMETER = "u/u/u/u/"
    TROCHAIC_TETRAMETER = "/u/u/u/u"
    DACTYLIC_HEXAMETER = "/uu/uu/uu/uu/uu/x"
    
    @classmethod
    def iambic_pentameter(cls) -> 'Meter':
        return cls(pattern=cls.IAMBIC_PENTAMETER)
    
    @classmethod
    def iambic_tetrameter(cls) -> 'Meter':
        return cls(pattern=cls.IAMBIC_TETRAMETER)
    
    @property
    def name(self) -> str:
        return f"meter_{self.pattern.replace('/', 'S').replace('u', 'U')}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        # Identify common patterns
        pattern_names = {
            self.IAMBIC_PENTAMETER: "iambic pentameter (da-DUM da-DUM da-DUM da-DUM da-DUM)",
            self.IAMBIC_TETRAMETER: "iambic tetrameter (da-DUM da-DUM da-DUM da-DUM)",
            self.TROCHAIC_TETRAMETER: "trochaic tetrameter (DUM-da DUM-da DUM-da DUM-da)",
        }
        
        pattern_desc = pattern_names.get(self.pattern, f"the pattern {self.pattern}")
        
        if self.line_index is not None:
            return f"Line {self.line_index + 1} must be in {pattern_desc}."
        return f"The poem should be written in {pattern_desc}."
    
    def _get_stress_pattern(self, line: str, cmu_dict: dict) -> str:
        """Get stress pattern for a line using CMU dict."""
        words = re.findall(r"[a-zA-Z']+", line)
        pattern = ""
        
        for word in words:
            word_lower = word.lower()
            if word_lower in cmu_dict:
                phonemes = cmu_dict[word_lower][0]
                for p in phonemes:
                    if any(c.isdigit() for c in p):
                        # Vowel with stress marker
                        stress = ''.join(c for c in p if c.isdigit())
                        if stress in ('1', '2'):
                            pattern += '/'
                        else:
                            pattern += 'u'
            else:
                # Heuristic for unknown words
                syllables = self._count_syllables_simple(word)
                # Assume alternating stress starting with unstressed
                for i in range(syllables):
                    pattern += 'u' if i % 2 == 0 else '/'
        
        return pattern
    
    def _count_syllables_simple(self, word: str) -> int:
        """Simple syllable count heuristic."""
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in word.lower():
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(1, count)
    
    def _compare_patterns(self, actual: str, expected: str) -> float:
        """Compare stress patterns with tolerance."""
        if not expected:
            return 1.0 if not actual else 0.0
        
        # Normalize lengths
        min_len = min(len(actual), len(expected))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for a, e in zip(actual[:min_len], expected[:min_len]) if a == e)
        length_penalty = abs(len(actual) - len(expected)) / len(expected)
        
        base_score = matches / min_len
        return max(0, base_score - length_penalty * 0.5)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        if self.line_index is not None:
            if self.line_index >= len(ctx.structure.lines):
                return VerificationResult(score=0.0, passed=False, constraint_name=self.name)
            actual = self._get_stress_pattern(ctx.structure.lines[self.line_index], ctx.cmu_dict)
            score = self._compare_patterns(actual, self.pattern)
            passed = score >= (1.0 - self.tolerance)
        else:
            scores = []
            for line in ctx.structure.lines:
                actual = self._get_stress_pattern(line, ctx.cmu_dict)
                scores.append(self._compare_patterns(actual, self.pattern))
            score = sum(scores) / len(scores) if scores else 0.0
            passed = score >= (1.0 - self.tolerance)
        
        if mode == MatchMode.EXACT:
            score = 1.0 if passed else 0.0
        
        return VerificationResult(score=score, passed=passed, constraint_name=self.name,
                                 details={'expected_pattern': self.pattern})
    
    def invert(self) -> 'Not':
        return Not(self)
    
    def rubric_name(self) -> str:
        return "Metrical Pattern"


@dataclass
class LineLength(Constraint):
    """Constraint on line length in characters."""
    bound: NumericBound
    line_index: Optional[int] = None  # None = all lines
    include_spaces: bool = True
    
    @classmethod
    def exactly(cls, n: int, line: Optional[int] = None) -> 'LineLength':
        return cls(NumericBound.exactly(n), line)
    
    @classmethod
    def at_most(cls, n: int, line: Optional[int] = None) -> 'LineLength':
        return cls(NumericBound.at_most(n), line)
    
    @classmethod
    def between(cls, min_n: int, max_n: int, line: Optional[int] = None) -> 'LineLength':
        return cls(NumericBound.between(min_n, max_n), line)
    
    @property
    def name(self) -> str:
        line_str = f"line_{self.line_index}" if self.line_index is not None else "all"
        return f"line_length_{line_str}_{self.bound.describe().replace(' ', '_')}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        bound_desc = self.bound.describe()
        char_type = "characters" if self.include_spaces else "characters (excluding spaces)"
        
        if self.line_index is not None:
            return f"Line {self.line_index + 1} must have {bound_desc} {char_type}."
        return f"Each line must have {bound_desc} {char_type}."
    
    def _count_chars(self, line: str) -> int:
        if self.include_spaces:
            return len(line)
        return len(line.replace(' ', ''))
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        if self.line_index is not None:
            if self.line_index >= len(ctx.structure.lines):
                return VerificationResult(score=0.0, passed=False, constraint_name=self.name)
            count = self._count_chars(ctx.structure.lines[self.line_index])
            score = self.bound.score(count, mode)
            passed = self.bound.check(count)
        else:
            scores = []
            passes = []
            counts = []
            for line in ctx.structure.lines:
                count = self._count_chars(line)
                counts.append(count)
                scores.append(self.bound.score(count, mode))
                passes.append(self.bound.check(count))
            score = sum(scores) / len(scores) if scores else 1.0
            passed = all(passes)
        
        return VerificationResult(score=score, passed=passed, constraint_name=self.name,
                                 details={'bound': self.bound.describe()})
    
    def invert(self) -> 'LineLength':
        return LineLength(self.bound.invert(), self.line_index, self.include_spaces)
```

---

## 11. Novel Form Generator (Enhanced)

python

```python
# ============================================================================
# ENHANCED NOVEL FORM GENERATOR
# ============================================================================

class NovelFormGenerator:
    """
    Generate novel poetic forms by composing constraints.
    
    Supports:
    - Random form generation
    - Constraint mutation (inversions, modifications)
    - Difficulty scaling
    - Form hybridization
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.form_library = FormLibrary()
    
    def generate_form(
        self,
        difficulty: str = 'medium',
        allow_inversions: bool = True,
        allow_conditionals: bool = True
    ) -> Tuple[FormTemplate, Dict[str, Any]]:
        """Generate a complete novel form."""
        
        metadata = {'difficulty': difficulty, 'inversions_used': [], 'constraint_types': []}
        constraints = []
        
        # Difficulty parameters
        params = {
            'easy': {
                'line_range': (4, 10),
                'max_constraints': 4,
                'inversion_prob': 0.1,
                'conditional_prob': 0.0,
            },
            'medium': {
                'line_range': (8, 20),
                'max_constraints': 7,
                'inversion_prob': 0.2,
                'conditional_prob': 0.1,
            },
            'hard': {
                'line_range': (15, 40),
                'max_constraints': 12,
                'inversion_prob': 0.3,
                'conditional_prob': 0.2,
            }
        }[difficulty]
        
        # 1. Structural constraints (always present)
        struct_constraints, struct_meta = self._generate_structure(
            params['line_range'][0],
            params['line_range'][1]
        )
        constraints.extend(struct_constraints)
        metadata['structure'] = struct_meta
        
        # 2. Maybe add rhyme scheme
        if self.rng.random() < 0.7:
            rhyme = self._generate_rhyme_scheme(struct_meta['total_lines'])
            if allow_inversions and self.rng.random() < params['inversion_prob']:
                rhyme = ~rhyme
                metadata['inversions_used'].append('rhyme_scheme')
            constraints.append(rhyme)
            metadata['constraint_types'].append('rhyme')
        
        # 3. Maybe add syllable constraints
        if self.rng.random() < 0.5:
            syllable = self._generate_syllable_constraint(struct_meta['total_lines'])
            if allow_inversions and self.rng.random() < params['inversion_prob']:
                syllable = ~syllable
                metadata['inversions_used'].append('syllable')
            constraints.append(syllable)
            metadata['constraint_types'].append('syllable')
        
        # 4. Maybe add title requirement
        if self.rng.random() < 0.4:
            title_required = self.rng.choice([True, False])
            title = HasTitle(required=title_required, 
                           format=self.rng.choice(list(TitleFormat)[:4]))
            constraints.append(title)
            
            # Maybe add title word count
            if title_required and self.rng.random() < 0.5:
                max_words = self.rng.randint(2, 6)
                constraints.append(TitleWordCount.at_most(max_words, title.format))
            
            metadata['constraint_types'].append('title')
        
        # 5. Maybe add acrostic
        if self.rng.random() < 0.2:
            word_length = min(struct_meta['total_lines'], self.rng.randint(3, 8))
            word = self._generate_random_word(word_length)
            acrostic = Acrostic.spelling(word)
            if allow_inversions and self.rng.random() < params['inversion_prob']:
                acrostic = ~acrostic
                metadata['inversions_used'].append('acrostic')
            constraints.append(acrostic)
            metadata['constraint_types'].append('acrostic')
        
        # 6. Maybe add word constraints
        if self.rng.random() < 0.3:
            word_constraint = self._generate_word_constraint()
            constraints.append(word_constraint)
            metadata['constraint_types'].append('word')
        
        # 7. Maybe add conditional constraint
        if allow_conditionals and self.rng.random() < params['conditional_prob']:
            cond = self._generate_conditional()
            constraints.append(cond)
            metadata['constraint_types'].append('conditional')
        
        # 8. Maybe add "exactly N of M" constraint
        if self.rng.random() < 0.15 and len(constraints) >= 3:
            subset = self.rng.sample(constraints, min(4, len(constraints)))
            n = self.rng.randint(1, len(subset) - 1)
            exactly_n = ExactlyN(subset, n)
            constraints.append(exactly_n)
            metadata['constraint_types'].append('exactly_n')
        
        # Generate name and description
        name = self._generate_form_name(metadata)
        description = self._generate_description(constraints, metadata)
        
        template = FormTemplate(
            name=name,
            description=description,
            constraints=constraints
        )
        
        return template, metadata
    
    def _generate_structure(
        self,
        min_lines: int,
        max_lines: int
    ) -> Tuple[List[Constraint], Dict]:
        """Generate structural constraints."""
        constraints = []
        
        num_stanzas = self.rng.randint(1, 6)
        strategy = self.rng.choice(['uniform', 'varied', 'final_different'])
        
        if strategy == 'uniform':
            size = self.rng.randint(2, 8)
            sizes = [size] * num_stanzas
        elif strategy == 'varied':
            sizes = [self.rng.randint(2, 6) for _ in range(num_stanzas)]
        else:
            main = self.rng.randint(3, 6)
            final = self.rng.randint(2, 4)
            sizes = [main] * (num_stanzas - 1) + [final]
        
        total = sum(sizes)
        
        constraints.append(LineCount.exactly(total))
        constraints.append(StanzaCount.exactly(num_stanzas))
        constraints.append(StanzaStructure(sizes))
        
        return constraints, {
            'total_lines': total,
            'stanza_count': num_stanzas,
            'sizes': sizes,
            'strategy': strategy
        }
    
    def _generate_rhyme_scheme(self, num_lines: int) -> RhymeScheme:
        """Generate a random rhyme scheme."""
        strategies = ['couplets', 'alternating', 'enclosed', 'random']
        strategy = self.rng.choice(strategies)
        
        if strategy == 'couplets':
            pattern = ''.join(chr(65 + i//2) for i in range(num_lines))
        elif strategy == 'alternating':
            pattern = ''.join(chr(65 + (i % 2)) for i in range(num_lines))
        elif strategy == 'enclosed' and num_lines >= 4:
            base = 'ABBA'
            pattern = (base * (num_lines // 4 + 1))[:num_lines]
        else:
            num_rhymes = self.rng.randint(2, min(6, num_lines // 2))
            pattern = ''.join(chr(65 + self.rng.randint(0, num_rhymes - 1)) 
                            for _ in range(num_lines))
        
        return RhymeScheme(pattern)
    
    def _generate_syllable_constraint(self, num_lines: int) -> Constraint:
        """Generate syllable constraint."""
        constraint_type = self.rng.choice(['per_line', 'pattern', 'total'])
        
        if constraint_type == 'per_line':
            syllables = self.rng.randint(5, 12)
            return SyllableCount.per_line_exactly(syllables)
        elif constraint_type == 'pattern' and num_lines <= 10:
            pattern = [self.rng.randint(4, 10) for _ in range(num_lines)]
            return SyllablePattern(pattern, tolerance=1)
        else:
            total = num_lines * self.rng.randint(6, 10)
            return SyllableCount(NumericBound.between(total - 10, total + 10), scope='total')
    
    def _generate_word_constraint(self) -> Constraint:
        """Generate word-related constraint."""
        constraint_type = self.rng.choice(['contains', 'not_contains', 'unique', 'word_count'])
        
        if constraint_type == 'contains':
            words = [self._generate_random_word(self.rng.randint(3, 8)) 
                    for _ in range(self.rng.randint(1, 3))]
            return ContainsWord(words, scope='poem', match_all=False)
        elif constraint_type == 'not_contains':
            common_words = ['the', 'and', 'but', 'is', 'are', 'was', 'were']
            forbidden = self.rng.sample(common_words, self.rng.randint(1, 3))
            return DoesNotContainWord(forbidden, scope='poem')
        elif constraint_type == 'unique':
            ratio = self.rng.choice([0.7, 0.8, 0.9, 1.0])
            return UniqueWords(scope='poem', min_unique_ratio=ratio)
        else:
            per_line = self.rng.randint(4, 10)
            return WordCount.per_line_at_least(per_line)
    
    def _generate_conditional(self) -> Implies:
        """Generate a conditional constraint."""
        # If title exists, then some constraint on title
        condition = HasTitle(required=True, format=TitleFormat.BRACKETED_FIRST)
        consequence = TitleWordCount.at_most(self.rng.randint(2, 5), TitleFormat.BRACKETED_FIRST)
        return Implies(condition, consequence)
    
    def _generate_random_word(self, length: int) -> str:
        """Generate a pronounceable random word."""
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvwxyz'
        
        word = ''
        use_vowel = self.rng.random() < 0.3
        
        for _ in range(length):
            if use_vowel:
                word += self.rng.choice(vowels)
            else:
                word += self.rng.choice(consonants)
            use_vowel = not use_vowel
        
        return word
    
    def _generate_form_name(self, metadata: Dict) -> str:
        """Generate a creative name for the form."""
        prefixes = ['neo-', 'pseudo-', 'inverse ', 'modified ', 'broken ', 
                   'spiral ', 'echo ', 'mirror ', 'shadow ', 'crystal ']
        roots = ['verse', 'stanza', 'lyric', 'ode', 'song', 'chant', 
                'fragment', 'sequence', 'meditation', 'reflection']
        
        name = self.rng.choice(prefixes) + self.rng.choice(roots)
        
        if metadata.get('inversions_used'):
            name = 'anti-' + name
        
        return name.title()
    
    def _generate_description(self, constraints: List[Constraint], metadata: Dict) -> str:
        """Generate description from constraints."""
        parts = [f"A {metadata['difficulty']} form with {metadata['structure']['total_lines']} lines"]
        
        if 'rhyme' in metadata.get('constraint_types', []):
            parts.append("featuring a specific rhyme scheme")
        if 'syllable' in metadata.get('constraint_types', []):
            parts.append("with syllabic constraints")
        if 'acrostic' in metadata.get('constraint_types', []):
            parts.append("incorporating an acrostic pattern")
        if metadata.get('inversions_used'):
            parts.append("using inverted/negated constraints")
        
        return ', '.join(parts) + '.'
    
    def mutate_form(
        self,
        template: FormTemplate,
        mutation_rate: float = 0.3
    ) -> FormTemplate:
        """
        Mutate an existing form by inverting, adding, or removing constraints.
        """
        new_constraints = []
        
        for c in template.constraints:
            if self.rng.random() < mutation_rate:
                # Decide mutation type
                mutation = self.rng.choice(['invert', 'modify', 'drop'])
                
                if mutation == 'invert':
                    new_constraints.append(~c)
                elif mutation == 'modify' and isinstance(c, LineCount):
                    # Modify numeric constraints
                    if hasattr(c, 'bound') and hasattr(c.bound, 'value'):
                        delta = self.rng.randint(-3, 3)
                        new_val = max(1, c.bound.value + delta)
                        new_constraints.append(LineCount.exactly(new_val))
                    else:
                        new_constraints.append(c)
                elif mutation == 'drop':
                    pass  # Don't add this constraint
                else:
                    new_constraints.append(c)
            else:
                new_constraints.append(c)
        
        return FormTemplate(
            name=f"Mutated {template.name}",
            description=f"Mutated version of {template.name}",
            constraints=new_constraints
        )
    
    def hybridize_forms(
        self,
        form1: FormTemplate,
        form2: FormTemplate
    ) -> FormTemplate:
        """
        Create a hybrid form by combining constraints from two forms.
        """
        # Take structure from form1
        struct_constraints = [c for c in form1.constraints 
                            if isinstance(c, (LineCount, StanzaCount, StanzaStructure))]
        
        # Take pattern constraints from form2
        pattern_constraints = [c for c in form2.constraints 
                              if isinstance(c, (RhymeScheme, RefrainGroup, EndWordRotation))]
        
        # Random selection from both for other constraints
        other1 = [c for c in form1.constraints if c not in struct_constraints]
        other2 = [c for c in form2.constraints if c not in pattern_constraints]
        
        other_combined = other1 + other2
        selected_other = self.rng.sample(other_combined, 
                                         min(len(other_combined), self.rng.randint(1, 4)))
        
        all_constraints = struct_constraints + pattern_constraints + selected_other
        
        return FormTemplate(
            name=f"{form1.name}-{form2.name} Hybrid",
            description=f"A hybrid combining elements of {form1.name} and {form2.name}.",
            constraints=all_constraints
        )
```

---

## 12. Complete Usage Example

python

```python
# ============================================================================
# COMPLETE DEMONSTRATION
# ============================================================================

def full_demonstration():
    """Complete demonstration of the system."""
    
    print("=" * 70)
    print("COMPOSABLE POETRY CONSTRAINT SYSTEM - FULL DEMONSTRATION")
    print("=" * 70)
    
    env = ComposablePoetryEnvironment()
    
    # ─────────────────────────────────────────────────────────────────────
    # 1. Evaluate Dylan Thomas's villanelle
    # ─────────────────────────────────────────────────────────────────────
    
    villanelle = """Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light.

Though wise men at their end know dark is right,
Because their words had forked no lightning they
Do not go gentle into that good night.

Good men, the last wave by, crying how bright
Their frail deeds might have danced in a green bay,
Rage, rage against the dying of the light.

Wild men who caught and sang the sun in flight,
And learn, too late, they grieved it on its way,
Do not go gentle into that good night.

Grave men, near death, who see with blinding sight
Blind eyes could blaze like meteors and be gay,
Rage, rage against the dying of the light.

And you, my father, there on the sad height,
Curse, bless, me now with your fierce tears, I pray.
Do not go gentle into that good night.
Rage, rage against the dying of the light."""

    print("\n" + "─" * 70)
    print("1. VILLANELLE EVALUATION")
    print("─" * 70)
    
    result, rubric, grade = env.evaluate(villanelle, 'villanelle')
    print(f"\nScore: {result.total_score:.2%}")
    print(grade.format_markdown())
    
    # ─────────────────────────────────────────────────────────────────────
    # 2. Create an "anti-sonnet"
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("2. ANTI-SONNET (Inverted Constraints)")
    print("─" * 70)
    
    anti_sonnet = env.create_custom_form(
        name="Anti-Sonnet",
        constraints=[
            LineCount.not_equal(14),           # NOT 14 lines
            LineCount.between(12, 16),         # But close
            ~RhymeScheme("ABAB CDCD EFEF GG"), # NOT Shakespearean rhyme
            ~Meter.iambic_pentameter(),        # NOT iambic pentameter
            HasTitle(required=True, format=TitleFormat.BRACKETED_FIRST),
            TitleWordCount.at_most(3, TitleFormat.BRACKETED_FIRST),
        ],
        description="A poem that deliberately violates sonnet conventions while staying close to the form."
    )
    
    print("\nINSTRUCTION:")
    print(anti_sonnet.to_instruction())
    print("\nRUBRIC:")
    print(anti_sonnet.generate_rubric().format_markdown())
    
    # ─────────────────────────────────────────────────────────────────────
    # 3. Generate novel forms
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("3. NOVEL FORM GENERATION")
    print("─" * 70)
    
    generator = NovelFormGenerator(seed=42)
    
    for difficulty in ['easy', 'medium', 'hard']:
        template, metadata = generator.generate_form(difficulty)
        print(f"\n{difficulty.upper()} FORM: {template.name}")
        print("-" * 50)
        print(template.to_instruction())
        print(f"\nMetadata: {metadata['constraint_types']}")
        if metadata.get('inversions_used'):
            print(f"Inversions: {metadata['inversions_used']}")
    
    # ─────────────────────────────────────────────────────────────────────
    # 4. Complex constraint composition
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("4. COMPLEX CONSTRAINT COMPOSITION")
    print("─" * 70)
    
    # "A poem that satisfies EXACTLY 2 of these 4 constraints"
    complex_form = env.create_custom_form(
        name="Choose Your Adventure",
        constraints=[
            LineCount.exactly(8),
            ExactlyN([
                RhymeScheme("AABBCCDD"),           # Couplets
                SyllablePattern([8]*8, tolerance=1), # 8 syllables each
                Acrostic.spelling("POETRY"),       # Spells POETRY (needs 6 lines, won't work)
                Alliteration(min_alliterative_words=3, scope='line'),
            ], n=2),
            ~ContainsWord(['love', 'heart', 'soul'], match_all=False),  # No clichés
        ],
        description="An 8-line poem satisfying exactly 2 of 4 pattern constraints, without romantic clichés."
    )
    
    print("\nINSTRUCTION:")
    print(complex_form.to_instruction())
    
    # ─────────────────────────────────────────────────────────────────────
    # 5. Conditional constraints
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("5. CONDITIONAL CONSTRAINTS")
    print("─" * 70)
    
    conditional_form = env.create_custom_form(
        name="Conditional Verse",
        constraints=[
            LineCount.between(6, 12),
            # If the poem has alliteration, it must also have internal rhyme
            Implies(
                Alliteration(min_alliterative_words=3),
                RhymeScheme("AABB")  # Simplified stand-in for internal rhyme
            ),
            # If title exists, must be exactly 2 words
            Implies(
                HasTitle(required=True, format=TitleFormat.BRACKETED_FIRST),
                TitleWordCount.exactly(2, TitleFormat.BRACKETED_FIRST)
            ),
        ],
        description="A poem with conditional requirements based on stylistic choices."
    )
    
    print("\nINSTRUCTION:")
    print(conditional_form.to_instruction())
    
    # ─────────────────────────────────────────────────────────────────────
    # 6. Form hybridization
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("6. FORM HYBRIDIZATION")
    print("─" * 70)
    
    haiku = FormLibrary.haiku()
    sonnet = FormLibrary.sonnet_shakespearean()
    
    # This won't make much sense structurally, but demonstrates the concept
    hybrid = generator.hybridize_forms(haiku, sonnet)
    
    print(f"\nHYBRID FORM: {hybrid.name}")
    print(hybrid.to_instruction())
    
    # ─────────────────────────────────────────────────────────────────────
    # 7. Training example generation
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("7. TRAINING EXAMPLE GENERATION")
    print("─" * 70)
    
    instruction, template, rubric, reward_fn = env.generate_training_example('medium')
    
    print("\nGENERATED INSTRUCTION:")
    print(instruction)
    print("\nGENERATED RUBRIC:")
    print(rubric.format_markdown()[:1000] + "...")  # Truncate for display
    
    # Test the reward function
    test_poem = "This is a test poem\nWith multiple lines\nTo see how it scores"
    score = reward_fn(test_poem)
    print(f"\nTest poem score: {score:.2%}")
    
    # ─────────────────────────────────────────────────────────────────────
    # 8. Demonstrate De Morgan inversions
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 70)
    print("8. DE MORGAN'S LAW DEMONSTRATIONS")
    print("─" * 70)
    
    a = LineCount.exactly(10)
    b = RhymeScheme("AABB")
    
    # NOT(A AND B) should equal NOT(A) OR NOT(B)
    not_a_and_b = ~(a & b)
    not_a_or_not_b = (~a) | (~b)
    
    print(f"\nOriginal A: {a.to_instruction()}")
    print(f"Original B: {b.to_instruction()}")
    print(f"\nNOT(A AND B) type: {type(not_a_and_b).__name__}")  # Should be Or
    print(f"NOT(A) OR NOT(B) type: {type(not_a_or_not_b).__name__}")  # Should be Or
    
    # NOT(A OR B) should equal NOT(A) AND NOT(B)
    not_a_or_b = ~(a | b)
    not_a_and_not_b = (~a) & (~b)
    
    print(f"\nNOT(A OR B) type: {type(not_a_or_b).__name__}")  # Should be And
    print(f"NOT(A) AND NOT(B) type: {type(not_a_and_not_b).__name__}")  # Should be And


if __name__ == "__main__":
    full_demonstration()
```

---

## Summary: Complete Constraint Taxonomy

|Category|Constraints|Invertible|Auto-Rubric|
|---|---|---|---|
|**Structural**|`LineCount`, `StanzaCount`, `StanzaStructure`, `WordCount`, `SyllableCount`, `SyllablePattern`, `LineLength`|✅ All|✅|
|**Title/Format**|`HasTitle`, `TitleWordCount`, `LineStartsWith`, `LineEndsWith`|✅ All|✅|
|**Rhyme/Sound**|`RhymeScheme`, `Rhymes`, `Alliteration`, `Meter`|✅ Via `Not`|✅|
|**Repetition**|`Repeats`, `RefrainGroup`, `EndWordRotation`|✅ Via `Not`|✅|
|**Pattern**|`Acrostic`, `ContainsWord`, `UniqueWords`|✅ All|✅|
|**Logical**|`Not`, `And`, `Or`, `ExactlyN`, `AtLeastN`, `AtMostN`, `Implies`|✅ De Morgan|✅|

This system provides:

1. **Complete expressiveness** - Can define any known poetic form
2. **Full invertibility** - Every constraint can be negated
3. **Automatic rubric generation** - From any constraint set
4. **Novel form generation** - For generalization training
5. **Composable algebra** - Build complex forms from primitives
6. **Spectrum rewards** - Fuzzy matching for RL training

give me some elliptical names for this system, also, can we reward free verse, internal rhymes, vowel sound matching etc

I love this direction—let's name this thing properly and then extend it to handle the subtler, more musical aspects of poetry that make free verse actually _work_.

<thinking> The user wants: 1. Evocative, elliptical names for the system 2. Extensions to handle: - Free verse (which has its own "rules" around rhythm, breath, line breaks, etc.) - Internal rhymes (rhymes within lines, not just at end) - Vowel sound matching (assonance) - Presumably also consonance, near-rhyme, etc.

For naming, I should think about what this system actually IS:

- A constraint algebra for poetry
- Bidirectional (generates instructions AND verifies)
- Compositional
- Handles both strict forms and free verse
- Musical/sonic analysis

Elliptical names might draw from:

- Musical terminology (counterpoint, harmonic, resonance)
- Weaving/textile (warp, weft, loom)
- Geometric/mathematical (lattice, manifold, topology)
- Natural phenomena (crystal, wave, tide)
- Linguistic/poetic (prosody, cadence, meter)
- Greek/Latin roots

Some ideas:

- Prosodia (prosody + -ia)
- Versiform (verse + form)
- Phonotaxis (sound arrangement)
- Scansion (metrical analysis)
- Cadence
- Resona (resonance)
- Tessera (mosaic piece)
- Glossa (tongue/language)
- Strophē (Greek: turn, stanza)
- Melos (Greek: song, melody)
- Harmonia

For the extensions, I need to think about:

**Free Verse Rewards:**

- Line break quality (enjambment vs end-stop, meaningful breaks)
- Rhythm variation (not metronomic but not chaotic)
- Breath units
- Visual spacing
- Image density
- Concrete vs abstract balance

**Internal Rhyme:**

- Position-agnostic rhyme detection within lines
- Leonine rhyme (middle rhymes with end)
- Cross-line internal rhymes

**Vowel Sound Matching (Assonance):**

- Phoneme extraction focusing on vowels
- Vowel sequence analysis
- Long vs short vowel patterns

**Other Sound Devices:**

- Consonance (repeated consonant sounds)
- Sibilance (s sounds)
- Plosives (p, b, t, d, k, g)
- Liquid consonants (l, r)
- Onomatopoeia detection

Let me build these out. </thinking>

# Elliptical Names & Extended Sonic Analysis

## Part 1: Names for the System

Here are names organized by the metaphorical territory they invoke:

### Musical/Sonic

|Name|Meaning/Resonance|
|---|---|
|**Melos**|Greek for "song" or "melody"—the musical phrase|
|**Resonant**|Sound that continues, reverberates, persists|
|**Harmonia**|The fitting together of disparate elements|
|**Cadenza**|The virtuosic passage; also "cadence"|
|**Tonos**|Greek for "tension" (of a string), gives us "tone"|
|**Partita**|A suite; a set of variations|

### Weaving/Structure

|Name|Meaning/Resonance|
|---|---|
|**Tessera**|A single piece of mosaic; the fragment that makes the whole|
|**Weft**|The horizontal threads; what's woven through|
|**Bobbin**|Holds the thread; the mechanism of weaving|
|**Jacquard**|The programmable loom; pattern from instruction|
|**Selvage**|The edge that prevents unraveling|

### Geometric/Abstract

|Name|Meaning/Resonance|
|---|---|
|**Lattice**|Regular structure with gaps; both constraint and freedom|
|**Manifold**|A space that locally resembles Euclidean space|
|**Trellis**|Support structure for growth|
|**Strophe**|Greek "turn"—the movement of verse|

### Natural/Elemental

|Name|Meaning/Resonance|
|---|---|
|**Breath**|The fundamental unit of spoken poetry|
|**Tide**|Pattern that isn't rigid; ebb and flow|
|**Murmur**|The low continuous sound; undersong|
|**Susurrus**|Whispering, rustling sound|

### My Favorites (with rationale)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   MELOS          - Simple, ancient, musical                     │
│                    "The melos of your verse scores 0.87"        │
│                                                                 │
│   TESSERA        - Each constraint is a tile in the mosaic     │
│                    "Compose a tessera for this form"            │
│                                                                 │
│   CADENZA        - The virtuosic constraint system              │
│                    "The cadenza detected internal rhyme"        │
│                                                                 │
│   SUSURRUS       - For the sonic analysis specifically          │
│                    "Susurrus assonance score: 0.73"             │
│                                                                 │
│   STROPHE        - Classical, refers to verse structure         │
│                    "Define a new strophe"                       │
│                                                                 │
│   JACQUARD       - The programmable loom metaphor               │
│                    "Weave constraints on the Jacquard"          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Combined/Portmanteau Options

|Name|Components|Feel|
|---|---|---|
|**Prosodex**|Prosody + index/codex|Technical, systematic|
|**Versecraft**|Verse + craft|Practical, artisanal|
|**Phonotaxis**|Phoneme + arrangement|Linguistic, precise|
|**Meloform**|Melos + form|Musical structure|
|**Sonorant**|Sonorous + -ant|Sounds that resonate|
|**Voxlattice**|Voice + lattice|The grid of sound|

---

## Part 2: Extended Sonic Analysis

Now let's build the machinery for free verse, internal rhyme, assonance, consonance, and other sonic textures.

python

```python
# ============================================================================
# PHONETIC FOUNDATION
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum, auto
import re
import math

class VowelType(Enum):
    """Classification of vowel sounds."""
    LONG_A = auto()    # "fate", "rain"
    SHORT_A = auto()   # "cat", "hat"
    LONG_E = auto()    # "feet", "sea"
    SHORT_E = auto()   # "bed", "head"
    LONG_I = auto()    # "fight", "my"
    SHORT_I = auto()   # "bit", "gym"
    LONG_O = auto()    # "coat", "flow"
    SHORT_O = auto()   # "hot", "cot"
    LONG_U = auto()    # "flute", "blue"
    SHORT_U = auto()   # "cup", "mud"
    SCHWA = auto()     # "about", unstressed
    DIPHTHONG_OI = auto()  # "boy", "coin"
    DIPHTHONG_OW = auto()  # "cow", "loud"

class ConsonantType(Enum):
    """Classification of consonant sounds."""
    # Manner of articulation
    PLOSIVE = auto()       # p, b, t, d, k, g
    FRICATIVE = auto()     # f, v, s, z, sh, th
    AFFRICATE = auto()     # ch, j
    NASAL = auto()         # m, n, ng
    LIQUID = auto()        # l, r
    GLIDE = auto()         # w, y
    SIBILANT = auto()      # s, z, sh, zh

# CMU Phoneme to type mappings
VOWEL_PHONEMES = {
    'AA': VowelType.SHORT_O,    # "odd"
    'AE': VowelType.SHORT_A,    # "at"
    'AH': VowelType.SCHWA,      # "hut" / unstressed
    'AO': VowelType.LONG_O,     # "ought"
    'AW': VowelType.DIPHTHONG_OW,  # "cow"
    'AY': VowelType.LONG_I,     # "hide"
    'EH': VowelType.SHORT_E,    # "ed"
    'ER': VowelType.SCHWA,      # "hurt" (r-colored)
    'EY': VowelType.LONG_A,     # "ate"
    'IH': VowelType.SHORT_I,    # "it"
    'IY': VowelType.LONG_E,     # "eat"
    'OW': VowelType.LONG_O,     # "oat"
    'OY': VowelType.DIPHTHONG_OI,  # "toy"
    'UH': VowelType.SHORT_U,    # "hood"
    'UW': VowelType.LONG_U,     # "two"
}

CONSONANT_PHONEMES = {
    'P': ConsonantType.PLOSIVE,
    'B': ConsonantType.PLOSIVE,
    'T': ConsonantType.PLOSIVE,
    'D': ConsonantType.PLOSIVE,
    'K': ConsonantType.PLOSIVE,
    'G': ConsonantType.PLOSIVE,
    'F': ConsonantType.FRICATIVE,
    'V': ConsonantType.FRICATIVE,
    'TH': ConsonantType.FRICATIVE,
    'DH': ConsonantType.FRICATIVE,
    'S': ConsonantType.SIBILANT,
    'Z': ConsonantType.SIBILANT,
    'SH': ConsonantType.SIBILANT,
    'ZH': ConsonantType.SIBILANT,
    'CH': ConsonantType.AFFRICATE,
    'JH': ConsonantType.AFFRICATE,
    'M': ConsonantType.NASAL,
    'N': ConsonantType.NASAL,
    'NG': ConsonantType.NASAL,
    'L': ConsonantType.LIQUID,
    'R': ConsonantType.LIQUID,
    'W': ConsonantType.GLIDE,
    'Y': ConsonantType.GLIDE,
    'HH': ConsonantType.FRICATIVE,
}


@dataclass
class PhoneticWord:
    """A word with its phonetic analysis."""
    text: str
    phonemes: List[str]
    vowels: List[Tuple[str, int]]  # (phoneme, stress_level)
    consonants: List[str]
    syllable_count: int
    stress_pattern: str  # e.g., "10" for iamb, "01" for trochee
    
    @classmethod
    def from_cmu(cls, word: str, cmu_dict: Dict) -> Optional['PhoneticWord']:
        """Create from CMU dictionary lookup."""
        word_lower = word.lower().strip('.,!?;:\'"')
        
        if word_lower not in cmu_dict:
            return None
        
        phonemes = cmu_dict[word_lower][0]
        
        vowels = []
        consonants = []
        stress_pattern = ""
        
        for p in phonemes:
            # Check for stress marker (digit)
            stress = None
            base_phoneme = p
            for char in p:
                if char.isdigit():
                    stress = int(char)
                    base_phoneme = p.replace(char, '')
            
            if base_phoneme in VOWEL_PHONEMES or any(c.isdigit() for c in p):
                vowels.append((base_phoneme, stress if stress is not None else 0))
                stress_pattern += str(stress) if stress else "0"
            elif base_phoneme in CONSONANT_PHONEMES:
                consonants.append(base_phoneme)
        
        return cls(
            text=word,
            phonemes=phonemes,
            vowels=vowels,
            consonants=consonants,
            syllable_count=len(vowels),
            stress_pattern=stress_pattern
        )


@dataclass
class PhoneticLine:
    """A line with full phonetic analysis."""
    text: str
    words: List[PhoneticWord]
    all_vowels: List[Tuple[str, int, int]]  # (phoneme, stress, word_index)
    all_consonants: List[Tuple[str, int]]   # (phoneme, word_index)
    stress_contour: str
    
    @classmethod
    def analyze(cls, line: str, cmu_dict: Dict) -> 'PhoneticLine':
        """Analyze a line phonetically."""
        raw_words = re.findall(r"[a-zA-Z']+", line)
        words = []
        all_vowels = []
        all_consonants = []
        stress_contour = ""
        
        for i, word in enumerate(raw_words):
            pw = PhoneticWord.from_cmu(word, cmu_dict)
            if pw:
                words.append(pw)
                for v, s in pw.vowels:
                    all_vowels.append((v, s, i))
                for c in pw.consonants:
                    all_consonants.append((c, i))
                stress_contour += pw.stress_pattern
            else:
                # Fallback for unknown words
                words.append(PhoneticWord(
                    text=word,
                    phonemes=[],
                    vowels=[],
                    consonants=[],
                    syllable_count=cls._estimate_syllables(word),
                    stress_pattern=""
                ))
        
        return cls(
            text=line,
            words=words,
            all_vowels=all_vowels,
            all_consonants=all_consonants,
            stress_contour=stress_contour
        )
    
    @staticmethod
    def _estimate_syllables(word: str) -> int:
        """Fallback syllable estimation."""
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in word.lower():
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)
```

---

## Part 3: Assonance, Consonance, and Internal Rhyme

python

```python
# ============================================================================
# SONIC TEXTURE CONSTRAINTS
# ============================================================================

@dataclass
class Assonance(Constraint):
    """
    Repeated vowel sounds within or across lines.
    
    Measures the density and patterns of vowel sound repetition.
    """
    min_repetitions: int = 3          # Minimum repeated vowel sounds
    scope: Literal['line', 'stanza', 'poem'] = 'line'
    vowel_types: Optional[List[VowelType]] = None  # Specific vowels to look for
    
    @property
    def name(self) -> str:
        return f"assonance_{self.scope}_{self.min_repetitions}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if self.vowel_types:
            vowel_desc = ", ".join(v.name.lower().replace('_', ' ') for v in self.vowel_types)
            return f"Use assonance with {vowel_desc} vowel sounds (at least {self.min_repetitions} repetitions per {self.scope})."
        return f"Each {self.scope} should feature assonance (repeated vowel sounds, at least {self.min_repetitions} instances of the same vowel sound)."
    
    def _analyze_assonance(self, phonetic_line: PhoneticLine) -> Dict[str, int]:
        """Count vowel sound occurrences."""
        vowel_counts: Dict[str, int] = {}
        
        for vowel, stress, _ in phonetic_line.all_vowels:
            # Normalize by removing stress markers
            base_vowel = vowel.rstrip('012')
            vowel_counts[base_vowel] = vowel_counts.get(base_vowel, 0) + 1
        
        return vowel_counts
    
    def _compute_assonance_score(self, vowel_counts: Dict[str, int]) -> Tuple[float, Dict]:
        """Compute assonance score from vowel counts."""
        if not vowel_counts:
            return 0.0, {}
        
        # Find the most repeated vowel
        max_vowel = max(vowel_counts, key=vowel_counts.get)
        max_count = vowel_counts[max_vowel]
        
        # Score based on how many meet threshold
        meets_threshold = sum(1 for c in vowel_counts.values() if c >= self.min_repetitions)
        
        if max_count >= self.min_repetitions:
            # Base score from reaching threshold
            base_score = 1.0
            # Bonus for exceeding threshold
            bonus = min(0.2, (max_count - self.min_repetitions) * 0.05)
            score = min(1.0, base_score + bonus)
        else:
            # Partial credit
            score = max_count / self.min_repetitions
        
        return score, {
            'most_common_vowel': max_vowel,
            'max_count': max_count,
            'all_counts': vowel_counts
        }
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        # Analyze each line
        phonetic_lines = [PhoneticLine.analyze(line, ctx.cmu_dict) for line in ctx.structure.lines]
        
        if self.scope == 'line':
            scores = []
            details_list = []
            for pl in phonetic_lines:
                vowel_counts = self._analyze_assonance(pl)
                score, details = self._compute_assonance_score(vowel_counts)
                scores.append(score)
                details_list.append(details)
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            passed = all(s >= 0.7 for s in scores)
            
        elif self.scope == 'stanza':
            scores = []
            for stanza in ctx.structure.stanzas:
                combined_counts: Dict[str, int] = {}
                for line in stanza:
                    pl = PhoneticLine.analyze(line, ctx.cmu_dict)
                    for vowel, _, _ in pl.all_vowels:
                        base = vowel.rstrip('012')
                        combined_counts[base] = combined_counts.get(base, 0) + 1
                score, _ = self._compute_assonance_score(combined_counts)
                scores.append(score)
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            passed = avg_score >= 0.7
            details_list = {'stanza_scores': scores}
            
        else:  # poem
            combined_counts: Dict[str, int] = {}
            for pl in phonetic_lines:
                for vowel, _, _ in pl.all_vowels:
                    base = vowel.rstrip('012')
                    combined_counts[base] = combined_counts.get(base, 0) + 1
            
            avg_score, details_list = self._compute_assonance_score(combined_counts)
            passed = avg_score >= 0.7
        
        if mode == MatchMode.EXACT:
            avg_score = 1.0 if passed else 0.0
        
        return VerificationResult(
            score=avg_score,
            passed=passed,
            constraint_name=self.name,
            details={'scope': self.scope, 'analysis': details_list}
        )
    
    def invert(self) -> 'Not':
        return Not(self)
    
    def rubric_name(self) -> str:
        return f"Assonance ({self.scope})"
    
    def rubric_description(self) -> str:
        return f"Effective use of repeated vowel sounds within each {self.scope}."


@dataclass
class Consonance(Constraint):
    """
    Repeated consonant sounds (not just at word beginnings—that's alliteration).
    """
    min_repetitions: int = 3
    scope: Literal['line', 'stanza', 'poem'] = 'line'
    consonant_types: Optional[List[ConsonantType]] = None
    
    @property
    def name(self) -> str:
        return f"consonance_{self.scope}_{self.min_repetitions}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if self.consonant_types:
            type_names = [t.name.lower() for t in self.consonant_types]
            return f"Use consonance with {', '.join(type_names)} sounds (at least {self.min_repetitions} per {self.scope})."
        return f"Each {self.scope} should feature consonance (repeated consonant sounds throughout, not just at word beginnings)."
    
    def _analyze_consonance(self, phonetic_line: PhoneticLine) -> Dict[str, int]:
        """Count consonant sound occurrences."""
        consonant_counts: Dict[str, int] = {}
        
        for consonant, _ in phonetic_line.all_consonants:
            consonant_counts[consonant] = consonant_counts.get(consonant, 0) + 1
        
        return consonant_counts
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        phonetic_lines = [PhoneticLine.analyze(line, ctx.cmu_dict) for line in ctx.structure.lines]
        
        if self.scope == 'line':
            scores = []
            for pl in phonetic_lines:
                counts = self._analyze_consonance(pl)
                if counts:
                    max_count = max(counts.values())
                    score = min(1.0, max_count / self.min_repetitions)
                else:
                    score = 0.0
                scores.append(score)
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            passed = avg_score >= 0.7
            
        else:  # stanza or poem
            # Aggregate across scope
            combined: Dict[str, int] = {}
            lines_to_check = ctx.structure.lines if self.scope == 'poem' else []
            
            if self.scope == 'stanza':
                scores = []
                for stanza in ctx.structure.stanzas:
                    stanza_counts: Dict[str, int] = {}
                    for line in stanza:
                        pl = PhoneticLine.analyze(line, ctx.cmu_dict)
                        for c, _ in pl.all_consonants:
                            stanza_counts[c] = stanza_counts.get(c, 0) + 1
                    if stanza_counts:
                        max_count = max(stanza_counts.values())
                        scores.append(min(1.0, max_count / self.min_repetitions))
                    else:
                        scores.append(0.0)
                avg_score = sum(scores) / len(scores) if scores else 0.0
            else:
                for pl in phonetic_lines:
                    for c, _ in pl.all_consonants:
                        combined[c] = combined.get(c, 0) + 1
                if combined:
                    max_count = max(combined.values())
                    avg_score = min(1.0, max_count / self.min_repetitions)
                else:
                    avg_score = 0.0
            
            passed = avg_score >= 0.7
        
        if mode == MatchMode.EXACT:
            avg_score = 1.0 if passed else 0.0
        
        return VerificationResult(
            score=avg_score,
            passed=passed,
            constraint_name=self.name,
            details={'scope': self.scope}
        )
    
    def invert(self) -> 'Not':
        return Not(self)


@dataclass
class Sibilance(Constraint):
    """
    Specific consonance with 's', 'z', 'sh' sounds.
    
    Creates a hissing, whispering quality.
    """
    min_per_line: int = 2
    max_per_line: Optional[int] = None  # Can constrain to avoid excess
    
    SIBILANT_PHONEMES = {'S', 'Z', 'SH', 'ZH'}
    
    @property
    def name(self) -> str:
        return f"sibilance_{self.min_per_line}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        base = f"Each line should feature sibilant sounds (s, z, sh) at least {self.min_per_line} times"
        if self.max_per_line:
            base += f" but no more than {self.max_per_line} times"
        return base + "."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        scores = []
        
        for line in ctx.structure.lines:
            pl = PhoneticLine.analyze(line, ctx.cmu_dict)
            sibilant_count = sum(1 for c, _ in pl.all_consonants if c in self.SIBILANT_PHONEMES)
            
            # Also count in raw text for fallback
            text_count = sum(1 for c in line.lower() if c in 'szSZ')
            text_count += line.lower().count('sh')
            
            count = max(sibilant_count, text_count)
            
            if count >= self.min_per_line:
                score = 1.0
                if self.max_per_line and count > self.max_per_line:
                    score = max(0.5, 1.0 - (count - self.max_per_line) * 0.1)
            else:
                score = count / self.min_per_line
            
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        passed = avg_score >= 0.7
        
        if mode == MatchMode.EXACT:
            avg_score = 1.0 if passed else 0.0
        
        return VerificationResult(
            score=avg_score,
            passed=passed,
            constraint_name=self.name,
            details={'line_scores': scores}
        )
    
    def invert(self) -> 'Not':
        return Not(self)


@dataclass
class InternalRhyme(Constraint):
    """
    Rhymes occurring within a line (not just at line endings).
    
    Includes:
    - Leonine rhyme: middle of line rhymes with end
    - Cross-rhyme: words within line rhyme with each other
    """
    min_instances: int = 1  # Per line
    scope: Literal['line', 'poem'] = 'line'
    rhyme_threshold: float = 0.7  # How similar sounds must be
    
    @property
    def name(self) -> str:
        return f"internal_rhyme_{self.scope}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if self.scope == 'line':
            return f"Each line should contain at least {self.min_instances} internal rhyme(s) (words within the line that rhyme with each other)."
        return f"The poem should feature internal rhymes throughout."
    
    def _find_internal_rhymes(self, phonetic_line: PhoneticLine, cmu_dict: Dict) -> List[Tuple[str, str, float]]:
        """Find rhyming word pairs within a line."""
        rhymes_found = []
        words = phonetic_line.words
        
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words[i+1:], i+1):
                if not word1.phonemes or not word2.phonemes:
                    continue
                
                # Check if they rhyme
                score = self._rhyme_score(word1, word2)
                if score >= self.rhyme_threshold:
                    rhymes_found.append((word1.text, word2.text, score))
        
        return rhymes_found
    
    def _rhyme_score(self, word1: PhoneticWord, word2: PhoneticWord) -> float:
        """Compute rhyme similarity between two words."""
        if not word1.vowels or not word2.vowels:
            return 0.0
        
        # Get rhyme portions (from last stressed vowel onward)
        def get_rhyme_portion(pw: PhoneticWord) -> List[str]:
            # Find last stressed vowel
            last_stressed_idx = -1
            for i, (v, s) in enumerate(pw.vowels):
                if s in (1, 2):
                    last_stressed_idx = i
            
            if last_stressed_idx == -1:
                last_stressed_idx = len(pw.vowels) - 1
            
            # Get phonemes from that point
            # This is approximate - proper implementation would track phoneme positions
            return [v for v, s in pw.vowels[last_stressed_idx:]]
        
        rhyme1 = get_rhyme_portion(word1)
        rhyme2 = get_rhyme_portion(word2)
        
        if not rhyme1 or not rhyme2:
            return 0.0
        
        # Compare
        matches = sum(1 for a, b in zip(rhyme1, rhyme2) if a == b)
        max_len = max(len(rhyme1), len(rhyme2))
        
        return matches / max_len if max_len > 0 else 0.0
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        phonetic_lines = [PhoneticLine.analyze(line, ctx.cmu_dict) for line in ctx.structure.lines]
        
        all_rhymes = []
        line_scores = []
        
        for pl in phonetic_lines:
            rhymes = self._find_internal_rhymes(pl, ctx.cmu_dict)
            all_rhymes.extend(rhymes)
            
            if len(rhymes) >= self.min_instances:
                line_scores.append(1.0)
            else:
                line_scores.append(len(rhymes) / self.min_instances if self.min_instances > 0 else 0.0)
        
        if self.scope == 'line':
            avg_score = sum(line_scores) / len(line_scores) if line_scores else 0.0
            passed = all(s >= 0.7 for s in line_scores)
        else:
            # Poem-wide: at least some internal rhymes exist
            avg_score = min(1.0, len(all_rhymes) / (len(ctx.structure.lines) * self.min_instances))
            passed = len(all_rhymes) >= self.min_instances
        
        if mode == MatchMode.EXACT:
            avg_score = 1.0 if passed else 0.0
        
        return VerificationResult(
            score=avg_score,
            passed=passed,
            constraint_name=self.name,
            details={
                'rhymes_found': [(r[0], r[1]) for r in all_rhymes[:10]],  # Limit for readability
                'total_count': len(all_rhymes)
            }
        )
    
    def invert(self) -> 'Not':
        return Not(self)
    
    def rubric_name(self) -> str:
        return "Internal Rhyme"


@dataclass
class NearRhyme(Constraint):
    """
    Slant rhyme / half rhyme / near rhyme detection.
    
    Words that almost rhyme but not perfectly—common in modern poetry.
    Examples: "love/move", "eyes/light", "soul/all"
    """
    positions: List[Tuple[int, int]]  # Pairs of line indices that should near-rhyme
    min_similarity: float = 0.5       # Lower than perfect rhyme
    max_similarity: float = 0.9       # But not too perfect
    
    @property
    def name(self) -> str:
        return f"near_rhyme_{len(self.positions)}_pairs"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        return f"Use slant rhyme (near rhyme) instead of perfect rhyme between specified line pairs."
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        scores = []
        
        for pos1, pos2 in self.positions:
            if pos1 >= len(ctx.end_words) or pos2 >= len(ctx.end_words):
                scores.append(0.0)
                continue
            
            word1, word2 = ctx.end_words[pos1], ctx.end_words[pos2]
            similarity = phoneme_rhyme_score(word1, word2, ctx.cmu_dict)
            
            # Near rhyme: not too similar, not too different
            if self.min_similarity <= similarity <= self.max_similarity:
                scores.append(1.0)
            elif similarity > self.max_similarity:
                # Too perfect—penalize slightly
                scores.append(0.7)
            else:
                # Too different
                scores.append(similarity / self.min_similarity)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        passed = avg_score >= 0.7
        
        return VerificationResult(
            score=avg_score,
            passed=passed,
            constraint_name=self.name,
            details={'pair_scores': scores}
        )
    
    def invert(self) -> 'Not':
        return Not(self)
```

---

## Part 4: Free Verse Constraints

python

```python
# ============================================================================
# FREE VERSE CONSTRAINTS
# ============================================================================

@dataclass
class Enjambment(Constraint):
    """
    Line breaks that occur mid-phrase/mid-sentence.
    
    Measures the "run-on" quality of lines—syntactic units spanning line breaks.
    """
    min_enjambed_lines: float = 0.3  # At least 30% of lines should enjamb
    max_enjambed_lines: float = 0.8  # But not all—some end-stops provide rest
    
    # Words that suggest continuation
    CONTINUATION_WORDS = {
        'the', 'a', 'an', 'and', 'but', 'or', 'of', 'to', 'in', 'for',
        'with', 'by', 'from', 'that', 'which', 'who', 'whom', 'whose',
        'as', 'if', 'when', 'where', 'while', 'because', 'although',
        'into', 'onto', 'upon', 'through', 'between', 'among'
    }
    
    # Punctuation that suggests end-stop
    END_STOP_PUNCT = {'.', '!', '?', ';', ':'}
    
    @property
    def name(self) -> str:
        return f"enjambment_{self.min_enjambed_lines}_{self.max_enjambed_lines}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        min_pct = int(self.min_enjambed_lines * 100)
        max_pct = int(self.max_enjambed_lines * 100)
        return f"Use enjambment (run-on lines) in {min_pct}%-{max_pct}% of lines—let phrases flow across line breaks."
    
    def _is_enjambed(self, line: str, next_line: Optional[str]) -> Tuple[bool, float]:
        """
        Determine if a line is enjambed (runs into the next).
        Returns (is_enjambed, confidence).
        """
        line = line.strip()
        
        if not line:
            return False, 0.0
        
        # Check for end-stop punctuation
        if line[-1] in self.END_STOP_PUNCT:
            return False, 0.9
        
        # Check last word
        words = line.split()
        if not words:
            return False, 0.0
        
        last_word = words[-1].lower().strip('.,!?;:\'"')
        
        # Strong enjambment indicators
        if last_word in self.CONTINUATION_WORDS:
            return True, 0.95
        
        # Check if next line starts with lowercase (suggests continuation)
        if next_line:
            next_line = next_line.strip()
            if next_line and next_line[0].islower():
                return True, 0.8
        
        # No punctuation at end suggests weak enjambment
        if line[-1].isalpha():
            return True, 0.6
        
        return False, 0.5
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        lines = ctx.structure.lines
        enjambed_count = 0
        enjambment_details = []
        
        for i, line in enumerate(lines[:-1]):  # Can't check last line
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            is_enj, conf = self._is_enjambed(line, next_line)
            enjambment_details.append({'line': i, 'enjambed': is_enj, 'confidence': conf})
            if is_enj:
                enjambed_count += 1
        
        total_checkable = len(lines) - 1
        if total_checkable == 0:
            return VerificationResult(score=0.5, passed=True, constraint_name=self.name)
        
        enjambment_ratio = enjambed_count / total_checkable
        
        # Score based on being in the target range
        if self.min_enjambed_lines <= enjambment_ratio <= self.max_enjambed_lines:
            score = 1.0
        elif enjambment_ratio < self.min_enjambed_lines:
            score = enjambment_ratio / self.min_enjambed_lines
        else:  # Too much enjambment
            excess = enjambment_ratio - self.max_enjambed_lines
            score = max(0.5, 1.0 - excess * 2)
        
        passed = self.min_enjambed_lines <= enjambment_ratio <= self.max_enjambed_lines
        
        return VerificationResult(
            score=score,
            passed=passed,
            constraint_name=self.name,
            details={
                'enjambment_ratio': enjambment_ratio,
                'enjambed_lines': enjambed_count,
                'details': enjambment_details[:5]  # Sample
            }
        )
    
    def invert(self) -> 'Enjambment':
        # Invert by swapping min/max expectations
        return Enjambment(
            min_enjambed_lines=self.max_enjambed_lines,
            max_enjambed_lines=1.0  # Require lots of enjambment
        )
    
    def rubric_name(self) -> str:
        return "Enjambment"


@dataclass
class LineBreakQuality(Constraint):
    """
    Evaluates whether line breaks occur at meaningful/interesting points.
    
    Good line breaks often:
    - Create tension/surprise
    - Emphasize important words
    - Create double meanings
    - Follow breath patterns
    """
    
    # Words that are STRONG to end on (nouns, verbs, adjectives)
    STRONG_ENDINGS = {'POS_NOUN', 'POS_VERB', 'POS_ADJ'}
    
    # Words that are WEAK to end on (articles, prepositions—unless intentional)
    WEAK_ENDINGS = {'the', 'a', 'an', 'of', 'to', 'in', 'for', 'and', 'but', 'or', 'is', 'are', 'was', 'were'}
    
    prefer_strong_endings: bool = True
    allow_weak_for_effect: bool = True  # Weak endings can be powerful if intentional
    
    @property
    def name(self) -> str:
        return "line_break_quality"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        if self.prefer_strong_endings:
            return "Place line breaks at meaningful points—end lines on strong words (nouns, verbs, images) when possible."
        return "Consider the effect of each line break. Weak endings can create tension if used intentionally."
    
    def _score_line_ending(self, line: str) -> Tuple[float, str]:
        """Score the quality of a line's ending."""
        line = line.strip()
        if not line:
            return 0.5, "empty"
        
        words = line.split()
        if not words:
            return 0.5, "empty"
        
        last_word = words[-1].lower().strip('.,!?;:\'"')
        
        # Check for weak endings
        if last_word in self.WEAK_ENDINGS:
            if self.allow_weak_for_effect:
                return 0.6, f"weak_word:{last_word}"
            return 0.3, f"weak_word:{last_word}"
        
        # Check for punctuation
        if line[-1] in '.!?':
            return 0.8, "end_stop"
        
        # Assume content words are strong
        if len(last_word) > 3:  # Longer words tend to be content words
            return 1.0, "strong_word"
        
        return 0.7, "neutral"
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        scores = []
        details = []
        
        for i, line in enumerate(ctx.structure.lines):
            score, reason = self._score_line_ending(line)
            scores.append(score)
            details.append({'line': i, 'score': score, 'reason': reason})
        
        avg_score = sum(scores) / len(scores) if scores else 0.5
        passed = avg_score >= 0.7
        
        return VerificationResult(
            score=avg_score,
            passed=passed,
            constraint_name=self.name,
            details={'line_details': details}
        )
    
    def invert(self) -> 'LineBreakQuality':
        return LineBreakQuality(
            prefer_strong_endings=not self.prefer_strong_endings,
            allow_weak_for_effect=self.allow_weak_for_effect
        )


@dataclass
class RhythmVariation(Constraint):
    """
    Free verse shouldn't be metronomic, but shouldn't be random either.
    
    Measures the variation in line lengths and rhythms—looking for
    purposeful variation rather than strict regularity or chaos.
    """
    min_variation: float = 0.15   # Some variation required
    max_variation: float = 0.6    # But not too chaotic
    metric: Literal['syllables', 'words', 'stresses'] = 'syllables'
    
    @property
    def name(self) -> str:
        return f"rhythm_variation_{self.metric}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        return f"Vary line lengths naturally—neither perfectly regular nor chaotically random. Aim for organic rhythm."
    
    def _compute_variation(self, values: List[int]) -> float:
        """Compute coefficient of variation."""
        if not values or len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance)
        
        return std_dev / mean  # Coefficient of variation
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        if self.metric == 'syllables':
            values = []
            for line in ctx.structure.lines:
                pl = PhoneticLine.analyze(line, ctx.cmu_dict)
                syllables = sum(w.syllable_count for w in pl.words)
                values.append(syllables)
        elif self.metric == 'words':
            values = [len(line.split()) for line in ctx.structure.lines]
        else:  # stresses
            values = []
            for line in ctx.structure.lines:
                pl = PhoneticLine.analyze(line, ctx.cmu_dict)
                stresses = sum(1 for v, s, _ in pl.all_vowels if s in (1, 2))
                values.append(stresses)
        
        variation = self._compute_variation(values)
        
        # Score based on being in target range
        if self.min_variation <= variation <= self.max_variation:
            score = 1.0
        elif variation < self.min_variation:
            # Too regular
            score = variation / self.min_variation if self.min_variation > 0 else 0.0
        else:
            # Too chaotic
            excess = variation - self.max_variation
            score = max(0.3, 1.0 - excess)
        
        passed = self.min_variation <= variation <= self.max_variation
        
        return VerificationResult(
            score=score,
            passed=passed,
            constraint_name=self.name,
            details={
                'variation': variation,
                'values': values,
                'metric': self.metric
            }
        )
    
    def invert(self) -> 'Not':
        return Not(self)
    
    def rubric_name(self) -> str:
        return "Rhythmic Variation"


@dataclass
class ImageDensity(Constraint):
    """
    Measures the density of concrete, sensory imagery.
    
    Good free verse often has high image density—concrete nouns,
    sensory verbs, specific details.
    """
    min_images_per_line: float = 0.5  # On average
    
    # Simplified image word detection
    # In production, use NLP/word embeddings
    CONCRETE_INDICATORS = {
        # Colors
        'red', 'blue', 'green', 'yellow', 'white', 'black', 'gold', 'silver',
        # Nature
        'sun', 'moon', 'star', 'sky', 'cloud', 'rain', 'snow', 'wind',
        'tree', 'leaf', 'flower', 'grass', 'stone', 'rock', 'river', 'sea',
        'mountain', 'hill', 'valley', 'forest', 'field', 'garden',
        # Body
        'eye', 'hand', 'heart', 'face', 'skin', 'bone', 'blood', 'breath',
        # Animals
        'bird', 'fish', 'dog', 'cat', 'horse', 'wolf', 'deer', 'bee',
        # Objects
        'door', 'window', 'wall', 'floor', 'roof', 'chair', 'table', 'bed',
        'glass', 'mirror', 'lamp', 'candle', 'flame', 'smoke', 'ash',
        # Sensory verbs
        'see', 'hear', 'touch', 'taste', 'smell', 'feel', 'burn', 'freeze',
        'shine', 'glow', 'gleam', 'flash', 'whisper', 'roar', 'sing',
    }
    
    @property
    def name(self) -> str:
        return f"image_density_{self.min_images_per_line}"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        return f"Use concrete, sensory imagery throughout. Average at least {self.min_images_per_line} image-words per line."
    
    def _count_images(self, line: str) -> int:
        """Count concrete/image words in a line."""
        words = re.findall(r"[a-zA-Z']+", line.lower())
        return sum(1 for w in words if w in self.CONCRETE_INDICATORS)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        image_counts = [self._count_images(line) for line in ctx.structure.lines]
        total_images = sum(image_counts)
        num_lines = len(ctx.structure.lines)
        
        if num_lines == 0:
            return VerificationResult(score=0.0, passed=False, constraint_name=self.name)
        
        images_per_line = total_images / num_lines
        
        if images_per_line >= self.min_images_per_line:
            score = 1.0
        else:
            score = images_per_line / self.min_images_per_line
        
        passed = images_per_line >= self.min_images_per_line
        
        return VerificationResult(
            score=score,
            passed=passed,
            constraint_name=self.name,
            details={
                'total_images': total_images,
                'images_per_line': images_per_line,
                'line_counts': image_counts
            }
        )
    
    def invert(self) -> 'Not':
        return Not(self)
    
    def rubric_name(self) -> str:
        return "Image Density"


@dataclass
class FreeVerseForm(Constraint):
    """
    Composite constraint for "good free verse."
    
    Free verse isn't formless—it has its own internal logic.
    This combines multiple free verse quality metrics.
    """
    require_enjambment: bool = True
    require_rhythm_variation: bool = True
    require_image_density: bool = True
    require_line_break_quality: bool = True
    require_sonic_texture: bool = True  # Assonance/consonance
    
    @property
    def name(self) -> str:
        return "free_verse_form"
    
    def to_instruction(self, context: Optional[Dict] = None) -> str:
        instructions = ["Write in free verse with attention to:"]
        
        if self.require_enjambment:
            instructions.append("- Meaningful enjambment (let phrases flow across line breaks)")
        if self.require_rhythm_variation:
            instructions.append("- Rhythmic variation (neither metronomic nor chaotic)")
        if self.require_image_density:
            instructions.append("- Concrete imagery (specific, sensory details)")
        if self.require_line_break_quality:
            instructions.append("- Intentional line breaks (end on strong words)")
        if self.require_sonic_texture:
            instructions.append("- Sonic texture (assonance, consonance, internal music)")
        
        return "\n".join(instructions)
    
    def verify(self, ctx: ConstraintContext, mode: MatchMode = MatchMode.FUZZY) -> VerificationResult:
        sub_results = []
        
        if self.require_enjambment:
            sub_results.append(Enjambment().verify(ctx, mode))
        
        if self.require_rhythm_variation:
            sub_results.append(RhythmVariation().verify(ctx, mode))
        
        if self.require_image_density:
            sub_results.append(ImageDensity().verify(ctx, mode))
        
        if self.require_line_break_quality:
            sub_results.append(LineBreakQuality().verify(ctx, mode))
        
        if self.require_sonic_texture:
            # Combine assonance and consonance
            asson = Assonance(min_repetitions=2, scope='poem').verify(ctx, mode)
            cons = Consonance(min_repetitions=2, scope='poem').verify(ctx, mode)
            sonic_score = (asson.score + cons.score) / 2
            sub_results.append(VerificationResult(
                score=sonic_score,
                passed=sonic_score >= 0.5,
                constraint_name='sonic_texture',
                sub_results=[asson, cons]
            ))
        
        if not sub_results:
            return VerificationResult(score=1.0, passed=True, constraint_name=self.name)
        
        avg_score = sum(r.score for r in sub_results) / len(sub_results)
        passed = all(r.passed for r in sub_results)
        
        return VerificationResult(
            score=avg_score,
            passed=passed,
            constraint_name=self.name,
            sub_results=sub_results
        )
    
    def invert(self) -> 'Not':
        return Not(self)
    
    def rubric_name(self) -> str:
        return "Free Verse Quality"
    
    def rubric_description(self) -> str:
        return "Overall quality of free verse technique including rhythm, imagery, line breaks, and sound."
```

---

## Part 5: Updated Form Library with Free Verse and Sonic Constraints

python

```python
# ============================================================================
# EXTENDED FORM LIBRARY
# ============================================================================

class ExtendedFormLibrary(FormLibrary):
    """Extended library including free verse and sonic forms."""
    
    @staticmethod
    def free_verse_lyric() -> FormTemplate:
        """
        Contemporary free verse lyric poem.
        """
        return FormTemplate(
            name="Free Verse Lyric",
            description="A contemporary lyric poem in free verse with attention to imagery and sound.",
            constraints=[
                LineCount.between(12, 30),
                FreeVerseForm(
                    require_enjambment=True,
                    require_rhythm_variation=True,
                    require_image_density=True,
                    require_line_break_quality=True,
                    require_sonic_texture=True
                ),
            ],
            weights={
                'line_count': 0.1,
                'free_verse_form': 0.9
            }
        )
    
    @staticmethod
    def prose_poem() -> FormTemplate:
        """
        Prose poem—poetry without line breaks.
        """
        return FormTemplate(
            name="Prose Poem",
            description="A poem written in prose paragraphs, emphasizing rhythm and imagery over line breaks.",
            constraints=[
                StanzaCount.exactly(1),  # Single block
                ~Enjambment(),  # No enjambment (no line breaks to enjamb)
                ImageDensity(min_images_per_line=0.8),  # Higher image density
                Assonance(min_repetitions=5, scope='poem'),
                Consonance(min_repetitions=5, scope='poem'),
            ]
        )
    
    @staticmethod
    def sound_poem() -> FormTemplate:
        """
        A poem emphasizing sonic texture over meaning.
        """
        return FormTemplate(
            name="Sound Poem",
            description="A poem foregrounding sonic texture—assonance, consonance, alliteration, internal rhyme.",
            constraints=[
                LineCount.between(8, 20),
                Assonance(min_repetitions=4, scope='line'),
                Consonance(min_repetitions=3, scope='line'),
                Alliteration(min_alliterative_words=2, scope='line'),
                InternalRhyme(min_instances=1, scope='poem'),
                Sibilance(min_per_line=1),
            ],
            weights={
                'line_count': 0.05,
                'assonance': 0.25,
                'consonance': 0.25,
                'alliteration': 0.2,
                'internal_rhyme': 0.15,
                'sibilance': 0.1
            }
        )
    
    @staticmethod
    def imagist_poem() -> FormTemplate:
        """
        Imagist poem in the style of Pound/H.D./Williams.
        
        Principles:
        - Direct treatment of the "thing"
        - No word that doesn't contribute
        - Musical phrase, not metronome
        """
        return FormTemplate(
            name="Imagist Poem",
            description="A short, precise poem in the Imagist tradition—concrete imagery, no wasted words.",
            constraints=[
                LineCount.between(3, 15),
                LineCount.at_most(15),
                WordCount(NumericBound.at_most(100), scope='total'),
                ImageDensity(min_images_per_line=0.7),
                RhythmVariation(min_variation=0.2, max_variation=0.5),
                # No end rhyme (distinguishes from traditional forms)
                ~RhymeScheme("AA"),
                ~RhymeScheme("AABB"),
            ]
        )
    
    @staticmethod
    def slant_rhyme_sonnet() -> FormTemplate:
        """
        Sonnet using only slant/near rhymes.
        """
        # Generate near-rhyme pairs for sonnet positions
        # ABAB CDCD EFEF GG
        near_rhyme_pairs = [
            (0, 2), (1, 3),   # First quatrain
            (4, 6), (5, 7),   # Second quatrain
            (8, 10), (9, 11), # Third quatrain
            (12, 13),         # Couplet
        ]
        
        return FormTemplate(
            name="Slant Rhyme Sonnet",
            description="A 14-line sonnet using near/slant rhymes instead of perfect rhymes.",
            constraints=[
                LineCount.exactly(14),
                SyllableCount(NumericBound.between(8, 12), scope='line'),
                # Near rhymes at traditional positions
                *[NearRhyme([pair], min_similarity=0.4, max_similarity=0.85) 
                  for pair in near_rhyme_pairs],
            ]
        )
    
    @staticmethod
    def blues_poem() -> FormTemplate:
        """
        Blues poem form—AAB pattern with repetition.
        """
        return FormTemplate(
            name="Blues Poem",
            description="A poem in blues form: statement, repetition (with variation), resolution.",
            constraints=[
                # Traditional blues: 3-line stanzas, first two lines similar
                StanzaStructure([3, 3, 3]),  # Three tercets
                # Lines 1 and 2 of each stanza should be similar (not identical)
                # This is approximate—true blues has varied repetition
                Assonance(min_repetitions=3, scope='stanza'),
                RhymeScheme("AAB AAB AAB"),
            ]
        )
    
    @staticmethod
    def breath_poem() -> FormTemplate:
        """
        Poem structured around breath units (à la Olson's projective verse).
        """
        return FormTemplate(
            name="Breath Poem",
            description="A poem structured around breath—each line a single breath unit.",
            constraints=[
                # Lines should be speakable in one breath (roughly 3-12 words)
                WordCount.per_line_at_least(3),
                WordCount.per_line_at_most(12),
                # Natural rhythm variation
                RhythmVariation(min_variation=0.2, max_variation=0.5),
                # Enjambment follows breath
                Enjambment(min_enjambed_lines=0.2, max_enjambed_lines=0.6),
            ]
        )


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_sonic_and_free_verse():
    """Demonstrate the extended sonic and free verse capabilities."""
    
    print("=" * 70)
    print("MELOS: Extended Sonic & Free Verse Demonstration")
    print("=" * 70)
    
    env = ComposablePoetryEnvironment()
    
    # Test with a William Carlos Williams poem (imagist/free verse)
    williams = """so much depends
upon

a red wheel
barrow

glazed with rain
water

beside the white
chickens"""
    
    print("\n" + "─" * 70)
    print("William Carlos Williams - 'The Red Wheelbarrow'")
    print("─" * 70)
    
    # Evaluate as imagist poem
    imagist_form = ExtendedFormLibrary.imagist_poem()
    result, rubric, grade = env.evaluate(williams, imagist_form)
    
    print(f"\nImagist Form Score: {result.total_score:.2%}")
    print(grade.format_markdown())
    
    # Test assonance detection
    print("\n" + "─" * 70)
    print("Assonance Analysis")
    print("─" * 70)
    
    assonance_heavy = """The rain in Spain falls mainly on the plain,
where pale lakes make their way to waiting gates,
and gray days fade to haze in lazy waves."""
    
    ctx = ConstraintContext.from_poem(assonance_heavy, {})
    assonance_constraint = Assonance(min_repetitions=3, scope='line')
    result = assonance_constraint.verify(ctx)
    
    print(f"Assonance Score: {result.score:.2%}")
    print(f"Details: {result.details}")
    
    # Test internal rhyme
    print("\n" + "─" * 70)
    print("Internal Rhyme Analysis")
    print("─" * 70)
    
    internal_rhyme_poem = """The cat sat flat upon the mat so fat,
while light bright white ignites the night."""
    
    ctx = ConstraintContext.from_poem(internal_rhyme_poem, {})
    internal_constraint = InternalRhyme(min_instances=1, scope='line')
    result = internal_constraint.verify(ctx)
    
    print(f"Internal Rhyme Score: {result.score:.2%}")
    print(f"Rhymes Found: {result.details.get('rhymes_found', [])}")
    
    # Test free verse quality
    print("\n" + "─" * 70)
    print("Free Verse Quality Analysis")
    print("─" * 70)
    
    free_verse = """The morning arrives
with its cargo of light—
spilling gold across the kitchen floor
where the cat
stretches, yawns,
and begins again her patient work
of watching dust
           drift
               through sun."""
    
    free_verse_form = ExtendedFormLibrary.free_verse_lyric()
    result, rubric, grade = env.evaluate(free_verse, free_verse_form)
    
    print(f"\nFree Verse Score: {result.total_score:.2%}")
    for name, r in result.constraint_results.items():
        print(f"  {name}: {r.score:.2%}")
    
    # Show available sonic constraints
    print("\n" + "─" * 70)
    print("Available Sonic Constraints")
    print("─" * 70)
    
    sonic_constraints = [
        ("Assonance", "Repeated vowel sounds"),
        ("Consonance", "Repeated consonant sounds"),
        ("Sibilance", "S/Z/SH sounds (hissing)"),
        ("Alliteration", "Initial consonant repetition"),
        ("InternalRhyme", "Rhymes within lines"),
        ("NearRhyme", "Slant/half rhymes"),
        ("Enjambment", "Run-on lines"),
        ("RhythmVariation", "Line length variation"),
        ("LineBreakQuality", "Meaningful line breaks"),
        ("ImageDensity", "Concrete imagery"),
        ("FreeVerseForm", "Composite free verse quality"),
    ]
    
    for name, desc in sonic_constraints:
        print(f"  {name}: {desc}")


if __name__ == "__main__":
    demonstrate_sonic_and_free_verse()
```

---

## Summary

### System Names (Ranked by Elegance)

|Rank|Name|Why|
|---|---|---|
|1|**Melos**|Simple, ancient, means "song"—the music of language|
|2|**Tessera**|Each constraint is a tile; the poem is a mosaic|
|3|**Cadenza**|The virtuosic passage; constraint as performance|
|4|**Strophe**|Classical term for verse unit; scholarly feel|
|5|**Jacquard**|The programmable loom—weaving pattern from code|

### New Constraint Categories

|Category|Constraints|Purpose|
|---|---|---|
|**Vowel Sounds**|`Assonance`, `VowelPattern`|Repeated vowel sounds|
|**Consonant Sounds**|`Consonance`, `Sibilance`, `Alliteration`|Repeated consonant patterns|
|**Rhyme Types**|`InternalRhyme`, `NearRhyme`, `RhymeScheme`|Various rhyme positions/types|
|**Free Verse**|`Enjambment`, `RhythmVariation`, `LineBreakQuality`, `ImageDensity`|Quality metrics for unmetered verse|
|**Composite**|`FreeVerseForm`, `SoundPoem`|Combined sonic/structural evaluation|

### New Form Templates

- **Free Verse Lyric** — Contemporary lyric with attention to all free verse qualities
- **Prose Poem** — Single block, high imagery, sonic texture
- **Sound Poem** — Foregrounds sonic texture over semantic content
- **Imagist Poem** — Short, precise, concrete (Pound/Williams style)
- **Slant Rhyme Sonnet** — Traditional structure with near rhymes
- **Blues Poem** — AAB pattern with repetition
- **Breath Poem** — Lines as breath units (Olson's projective verse)

  

Claude
