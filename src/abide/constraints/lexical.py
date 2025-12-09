"""
Lexical constraints for unusual and experimental poetry forms.

These constraints focus on word-level and character-level patterns
that go beyond traditional poetic structures.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from abide.constraints.base import Constraint
from abide.constraints.types import ConstraintType, VerificationResult

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class WordCount(Constraint):
    """
    Constraint on exact word count per line.

    Examples:
        >>> wc = WordCount([3, 5, 3])  # Lines have 3, 5, 3 words
        >>> wc = WordCount(4)  # All lines have 4 words
    """

    name = "Word Count"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        words_per_line: int | list[int],
        tolerance: int = 0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if isinstance(words_per_line, int):
            self.words_per_line = [words_per_line]
            self.uniform = True
        else:
            self.words_per_line = words_per_line
            self.uniform = False
        self.tolerance = tolerance

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        matches = 0
        details = []

        for i, line in enumerate(structure.lines):
            words = line.split()
            actual = len(words)
            expected = self.words_per_line[i % len(self.words_per_line)]

            if abs(actual - expected) <= self.tolerance:
                matches += 1
                details.append(f"Line {i + 1}: ✓ {actual} words")
            else:
                details.append(f"Line {i + 1}: ✗ {actual} words (expected {expected})")

        score = matches / max(1, len(structure.lines))
        passed = score >= 0.8

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        if self.uniform:
            return f"Word Count: {self.words_per_line[0]} words per line"
        return f"Word Count: {self.words_per_line} words pattern"


class ForcedWords(Constraint):
    """
    Constraint requiring specific words to appear in the poem.

    Examples:
        >>> fw = ForcedWords(["moon", "tide", "whisper"])
        >>> fw = ForcedWords(["love"], min_occurrences=3)
    """

    name = "Forced Words"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        required_words: list[str],
        min_occurrences: int = 1,
        case_sensitive: bool = False,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.required_words = required_words
        self.min_occurrences = min_occurrences
        self.case_sensitive = case_sensitive

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        text = " ".join(structure.lines)
        if not self.case_sensitive:
            text = text.lower()

        found = {}
        for word in self.required_words:
            check_word = word if self.case_sensitive else word.lower()
            # Word boundary search
            pattern = r"\b" + re.escape(check_word) + r"\b"
            count = len(re.findall(pattern, text))
            found[word] = count

        # Score: how many required words met the minimum
        satisfied = sum(1 for count in found.values() if count >= self.min_occurrences)
        score = satisfied / max(1, len(self.required_words))
        passed = satisfied == len(self.required_words)

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"required": self.required_words, "found": found},
        )

    def describe(self) -> str:
        words = ", ".join(self.required_words[:3])
        suffix = "..." if len(self.required_words) > 3 else ""
        return f"Forced Words: must include {words}{suffix}"


class AllWordsUnique(Constraint):
    """
    Constraint requiring all words in the poem to be unique.

    Examples:
        >>> unique = AllWordsUnique()
        >>> unique = AllWordsUnique(ignore_words=["the", "a", "an"])
    """

    name = "All Words Unique"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        ignore_words: list[str] | None = None,
        min_words: int = 10,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.ignore_words = {w.lower() for w in (ignore_words or [])}
        self.min_words = min_words

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        text = " ".join(structure.lines).lower()
        words = re.findall(r"\b[a-z]+\b", text)

        # Filter ignored words
        words = [w for w in words if w not in self.ignore_words]

        unique_words = set(words)
        total_words = len(words)
        unique_count = len(unique_words)

        if total_words == 0:
            score = 0.0
        else:
            score = unique_count / total_words

        passed = score == 1.0 and total_words >= self.min_words

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "total_words": total_words,
                "unique_words": unique_count,
                "duplicates": total_words - unique_count,
            },
        )

    def describe(self) -> str:
        return f"All Words Unique: no repeated words (min {self.min_words})"


class WordLengthPattern(Constraint):
    """
    Constraint on word lengths in each line.

    Examples:
        >>> wlp = WordLengthPattern([1, 2, 3, 4, 5])  # Words have 1,2,3,4,5 letters
        >>> wlp = WordLengthPattern([[3, 3], [4, 4, 4]])  # Line 1: two 3-letter words
    """

    name = "Word Length Pattern"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        pattern: list[int] | list[list[int]],
        tolerance: int = 0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        # If flat list, each number is a word length in sequence
        # If nested list, each inner list is word lengths for a line
        self.line_patterns: list[list[int]] = []
        self.word_pattern: list[int] = []
        if pattern and isinstance(pattern[0], list):
            self.line_patterns = pattern  # type: ignore[assignment]
            self.flat = False
        else:
            self.word_pattern = pattern  # type: ignore[assignment]
            self.flat = True
        self.tolerance = tolerance

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        matches = 0
        total = 0
        details = []

        if self.flat:
            # All words in poem should follow pattern in sequence
            all_words = []
            for line in structure.lines:
                all_words.extend(re.findall(r"\b[a-zA-Z]+\b", line))

            for i, word in enumerate(all_words):
                expected = self.word_pattern[i % len(self.word_pattern)]
                actual = len(word)
                total += 1
                if abs(actual - expected) <= self.tolerance:
                    matches += 1

            details.append(f"Words matching pattern: {matches}/{total}")
        else:
            # Each line has its own pattern
            for i, line in enumerate(structure.lines):
                words = re.findall(r"\b[a-zA-Z]+\b", line)
                line_pattern = self.line_patterns[i % len(self.line_patterns)]

                if len(words) != len(line_pattern):
                    details.append(f"Line {i + 1}: ✗ wrong word count")
                    total += len(line_pattern)
                    continue

                line_matches = 0
                for word, expected in zip(words, line_pattern):
                    total += 1
                    if abs(len(word) - expected) <= self.tolerance:
                        line_matches += 1
                        matches += 1

                details.append(f"Line {i + 1}: {line_matches}/{len(line_pattern)} match")

        score = matches / max(1, total)
        passed = score >= 0.8

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "total": total, "line_details": details},
        )

    def describe(self) -> str:
        if self.flat:
            return f"Word Length Pattern: {self.word_pattern} letters"
        return f"Word Length Pattern: {len(self.line_patterns)} line patterns"


class Alliteration(Constraint):
    """
    Constraint requiring alliteration (words starting with same letter).

    Examples:
        >>> allit = Alliteration(min_consecutive=3)  # 3+ words same letter
        >>> allit = Alliteration(letter="S", min_words=5)  # 5 words starting with S
    """

    name = "Alliteration"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        letter: str | None = None,
        min_consecutive: int = 2,
        min_words: int = 0,
        per_line: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.letter = letter.upper() if letter else None
        self.min_consecutive = min_consecutive
        self.min_words = min_words
        self.per_line = per_line

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if self.letter:
            # Count words starting with specific letter
            all_words = []
            for line in structure.lines:
                all_words.extend(re.findall(r"\b[a-zA-Z]+\b", line))

            matching = sum(1 for w in all_words if w[0].upper() == self.letter)
            score = min(1.0, matching / max(1, self.min_words)) if self.min_words else 1.0
            passed = matching >= self.min_words

            return VerificationResult(
                score=score,
                passed=passed,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"letter": self.letter, "matching_words": matching},
            )
        else:
            # Check for consecutive alliteration
            lines_with_allit = 0

            for line in structure.lines:
                words = re.findall(r"\b[a-zA-Z]+\b", line)
                if len(words) < self.min_consecutive:
                    continue

                # Check for consecutive words with same starting letter
                for i in range(len(words) - self.min_consecutive + 1):
                    chunk = words[i : i + self.min_consecutive]
                    first_letters = [w[0].upper() for w in chunk]
                    if len(set(first_letters)) == 1:
                        lines_with_allit += 1
                        break

            score = lines_with_allit / max(1, len(structure.lines))
            passed = score >= 0.5

            return VerificationResult(
                score=score,
                passed=passed,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"lines_with_alliteration": lines_with_allit},
            )

    def describe(self) -> str:
        if self.letter:
            return f"Alliteration: {self.min_words}+ words starting with '{self.letter}'"
        return f"Alliteration: {self.min_consecutive}+ consecutive words same letter"


class CharacterCount(Constraint):
    """
    Constraint on exact character count per line.

    Examples:
        >>> cc = CharacterCount(40)  # Each line has 40 characters
        >>> cc = CharacterCount([10, 20, 30])  # Lines have 10, 20, 30 chars
    """

    name = "Character Count"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        chars_per_line: int | list[int],
        count_spaces: bool = True,
        tolerance: int = 0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if isinstance(chars_per_line, int):
            self.chars_per_line = [chars_per_line]
            self.uniform = True
        else:
            self.chars_per_line = chars_per_line
            self.uniform = False
        self.count_spaces = count_spaces
        self.tolerance = tolerance

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        matches = 0
        details = []

        for i, line in enumerate(structure.lines):
            if self.count_spaces:
                actual = len(line.strip())
            else:
                actual = len(line.replace(" ", "").strip())

            expected = self.chars_per_line[i % len(self.chars_per_line)]

            if abs(actual - expected) <= self.tolerance:
                matches += 1
                details.append(f"Line {i + 1}: ✓ {actual} chars")
            else:
                details.append(f"Line {i + 1}: ✗ {actual} chars (expected {expected})")

        score = matches / max(1, len(structure.lines))
        passed = score >= 0.8

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        if self.uniform:
            return f"Character Count: {self.chars_per_line[0]} chars per line"
        return f"Character Count: {self.chars_per_line} chars pattern"


class LineStartsWith(Constraint):
    """
    Constraint requiring lines to start with specific patterns.

    Examples:
        >>> ls = LineStartsWith(["I ", "You ", "We "])  # Rotating starts
        >>> ls = LineStartsWith("The")  # All lines start with "The"
    """

    name = "Line Starts With"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        patterns: str | list[str],
        case_sensitive: bool = False,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if isinstance(patterns, str):
            self.patterns = [patterns]
        else:
            self.patterns = patterns
        self.case_sensitive = case_sensitive

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        matches = 0
        details = []

        for i, line in enumerate(structure.lines):
            expected = self.patterns[i % len(self.patterns)]
            check_line = line.strip() if self.case_sensitive else line.strip().lower()
            check_pattern = expected if self.case_sensitive else expected.lower()

            if check_line.startswith(check_pattern):
                matches += 1
                details.append(f"Line {i + 1}: ✓ starts with '{expected}'")
            else:
                details.append(f"Line {i + 1}: ✗ expected '{expected}'")

        score = matches / max(1, len(structure.lines))
        passed = score >= 0.8

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        if len(self.patterns) == 1:
            return f"Line Starts With: '{self.patterns[0]}'"
        return f"Line Starts With: rotating {self.patterns}"


class LineEndsWith(Constraint):
    """
    Constraint requiring lines to end with specific patterns.

    Examples:
        >>> le = LineEndsWith([".", "!", "?"])  # Rotating endings
        >>> le = LineEndsWith("ing")  # All lines end with "ing"
    """

    name = "Line Ends With"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        patterns: str | list[str],
        case_sensitive: bool = False,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if isinstance(patterns, str):
            self.patterns = [patterns]
        else:
            self.patterns = patterns
        self.case_sensitive = case_sensitive

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        matches = 0
        details = []

        for i, line in enumerate(structure.lines):
            expected = self.patterns[i % len(self.patterns)]
            check_line = line.strip() if self.case_sensitive else line.strip().lower()
            check_pattern = expected if self.case_sensitive else expected.lower()

            if check_line.endswith(check_pattern):
                matches += 1
                details.append(f"Line {i + 1}: ✓ ends with '{expected}'")
            else:
                details.append(f"Line {i + 1}: ✗ expected '{expected}'")

        score = matches / max(1, len(structure.lines))
        passed = score >= 0.8

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        if len(self.patterns) == 1:
            return f"Line Ends With: '{self.patterns[0]}'"
        return f"Line Ends With: rotating {self.patterns}"


class LetterFrequency(Constraint):
    """
    Constraint on frequency of specific letters.

    Examples:
        >>> lf = LetterFrequency("S", min_percent=10)  # At least 10% S
        >>> lf = LetterFrequency("E", max_percent=5)  # No more than 5% E
    """

    name = "Letter Frequency"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        letter: str,
        min_percent: float = 0,
        max_percent: float = 100,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.letter = letter.upper()
        self.min_percent = min_percent
        self.max_percent = max_percent

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        text = "".join(structure.lines).upper()

        total_letters = sum(1 for c in text if c.isalpha())
        target_count = sum(1 for c in text if c == self.letter)

        if total_letters == 0:
            percent = 0.0
        else:
            percent = (target_count / total_letters) * 100

        in_range = self.min_percent <= percent <= self.max_percent

        if in_range:
            score = 1.0
        else:
            # How far outside the range?
            if percent < self.min_percent:
                score = percent / self.min_percent if self.min_percent > 0 else 0
            else:
                score = self.max_percent / percent if percent > 0 else 0

        return VerificationResult(
            score=score,
            passed=in_range,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "letter": self.letter,
                "percent": round(percent, 2),
                "min_percent": self.min_percent,
                "max_percent": self.max_percent,
            },
        )

    def describe(self) -> str:
        if self.min_percent > 0 and self.max_percent < 100:
            return (
                f"Letter Frequency: '{self.letter}' between {self.min_percent}-{self.max_percent}%"
            )
        elif self.min_percent > 0:
            return f"Letter Frequency: '{self.letter}' at least {self.min_percent}%"
        else:
            return f"Letter Frequency: '{self.letter}' at most {self.max_percent}%"


class NoConsecutiveRepeats(Constraint):
    """
    Constraint preventing consecutive repeated words.

    Examples:
        >>> ncr = NoConsecutiveRepeats()  # No word appears twice in a row
    """

    name = "No Consecutive Repeats"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        min_lines: int = 3,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.min_lines = min_lines

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        text = " ".join(structure.lines).lower()
        words = re.findall(r"\b[a-z]+\b", text)

        consecutive_repeats = 0
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                consecutive_repeats += 1

        if len(words) <= 1:
            score = 1.0
        else:
            # Penalize based on how many repeats
            score = max(0, 1.0 - (consecutive_repeats / (len(words) - 1)))

        passed = consecutive_repeats == 0 and len(structure.lines) >= self.min_lines

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"consecutive_repeats": consecutive_repeats, "total_words": len(words)},
        )

    def describe(self) -> str:
        return "No Consecutive Repeats: no word appears twice in a row"


class VowelConsonantPattern(Constraint):
    """
    Constraint on vowel/consonant patterns at word starts.

    Examples:
        >>> vcp = VowelConsonantPattern("VCVC")  # Vowel, consonant, vowel, consonant
        >>> vcp = VowelConsonantPattern("VVCC")  # Two vowel-start, two consonant-start
    """

    name = "Vowel/Consonant Pattern"
    constraint_type = ConstraintType.SEMANTIC

    VOWELS: ClassVar[set[str]] = {"A", "E", "I", "O", "U"}

    def __init__(
        self,
        pattern: str,
        per_line: bool = False,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.pattern = pattern.upper()
        self.per_line = per_line

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if self.per_line:
            # Pattern applies to each line
            matches = 0
            for line in structure.lines:
                words = re.findall(r"\b[a-zA-Z]+\b", line)
                if len(words) < len(self.pattern):
                    continue

                line_match = True
                for word, expected in zip(words, self.pattern):
                    first = word[0].upper()
                    is_vowel = first in self.VOWELS
                    expects_vowel = expected == "V"

                    if is_vowel != expects_vowel:
                        line_match = False
                        break

                if line_match:
                    matches += 1

            score = matches / max(1, len(structure.lines))
        else:
            # Pattern applies to all words in sequence
            all_words = []
            for line in structure.lines:
                all_words.extend(re.findall(r"\b[a-zA-Z]+\b", line))

            matches = 0
            for i, word in enumerate(all_words):
                expected = self.pattern[i % len(self.pattern)]
                first = word[0].upper()
                is_vowel = first in self.VOWELS
                expects_vowel = expected == "V"

                if is_vowel == expects_vowel:
                    matches += 1

            score = matches / max(1, len(all_words))

        passed = score >= 0.8

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"pattern": self.pattern, "score": score},
        )

    def describe(self) -> str:
        return f"V/C Pattern: words start {self.pattern} (V=vowel, C=consonant)"


class MonosyllabicOnly(Constraint):
    """
    Constraint requiring only one-syllable words.

    Examples:
        >>> mono = MonosyllabicOnly()
    """

    name = "Monosyllabic Only"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        min_words: int = 10,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.min_words = min_words

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        from abide.primitives.phonetics import count_syllables

        structure = self._ensure_structure(poem)
        all_words = []
        for line in structure.lines:
            all_words.extend(re.findall(r"\b[a-zA-Z]+\b", line))

        monosyllabic = 0
        for word in all_words:
            syllables = count_syllables(word)
            if syllables == 1:
                monosyllabic += 1

        total = len(all_words)
        score = monosyllabic / max(1, total)
        passed = score == 1.0 and total >= self.min_words

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"monosyllabic": monosyllabic, "total": total},
        )

    def describe(self) -> str:
        return f"Monosyllabic Only: all words one syllable (min {self.min_words} words)"
