"""
Lexical constraints for unusual and experimental poetry forms.

These constraints focus on word-level and character-level patterns
that go beyond traditional poetic structures.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, cast

from abide.constraints._validation import (
    require_alphabetic_text,
    require_nonempty_patterns,
    require_nonnegative,
    require_percentage,
    require_positive,
    require_single_alphabetic_character,
    require_single_character_string,
    require_word_list,
)
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
        require_nonnegative(tolerance, "tolerance")
        values: tuple[int, ...]
        if isinstance(words_per_line, int):
            values = (words_per_line,)
            self.uniform = True
        else:
            values = tuple(words_per_line)
            self.uniform = False
        if not values or any(count <= 0 for count in values):
            raise ValueError("words_per_line must contain at least one positive count")
        self.words_per_line = list(values)
        self.tolerance = tolerance

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        matches = 0
        details = []
        expected_total = len(structure.lines)
        if not self.uniform and len(structure.lines) < len(self.words_per_line):
            expected_total = len(self.words_per_line)
            missing = len(self.words_per_line) - len(structure.lines)
            details.append(f"Missing {missing} expected line(s)")

        for i, line in enumerate(structure.lines):
            words = line.split()
            actual = len(words)
            expected = self.words_per_line[i % len(self.words_per_line)]

            if abs(actual - expected) <= self.tolerance:
                matches += 1
                details.append(f"Line {i + 1}: ✓ {actual} words")
            else:
                details.append(f"Line {i + 1}: ✗ {actual} words (expected {expected})")

        # Quadratic penalty: 3/9 correct = 0.11 instead of 0.33
        # This makes partial compliance less rewarding in GRPO training
        linear_score = matches / max(1, expected_total)
        score = linear_score**2
        passed = linear_score >= 0.9  # Stricter pass threshold

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "linear_score": linear_score, "line_details": details},
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
        self.required_words = list(require_word_list(required_words, "required_words"))
        require_positive(min_occurrences, "min_occurrences")
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
        # Quadratic penalty for stricter GRPO training
        linear_score = satisfied / max(1, len(self.required_words))
        score = linear_score**2
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
            passed = False
            linear_score = 0.0
            adequacy = 0.0
        else:
            # Quadratic penalty for stricter GRPO training
            adequacy = min(1.0, total_words / self.min_words)
            linear_score = (unique_count / total_words) * adequacy
            score = linear_score**2
            passed = linear_score == 1.0 and total_words >= self.min_words

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
                "min_words": self.min_words,
                "adequacy": adequacy,
                "linear_score": linear_score,
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
        require_nonnegative(tolerance, "tolerance")
        # If flat list, each number is a word length in sequence
        # If nested list, each inner list is word lengths for a line
        self.line_patterns: list[list[int]] = []
        self.word_pattern: list[int] = []
        if not pattern:
            raise ValueError("pattern must contain at least one positive length")
        if isinstance(pattern[0], list):
            nested_pattern = cast("list[list[int]]", pattern)
            self.line_patterns = [list(line_pattern) for line_pattern in nested_pattern]
            if any(
                not line_pattern or any(length <= 0 for length in line_pattern)
                for line_pattern in self.line_patterns
            ):
                raise ValueError("pattern must contain at least one positive length")
            self.flat = False
        else:
            flat_pattern = cast("list[int]", pattern)
            self.word_pattern = list(flat_pattern)
            if any(length <= 0 for length in self.word_pattern):
                raise ValueError("pattern must contain at least one positive length")
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

        # Quadratic penalty for stricter GRPO training
        linear_score = matches / max(1, total)
        score = linear_score**2
        passed = linear_score >= 0.8

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
        if letter is None:
            self.letter = None
            require_positive(min_consecutive, "min_consecutive")
        else:
            require_single_alphabetic_character(letter, "letter")
            self.letter = letter.upper()
            require_positive(min_words, "min_words")
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
            # Quadratic penalty for stricter GRPO training
            linear_score = min(1.0, matching / max(1, self.min_words)) if self.min_words else 1.0
            score = linear_score**2
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

            # Quadratic penalty for stricter GRPO training
            linear_score = lines_with_allit / max(1, len(structure.lines))
            score = linear_score**2
            passed = linear_score >= 0.5

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
        require_nonnegative(tolerance, "tolerance")
        values: tuple[int, ...]
        if isinstance(chars_per_line, int):
            values = (chars_per_line,)
            self.uniform = True
        else:
            values = tuple(chars_per_line)
            self.uniform = False
        if not values or any(count <= 0 for count in values):
            raise ValueError("chars_per_line must contain at least one positive count")
        self.chars_per_line = list(values)
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

        # Quadratic penalty for stricter GRPO training
        linear_score = matches / max(1, len(structure.lines))
        score = linear_score**2
        passed = linear_score >= 0.9

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "linear_score": linear_score, "line_details": details},
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
        pattern_values: tuple[str, ...]
        if isinstance(patterns, str):
            pattern_values = (patterns,)
        else:
            pattern_values = tuple(patterns)
        self.patterns = list(require_nonempty_patterns(pattern_values, "patterns"))
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

        # Quadratic penalty for stricter GRPO training
        linear_score = matches / max(1, len(structure.lines))
        score = linear_score**2
        passed = linear_score >= 0.8

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
        pattern_values: tuple[str, ...]
        if isinstance(patterns, str):
            pattern_values = (patterns,)
        else:
            pattern_values = tuple(patterns)
        self.patterns = list(require_nonempty_patterns(pattern_values, "patterns"))
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

        # Quadratic penalty for stricter GRPO training
        linear_score = matches / max(1, len(structure.lines))
        score = linear_score**2
        passed = linear_score >= 0.8

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
        require_single_alphabetic_character(letter, "letter")
        require_percentage(min_percent, "min_percent")
        require_percentage(max_percent, "max_percent")
        if max_percent < min_percent:
            raise ValueError("max_percent must be greater than or equal to min_percent")
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
            # How far outside the range? Quadratic penalty for stricter GRPO training
            if percent < self.min_percent:
                linear_score = percent / self.min_percent if self.min_percent > 0 else 0
            else:
                linear_score = self.max_percent / percent if percent > 0 else 0
            score = linear_score**2

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
        require_positive(min_lines, "min_lines")
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
            # Quadratic penalty for stricter GRPO training
            linear_score = max(0, 1.0 - (consecutive_repeats / (len(words) - 1)))
            score = linear_score**2

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
        if not pattern or any(char not in {"V", "C"} for char in pattern.upper()):
            raise ValueError("pattern must contain only V and C characters")
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

            # Quadratic penalty for stricter GRPO training
            linear_score = matches / max(1, len(structure.lines))
            score = linear_score**2
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

            # Quadratic penalty for stricter GRPO training
            linear_score = matches / max(1, len(all_words))
            score = linear_score**2

        passed = linear_score >= 0.8

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
        require_positive(min_words, "min_words")
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
        # Quadratic penalty for stricter GRPO training
        adequacy = min(1.0, total / self.min_words) if total > 0 else 0.0
        linear_score = (monosyllabic / max(1, total)) * adequacy
        score = linear_score**2
        passed = linear_score == 1.0 and total >= self.min_words

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "monosyllabic": monosyllabic,
                "total": total,
                "min_words": self.min_words,
                "adequacy": adequacy,
                "linear_score": linear_score,
            },
        )

    def describe(self) -> str:
        return f"Monosyllabic Only: all words one syllable (min {self.min_words} words)"


# =============================================================================
# HARD CONSTRAINTS - These exploit known LLM weaknesses
# =============================================================================


class ExactTotalCharacters(Constraint):
    """
    Constraint requiring an EXACT total character count for the entire poem.
    LLMs are notoriously bad at counting characters precisely.

    Examples:
        >>> etc = ExactTotalCharacters(100)  # Poem must have exactly 100 characters
        >>> etc = ExactTotalCharacters(250, count_spaces=False)  # 250 non-space chars
    """

    name = "Exact Total Characters"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        total: int,
        count_spaces: bool = True,
        count_newlines: bool = False,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        require_positive(total, "total")
        self.total = total
        self.count_spaces = count_spaces
        self.count_newlines = count_newlines

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if self.count_newlines:
            text = "\n".join(structure.lines)
        else:
            text = "".join(structure.lines)

        if not self.count_spaces:
            text = text.replace(" ", "")

        actual = len(text)
        diff = abs(actual - self.total)

        # Exact match required - partial credit for being close
        # Quadratic penalty for stricter GRPO training
        if diff == 0:
            score = 1.0
        else:
            linear_score = max(0.0, 1.0 - (diff / 10))
            score = linear_score**2

        passed = actual == self.total

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "expected": self.total,
                "actual": actual,
                "difference": diff,
            },
        )

    def describe(self) -> str:
        space_note = "" if self.count_spaces else " (excluding spaces)"
        return f"Exact Total Characters: poem must have exactly {self.total} characters{space_note}"


class ExactTotalVowels(Constraint):
    """
    Constraint requiring an EXACT count of vowels in the entire poem.
    Forces precise planning and counting that LLMs struggle with.

    Examples:
        >>> etv = ExactTotalVowels(50)  # Poem must have exactly 50 vowels
    """

    name = "Exact Total Vowels"
    constraint_type = ConstraintType.STRUCTURAL

    VOWELS: ClassVar[set[str]] = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}

    def __init__(
        self,
        total: int,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        require_positive(total, "total")
        self.total = total

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        text = "".join(structure.lines)

        actual = sum(1 for c in text if c in self.VOWELS)
        diff = abs(actual - self.total)

        # Quadratic penalty for stricter GRPO training
        if diff == 0:
            score = 1.0
        else:
            linear_score = max(0.0, 1.0 - (diff / 10))
            score = linear_score**2

        passed = actual == self.total

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "expected": self.total,
                "actual": actual,
                "difference": diff,
            },
        )

    def describe(self) -> str:
        return f"Exact Total Vowels: poem must contain exactly {self.total} vowels"


class WordLengthStaircase(Constraint):
    """
    Constraint where word N must have exactly N letters.
    Word 1 = 1 letter, word 2 = 2 letters, word 3 = 3 letters, etc.

    This is extremely difficult for LLMs as it requires planning
    word choices based on position.

    Examples:
        >>> wls = WordLengthStaircase(max_words=10)  # First 10 words follow pattern
        >>> wls = WordLengthStaircase(ascending=False)  # Descending: N, N-1, N-2...
    """

    name = "Word Length Staircase"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        max_words: int = 10,
        ascending: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        require_positive(max_words, "max_words")
        self.max_words = max_words
        self.ascending = ascending

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        all_words = []
        for line in structure.lines:
            all_words.extend(re.findall(r"\b[a-zA-Z]+\b", line))

        words_to_check = all_words[: self.max_words]
        matches = 0
        details = []

        for i, word in enumerate(words_to_check):
            if self.ascending:
                expected_len = i + 1
            else:
                expected_len = self.max_words - i

            actual_len = len(word)

            if actual_len == expected_len:
                matches += 1
                details.append(f"Word {i + 1} '{word}': ✓ {actual_len} letters")
            else:
                details.append(
                    f"Word {i + 1} '{word}': ✗ {actual_len} letters (expected {expected_len})"
                )

        total = len(words_to_check)
        # Quadratic penalty for stricter GRPO training
        linear_score = matches / max(1, total)
        score = linear_score**2
        passed = matches == total and total >= self.max_words

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "matches": matches,
                "total": total,
                "word_details": details,
            },
        )

    def describe(self) -> str:
        direction = "ascending" if self.ascending else "descending"
        return f"Word Length Staircase: {direction} word lengths 1,2,3...{self.max_words}"


class CrossLineVowelWordCount(Constraint):
    """
    Constraint where line N must have as many WORDS as line N-1 has VOWELS.
    This requires cross-line planning and precise counting.

    Examples:
        >>> clv = CrossLineVowelWordCount()
    """

    name = "Cross-Line Vowel-Word Count"
    constraint_type = ConstraintType.STRUCTURAL

    VOWELS: ClassVar[set[str]] = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}

    def __init__(
        self,
        start_words: int = 3,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        require_positive(start_words, "start_words")
        self.start_words = start_words  # First line must have this many words

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if len(structure.lines) < 2:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": "Need at least 2 lines"},
            )

        matches = 0
        details = []

        for i, line in enumerate(structure.lines):
            words = line.split()
            word_count = len(words)
            vowels_in_line = sum(1 for c in line if c in self.VOWELS)

            if i == 0:
                # First line: check against start_words
                if word_count == self.start_words:
                    matches += 1
                    details.append(f"Line 1: ✓ {word_count} words (required {self.start_words})")
                else:
                    details.append(f"Line 1: ✗ {word_count} words (required {self.start_words})")
            else:
                # Subsequent lines: word count should equal prev line's vowel count
                prev_line = structure.lines[i - 1]
                prev_vowels = sum(1 for c in prev_line if c in self.VOWELS)

                if word_count == prev_vowels:
                    matches += 1
                    details.append(
                        f"Line {i + 1}: ✓ {word_count} words (prev line had {prev_vowels} vowels)"
                    )
                else:
                    details.append(
                        f"Line {i + 1}: ✗ {word_count} words (prev line had {prev_vowels} vowels)"
                    )

            details[-1] += f" [this line has {vowels_in_line} vowels]"

        # Quadratic penalty for stricter GRPO training
        linear_score = matches / max(1, len(structure.lines))
        score = linear_score**2
        passed = matches == len(structure.lines)

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        return (
            f"Cross-Line Vowel-Word: line N has as many words as line N-1 has vowels "
            f"(line 1 has {self.start_words} words)"
        )


class NoSharedLetters(Constraint):
    """
    Constraint requiring that specified lines share NO letters.
    Extremely difficult as it requires tracking all used letters.

    Examples:
        >>> nsl = NoSharedLetters([(1, 3), (2, 4)])  # Lines 1&3, 2&4 share no letters
        >>> nsl = NoSharedLetters("consecutive")  # Each line shares no letters with next
    """

    name = "No Shared Letters"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        pairs: list[tuple[int, int]] | str = "consecutive",
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.pairs: str | list[tuple[int, int]]
        if isinstance(pairs, str):
            if pairs not in {"consecutive", "alternating"}:
                raise ValueError(
                    "pairs must be 'consecutive', 'alternating', or a list of line pairs"
                )
            self.pairs = pairs
        else:
            if not pairs:
                raise ValueError("pairs must contain at least one line pair")
            validated_pairs: list[tuple[int, int]] = []
            for pair in pairs:
                try:
                    line_a, line_b = pair
                except (TypeError, ValueError):
                    raise ValueError("pairs must be a list of 1-based line pairs") from None
                if line_a <= 0 or line_b <= 0:
                    raise ValueError("pair line numbers must be positive")
                validated_pairs.append((line_a, line_b))
            self.pairs = validated_pairs

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Build pairs to check
        if self.pairs == "consecutive":
            pairs_to_check = [(i, i + 1) for i in range(len(structure.lines) - 1)]
        elif self.pairs == "alternating":
            pairs_to_check = [(i, i + 2) for i in range(len(structure.lines) - 2)]
        else:
            # Convert 1-indexed to 0-indexed
            pair_list = cast("list[tuple[int, int]]", self.pairs)
            pairs_to_check = [(a - 1, b - 1) for a, b in pair_list]

        matches = 0
        details = []

        for line_a_idx, line_b_idx in pairs_to_check:
            if line_a_idx >= len(structure.lines) or line_b_idx >= len(structure.lines):
                continue

            line_a = structure.lines[line_a_idx].lower()
            line_b = structure.lines[line_b_idx].lower()

            letters_a = {c for c in line_a if c.isalpha()}
            letters_b = {c for c in line_b if c.isalpha()}
            if not letters_a or not letters_b:
                details.append(
                    f"Lines {line_a_idx + 1} & {line_b_idx + 1}: ✗ missing alphabetic content"
                )
                continue

            shared = letters_a & letters_b

            if not shared:
                matches += 1
                details.append(f"Lines {line_a_idx + 1} & {line_b_idx + 1}: ✓ no shared letters")
            else:
                shared_list = sorted(shared)[:5]
                details.append(
                    f"Lines {line_a_idx + 1} & {line_b_idx + 1}: ✗ shared: {shared_list}"
                )

        total = len(pairs_to_check)
        # Quadratic penalty for stricter GRPO training
        linear_score = matches / max(1, total)
        score = linear_score**2
        passed = matches == total and total > 0

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "total": total, "pair_details": details},
        )

    def describe(self) -> str:
        if self.pairs == "consecutive":
            return "No Shared Letters: consecutive lines share no letters"
        elif self.pairs == "alternating":
            return "No Shared Letters: alternating lines (1&3, 2&4) share no letters"
        return "No Shared Letters: specified line pairs share no letters"


class ExactWordCount(Constraint):
    """
    Constraint requiring an EXACT total word count for the entire poem.

    Examples:
        >>> ewc = ExactWordCount(50)  # Poem must have exactly 50 words
    """

    name = "Exact Word Count"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        total: int,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        require_positive(total, "total")
        self.total = total

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        all_words = []
        for line in structure.lines:
            all_words.extend(re.findall(r"\b[a-zA-Z]+\b", line))

        actual = len(all_words)
        diff = abs(actual - self.total)

        # Quadratic penalty for stricter GRPO training
        if diff == 0:
            score = 1.0
        else:
            linear_score = max(0.0, 1.0 - (diff / 5))
            score = linear_score**2

        passed = actual == self.total

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "expected": self.total,
                "actual": actual,
                "difference": diff,
            },
        )

    def describe(self) -> str:
        return f"Exact Word Count: poem must have exactly {self.total} words"


class CharacterPalindrome(Constraint):
    """
    Constraint requiring lines to be character-level palindromes.
    Much harder than word palindromes - "A man a plan a canal Panama"

    Examples:
        >>> cp = CharacterPalindrome()  # All lines must be palindromes
        >>> cp = CharacterPalindrome(lines=[1, 3, 5])  # Only specified lines
    """

    name = "Character Palindrome"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        lines: list[int] | None = None,
        ignore_spaces: bool = True,
        ignore_punctuation: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if lines is not None:
            if not lines:
                raise ValueError("lines must contain at least one 1-based line number")
            if any(line <= 0 for line in lines):
                raise ValueError("lines must be 1-based positive integers")
        self.lines = lines  # 1-indexed, None means all
        self.ignore_spaces = ignore_spaces
        self.ignore_punctuation = ignore_punctuation

    def _normalize(self, text: str) -> str:
        """Normalize text for palindrome comparison."""
        text = text.lower()
        if self.ignore_spaces:
            text = text.replace(" ", "")
        if self.ignore_punctuation:
            text = re.sub(r"[^\w]", "", text)
        return text

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if self.lines:
            lines_to_check = [
                (i - 1, structure.lines[i - 1]) for i in self.lines if i - 1 < len(structure.lines)
            ]
        else:
            lines_to_check = list(enumerate(structure.lines))

        matches = 0
        details = []

        for idx, line in lines_to_check:
            normalized = self._normalize(line)
            is_palindrome = normalized == normalized[::-1]

            if is_palindrome:
                matches += 1
                details.append(f"Line {idx + 1}: ✓ palindrome")
            else:
                # Show what went wrong
                rev = normalized[::-1]
                first_diff = next(
                    (i for i, (a, b) in enumerate(zip(normalized, rev)) if a != b), len(normalized)
                )
                details.append(
                    f"Line {idx + 1}: ✗ not palindrome (differs at position {first_diff})"
                )

        total = len(lines_to_check)
        # Quadratic penalty for stricter GRPO training
        linear_score = matches / max(1, total)
        score = linear_score**2
        passed = matches == total and total > 0

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "total": total, "line_details": details},
        )

    def describe(self) -> str:
        if self.lines:
            return f"Character Palindrome: lines {self.lines} must read same forwards/backwards"
        return "Character Palindrome: all lines must read same forwards/backwards"


class DoubleAcrostic(Constraint):
    """
    Constraint requiring first letters spell one word, last letters spell another.
    Extremely difficult as it constrains both ends of each line.

    Examples:
        >>> da = DoubleAcrostic("HELLO", "WORLD")  # First letters spell HELLO, last spell WORLD
    """

    name = "Double Acrostic"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        first_word: str,
        last_word: str,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        require_alphabetic_text(first_word, "first_word")
        require_alphabetic_text(last_word, "last_word")
        self.first_word = "".join(char for char in first_word.upper() if char.isalpha())
        self.last_word = "".join(char for char in last_word.upper() if char.isalpha())

        if len(self.first_word) != len(self.last_word):
            raise ValueError("First and last words must have same length")

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Need exactly the right number of lines
        expected_lines = len(self.first_word)

        if len(structure.lines) != expected_lines:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={
                    "error": f"Need exactly {expected_lines} lines, got {len(structure.lines)}"
                },
            )

        first_matches = 0
        last_matches = 0
        details = []

        for i, line in enumerate(structure.lines):
            # Get first letter (skip leading spaces/punctuation)
            first_char = ""
            for c in line:
                if c.isalpha():
                    first_char = c.upper()
                    break

            # Get last letter (skip trailing spaces/punctuation)
            last_char = ""
            for c in reversed(line):
                if c.isalpha():
                    last_char = c.upper()
                    break

            expected_first = self.first_word[i]
            expected_last = self.last_word[i]

            first_ok = first_char == expected_first
            last_ok = last_char == expected_last

            if first_ok:
                first_matches += 1
            if last_ok:
                last_matches += 1

            status_first = "✓" if first_ok else f"✗ got '{first_char}'"
            status_last = "✓" if last_ok else f"✗ got '{last_char}'"

            details.append(
                f"Line {i + 1}: first='{expected_first}' {status_first}, "
                f"last='{expected_last}' {status_last}"
            )

        total_checks = expected_lines * 2
        total_matches = first_matches + last_matches
        # Quadratic penalty for stricter GRPO training
        linear_score = total_matches / total_checks
        score = linear_score**2
        passed = total_matches == total_checks

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "first_word": self.first_word,
                "last_word": self.last_word,
                "first_matches": first_matches,
                "last_matches": last_matches,
                "line_details": details,
            },
        )

    def describe(self) -> str:
        return (
            f"Double Acrostic: first letters spell '{self.first_word}', "
            f"last letters spell '{self.last_word}'"
        )


class ExactCharacterBudget(Constraint):
    """
    Constraint requiring an exact count of a specific character.
    LLMs struggle with precise character counting.

    Examples:
        >>> ecb = ExactCharacterBudget("e", 10)  # Exactly 10 letter e's
        >>> ecb = ExactCharacterBudget("!", 3)  # Exactly 3 exclamation marks
    """

    name = "Exact Character Budget"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        character: str,
        count: int,
        case_sensitive: bool = False,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        require_single_character_string(character, "character")
        require_positive(count, "count")
        self.character = character
        self.count = count
        self.case_sensitive = case_sensitive

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)
        text = "".join(structure.lines)

        if self.case_sensitive:
            actual = text.count(self.character)
        else:
            actual = text.lower().count(self.character.lower())

        diff = abs(actual - self.count)

        # Quadratic penalty for stricter GRPO training
        if diff == 0:
            score = 1.0
        else:
            linear_score = max(0.0, 1.0 - (diff / 5))
            score = linear_score**2

        passed = actual == self.count

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "character": self.character,
                "expected": self.count,
                "actual": actual,
                "difference": diff,
            },
        )

    def describe(self) -> str:
        return f"Exact Character Budget: exactly {self.count} occurrences of '{self.character}'"


class PositionalCharacter(Constraint):
    """
    Constraint requiring specific characters at specific positions.
    Position can be absolute (char 5) or relative (5th char of each line).

    Examples:
        >>> pc = PositionalCharacter([(5, "x"), (10, "y")])  # Char 5 is x, char 10 is y
        >>> pc = PositionalCharacter([(3, "a")], per_line=True)  # 3rd char of each line is a
    """

    name = "Positional Character"
    constraint_type = ConstraintType.STRUCTURAL

    def __init__(
        self,
        positions: list[tuple[int, str]],
        per_line: bool = False,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if not positions:
            raise ValueError("positions must contain at least one (position, character) pair")
        validated_positions: list[tuple[int, str]] = []
        for item in positions:
            try:
                position, character = item
            except (TypeError, ValueError):
                raise ValueError("positions must be (position, character) pairs") from None
            if position <= 0:
                raise ValueError("positions must be 1-based positive integers")
            require_single_character_string(character, "position characters")
            validated_positions.append((position, character))
        self.positions = validated_positions
        self.per_line = per_line

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        matches = 0
        total = 0
        details = []

        if self.per_line:
            for line_idx, line in enumerate(structure.lines):
                for pos, expected_char in self.positions:
                    total += 1
                    if pos <= len(line):
                        actual = line[pos - 1]  # 1-indexed
                        if actual.lower() == expected_char.lower():
                            matches += 1
                            details.append(f"Line {line_idx + 1} pos {pos}: ✓ '{expected_char}'")
                        else:
                            details.append(
                                f"Line {line_idx + 1} pos {pos}: ✗ got '{actual}' expected '{expected_char}'"
                            )
                    else:
                        details.append(f"Line {line_idx + 1} pos {pos}: ✗ line too short")
        else:
            # Absolute position in entire poem text
            text = "\n".join(structure.lines)
            for pos, expected_char in self.positions:
                total += 1
                if pos <= len(text):
                    actual = text[pos - 1]
                    if actual.lower() == expected_char.lower():
                        matches += 1
                        details.append(f"Position {pos}: ✓ '{expected_char}'")
                    else:
                        details.append(
                            f"Position {pos}: ✗ got '{actual}' expected '{expected_char}'"
                        )
                else:
                    details.append(f"Position {pos}: ✗ poem too short")

        # Quadratic penalty for stricter GRPO training
        linear_score = matches / max(1, total)
        score = linear_score**2
        passed = matches == total and total > 0

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "total": total, "position_details": details},
        )

    def describe(self) -> str:
        pos_str = ", ".join(f"pos {p}='{c}'" for p, c in self.positions[:3])
        suffix = "..." if len(self.positions) > 3 else ""
        scope = "per line" if self.per_line else "in poem"
        return f"Positional Character: {pos_str}{suffix} ({scope})"
