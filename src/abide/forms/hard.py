"""
Hard poetic forms designed to exploit LLM weaknesses.

These forms use constraints that LLMs struggle with:
- Precise counting (characters, vowels, words)
- Cross-line dependencies
- Character-level manipulation
- Position-based requirements

Use these to test instruction-following beyond pattern matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abide.constraints import (
    And,
    CharacterPalindrome,
    Constraint,
    ConstraintType,
    CrossLineVowelWordCount,
    DoubleAcrostic,
    ExactCharacterBudget,
    ExactTotalCharacters,
    ExactTotalVowels,
    ExactWordCount,
    LineCount,
    NoSharedLetters,
    PositionalCharacter,
    SyllablesPerLine,
    VerificationResult,
    WordLengthStaircase,
)
from abide.constraints.lexical import CharacterCount

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class PrecisionVerse(Constraint):
    """
    A 4-line poem where each line has exactly 30 characters.
    Tests precise character counting ability.
    """

    name = "PrecisionVerse"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, chars_per_line: int = 30, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.chars_per_line = chars_per_line
        self._constraint = And(
            [
                LineCount(4),
                CharacterCount(chars_per_line, tolerance=0),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"PrecisionVerse: Write exactly 4 lines. "
            f"Each line must have EXACTLY {self.chars_per_line} characters (including spaces). "
            f"Count carefully!"
        )


class VowelBudgetPoem(Constraint):
    """
    A poem with exactly 50 vowels total.
    Tests global counting across the entire text.
    """

    name = "VowelBudgetPoem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, vowel_count: int = 50, min_lines: int = 4, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.vowel_count = vowel_count
        self.min_lines = min_lines
        self._constraint = And(
            [
                LineCount(min_lines, min_lines + 10),
                ExactTotalVowels(vowel_count),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"VowelBudgetPoem: Write a poem of {self.min_lines}-{self.min_lines + 10} lines "
            f"containing EXACTLY {self.vowel_count} vowels (a, e, i, o, u) total. "
            f"Count every vowel!"
        )


class StaircasePoem(Constraint):
    """
    A poem where word N has exactly N letters.
    Word 1 = 1 letter, word 2 = 2 letters, etc.
    """

    name = "StaircasePoem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, num_words: int = 10, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_words = num_words
        self._constraint = And(
            [
                ExactWordCount(num_words),
                WordLengthStaircase(max_words=num_words, ascending=True),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"StaircasePoem: Write exactly {self.num_words} words where "
            f"word 1 has 1 letter, word 2 has 2 letters, word 3 has 3 letters, "
            f"and so on up to word {self.num_words} with {self.num_words} letters."
        )


class DescendingStaircasePoem(Constraint):
    """
    Like StaircasePoem but descending: first word has N letters, last has 1.
    """

    name = "DescendingStaircasePoem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, num_words: int = 10, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_words = num_words
        self._constraint = And(
            [
                ExactWordCount(num_words),
                WordLengthStaircase(max_words=num_words, ascending=False),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"DescendingStaircasePoem: Write exactly {self.num_words} words where "
            f"word 1 has {self.num_words} letters, word 2 has {self.num_words - 1} letters, "
            f"down to word {self.num_words} with 1 letter."
        )


class ArithmeticVerse(Constraint):
    """
    Each line must have as many words as the previous line has vowels.
    Tests cross-line dependency tracking.
    """

    name = "ArithmeticVerse"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, num_lines: int = 5, start_words: int = 3, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_lines = num_lines
        self.start_words = start_words
        self._constraint = And(
            [
                LineCount(num_lines),
                CrossLineVowelWordCount(start_words=start_words),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"ArithmeticVerse: Write {self.num_lines} lines. "
            f"Line 1 must have exactly {self.start_words} words. "
            f"Each subsequent line must have as many WORDS as the previous line has VOWELS. "
            f"Plan carefully - the vowel count in line N determines word count in line N+1."
        )


class IsolatedCouplet(Constraint):
    """
    Two lines that share NO letters with each other.
    Extremely difficult constraint.
    """

    name = "IsolatedCouplet"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._constraint = And(
            [
                LineCount(2),
                NoSharedLetters("consecutive"),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            "IsolatedCouplet: Write exactly 2 lines where the lines share NO letters. "
            "If line 1 uses 'a', line 2 cannot use 'a'. "
            "You must use completely different sets of letters in each line."
        )


class AlternatingIsolation(Constraint):
    """
    Four lines where odd lines share no letters with even lines.
    Lines 1&3 can share letters. Lines 2&4 can share letters.
    But 1&2, 2&3, 3&4 cannot share.
    """

    name = "AlternatingIsolation"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._constraint = And(
            [
                LineCount(4),
                NoSharedLetters("consecutive"),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            "AlternatingIsolation: Write 4 lines. "
            "Each consecutive pair (1&2, 2&3, 3&4) must share NO letters. "
            "This requires careful planning of which letters appear where."
        )


class CharacterPalindromePoem(Constraint):
    """
    Every line must be a character-level palindrome.
    Like "A man a plan a canal Panama" but for each line.
    """

    name = "CharacterPalindromePoem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, num_lines: int = 3, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_lines = num_lines
        self._constraint = And(
            [
                LineCount(num_lines),
                CharacterPalindrome(),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"CharacterPalindromePoem: Write {self.num_lines} lines. "
            f"Each line must read the same forwards and backwards (ignoring spaces/punctuation). "
            f"Example: 'Was it a car or a cat I saw'"
        )


class DoubleAcrosticPoem(Constraint):
    """
    First letters of lines spell one word, last letters spell another.
    Both words must have the same length.
    """

    name = "DoubleAcrosticPoem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self, first_word: str = "LOVE", last_word: str = "HATE", weight: float = 1.0
    ) -> None:
        if len(first_word) != len(last_word):
            raise ValueError("Words must have same length")
        super().__init__(weight)
        self.first_word = first_word.upper()
        self.last_word = last_word.upper()
        self._constraint = DoubleAcrostic(first_word, last_word)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"DoubleAcrosticPoem: Write {len(self.first_word)} lines. "
            f"The FIRST letters of each line must spell '{self.first_word}'. "
            f"The LAST letters of each line must spell '{self.last_word}'. "
            f"Plan both the beginning AND ending of each line."
        )


class CharacterBudgetPoem(Constraint):
    """
    Poem must contain exactly N of a specific character.
    """

    name = "CharacterBudgetPoem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        character: str = "e",
        count: int = 15,
        min_lines: int = 4,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.character = character
        self.count = count
        self.min_lines = min_lines
        self._constraint = And(
            [
                LineCount(min_lines, min_lines + 6),
                ExactCharacterBudget(character, count),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"CharacterBudgetPoem: Write {self.min_lines}-{self.min_lines + 6} lines "
            f"containing EXACTLY {self.count} occurrences of the letter '{self.character}'. "
            f"Count every '{self.character}' carefully!"
        )


class TotalCharacterPoem(Constraint):
    """
    Entire poem must have exactly N characters.
    """

    name = "TotalCharacterPoem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, total_chars: int = 100, min_lines: int = 3, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.total_chars = total_chars
        self.min_lines = min_lines
        self._constraint = And(
            [
                LineCount(min_lines, min_lines + 5),
                ExactTotalCharacters(total_chars, count_spaces=True),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"TotalCharacterPoem: Write {self.min_lines}-{self.min_lines + 5} lines "
            f"totaling EXACTLY {self.total_chars} characters (including spaces, excluding newlines). "
            f"Count every character!"
        )


class PositionalPoem(Constraint):
    """
    Specific characters must appear at specific positions in each line.
    """

    name = "PositionalPoem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        positions: list[tuple[int, str]] | None = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if positions is None:
            positions = [(1, "T"), (5, "e")]  # Default: T at start, e at position 5
        self.positions = positions
        self._constraint = And(
            [
                LineCount(4, 8),
                PositionalCharacter(positions, per_line=True),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        pos_desc = ", ".join(f"position {p}='{c}'" for p, c in self.positions)
        return (
            f"PositionalPoem: Write 4-8 lines. "
            f"In EVERY line, these characters must appear at these positions: {pos_desc}. "
            f"Position 1 is the first character of the line."
        )


class ExactWordPoem(Constraint):
    """
    Poem must have exactly N words total.
    """

    name = "ExactWordPoem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, word_count: int = 25, min_lines: int = 4, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.word_count = word_count
        self.min_lines = min_lines
        self._constraint = And(
            [
                LineCount(min_lines, min_lines + 6),
                ExactWordCount(word_count),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"ExactWordPoem: Write {self.min_lines}-{self.min_lines + 6} lines "
            f"containing EXACTLY {self.word_count} words total. Count every word!"
        )


class PrecisionHaiku(Constraint):
    """
    A haiku with EXACT syllable counts AND exact character counts per line.
    Much harder than regular haiku.
    """

    name = "PrecisionHaiku"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        chars_per_line: tuple[int, int, int] = (15, 21, 15),
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.chars_per_line = chars_per_line
        self._constraint = And(
            [
                LineCount(3),
                SyllablesPerLine([5, 7, 5], tolerance=0),
                CharacterCount(list(chars_per_line), tolerance=0),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"PrecisionHaiku: Write a haiku (5-7-5 syllables) where "
            f"line 1 has exactly {self.chars_per_line[0]} characters, "
            f"line 2 has exactly {self.chars_per_line[1]} characters, "
            f"line 3 has exactly {self.chars_per_line[2]} characters. "
            f"Both syllables AND characters must be exact!"
        )


class CombinedChallenge(Constraint):
    """
    Multiple hard constraints combined:
    - Exact vowel count
    - Exact word count
    - Staircase word lengths
    """

    name = "CombinedChallenge"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, num_words: int = 7, vowel_count: int = 20, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_words = num_words
        self.vowel_count = vowel_count
        self._constraint = And(
            [
                ExactWordCount(num_words),
                ExactTotalVowels(vowel_count),
                WordLengthStaircase(max_words=num_words),
            ]
        )

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        result = self._constraint.verify(poem)
        return VerificationResult(
            score=result.score,
            passed=result.passed,
            rubric=result.rubric,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details=result.details,
        )

    def describe(self) -> str:
        return (
            f"CombinedChallenge: Write exactly {self.num_words} words where "
            f"word 1 has 1 letter, word 2 has 2 letters, up to word {self.num_words} "
            f"with {self.num_words} letters. The entire text must contain "
            f"EXACTLY {self.vowel_count} vowels total."
        )
