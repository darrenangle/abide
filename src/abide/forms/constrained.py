"""
Constrained writing forms and lipograms.

Abecedarian, Lipogram, Univocalic, Mesostic, Anaphora, Palindrome.
These forms focus on letter/word constraints rather than structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from abide.constraints import (
    Constraint,
    ConstraintType,
    LineCount,
    NumericBound,
    VerificationResult,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Abecedarian(Constraint):
    """
    Abecedarian: Lines begin with successive letters of the alphabet.

    Structure:
    - Each line starts with the next letter: A, B, C, D, ...
    - Can be 26 lines (full alphabet) or partial
    - Ancient form used in Hebrew acrostic psalms

    Examples:
        >>> abc = Abecedarian()
        >>> result = abc.verify(poem)

        >>> # Partial alphabet
        >>> abc = Abecedarian(letters="ABCDEFGH")
    """

    name = "Abecedarian"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        letters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        weight: float = 1.0,
    ) -> None:
        """
        Initialize abecedarian constraint.

        Args:
            letters: Sequence of letters for line starts (default: full alphabet)
            weight: Relative weight for composition
        """
        super().__init__(weight)
        self.letters = letters.upper()
        self.num_lines = len(self.letters)

        self._line_count = LineCount(self.num_lines, weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check line count
        line_result = self._line_count.verify(poem)

        # Check each line starts with correct letter
        matches = 0
        details = []

        for i, (expected_letter, line) in enumerate(zip(self.letters, structure.lines)):
            line_stripped = line.strip()
            if line_stripped:
                actual_first = line_stripped[0].upper()
                if actual_first == expected_letter:
                    matches += 1
                    details.append(f"Line {i + 1}: ✓ starts with '{expected_letter}'")
                else:
                    details.append(
                        f"Line {i + 1}: ✗ expected '{expected_letter}', got '{actual_first}'"
                    )
            else:
                details.append(f"Line {i + 1}: ✗ empty line")

        # Quadratic penalty for stricter GRPO training
        linear_letter = matches / max(1, min(len(self.letters), len(structure.lines)))
        letter_score = linear_letter**2

        # Combine scores
        score = line_result.score * 0.1 + letter_score * 0.9
        passed = score >= 0.8

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "matches": matches,
                "expected": self.num_lines,
                "letter_score": letter_score,
                "line_details": details,
            },
        )

    def describe(self) -> str:
        if len(self.letters) == 26:
            return "Abecedarian: 26 lines starting A-Z"
        return f"Abecedarian: {len(self.letters)} lines starting {self.letters[:3]}..."


class Lipogram(Constraint):
    """
    Lipogram: Writing that omits one or more letters.

    Famous example: "Gadsby" by Ernest Vincent Wright omits 'E'.

    Examples:
        >>> lipo = Lipogram(forbidden="E")  # No letter E
        >>> result = lipo.verify(poem)

        >>> lipo = Lipogram(forbidden="AEIOU")  # No vowels
    """

    name = "Lipogram"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        forbidden: str = "E",
        min_lines: int = 1,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize lipogram constraint.

        Args:
            forbidden: Letters that must not appear
            min_lines: Minimum number of lines
            weight: Relative weight for composition
        """
        super().__init__(weight)
        self.forbidden = set(forbidden.upper())
        self.min_lines = min_lines

        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=0.5)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check line count
        line_result = self._line_count.verify(poem)

        # Count forbidden letters
        text_upper = "".join(structure.lines).upper()
        total_letters = sum(1 for c in text_upper if c.isalpha())
        forbidden_count = sum(1 for c in text_upper if c in self.forbidden)

        if total_letters == 0:
            lipogram_score = 0.0
        else:
            # Perfect score if no forbidden letters
            lipogram_score = 1.0 - (forbidden_count / total_letters)

        # Combine scores
        score = line_result.score * 0.1 + lipogram_score * 0.9
        passed = lipogram_score == 1.0 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "forbidden": "".join(sorted(self.forbidden)),
                "forbidden_count": forbidden_count,
                "total_letters": total_letters,
            },
        )

    def describe(self) -> str:
        forbidden = "".join(sorted(self.forbidden))
        return f"Lipogram: no letter{'s' if len(self.forbidden) > 1 else ''} {forbidden}"


class Univocalic(Constraint):
    """
    Univocalic: Writing that uses only one vowel.

    Example: "Eggshells" is a univocalic poem using only 'E'.

    Examples:
        >>> uni = Univocalic(vowel="E")
        >>> result = uni.verify(poem)
    """

    name = "Univocalic"
    constraint_type = ConstraintType.SEMANTIC

    VOWELS: ClassVar[set[str]] = set("AEIOU")

    def __init__(
        self,
        vowel: str = "E",
        min_lines: int = 1,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize univocalic constraint.

        Args:
            vowel: The single allowed vowel
            min_lines: Minimum number of lines
            weight: Relative weight for composition
        """
        super().__init__(weight)
        self.allowed_vowel = vowel.upper()
        self.forbidden_vowels = self.VOWELS - {self.allowed_vowel}
        self.min_lines = min_lines

        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=0.5)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check line count
        line_result = self._line_count.verify(poem)

        # Count vowel usage
        text_upper = "".join(structure.lines).upper()
        total_vowels = sum(1 for c in text_upper if c in self.VOWELS)
        correct_vowels = sum(1 for c in text_upper if c == self.allowed_vowel)
        wrong_vowels = total_vowels - correct_vowels

        if total_vowels == 0:
            univocalic_score = 0.0
        else:
            univocalic_score = correct_vowels / total_vowels

        # Combine scores
        score = line_result.score * 0.1 + univocalic_score * 0.9
        passed = wrong_vowels == 0 and line_result.passed and total_vowels > 0

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "allowed_vowel": self.allowed_vowel,
                "correct_vowels": correct_vowels,
                "wrong_vowels": wrong_vowels,
                "total_vowels": total_vowels,
            },
        )

    def describe(self) -> str:
        return f"Univocalic: uses only the vowel '{self.allowed_vowel}'"


class Mesostic(Constraint):
    """
    Mesostic: Word/phrase spelled out by middle letters of lines.

    Like an acrostic, but uses middle letters instead of first letters.
    Invented/popularized by John Cage.

    Examples:
        >>> meso = Mesostic(word="CAGE")
        >>> result = meso.verify(poem)
    """

    name = "Mesostic"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        word: str = "POEM",
        weight: float = 1.0,
    ) -> None:
        """
        Initialize mesostic constraint.

        Args:
            word: The word to spell with middle letters (default: POEM)
            weight: Relative weight for composition
        """
        super().__init__(weight)
        self.target_word = word.upper()
        self.num_lines = len(self.target_word)

        self._line_count = LineCount(self.num_lines, weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check line count
        line_result = self._line_count.verify(poem)

        # For each line, find if the target letter appears in the middle
        # (not as first or last letter)
        matches = 0
        details = []

        for i, (target_letter, line) in enumerate(zip(self.target_word, structure.lines)):
            # Get letters only
            letters = [c.upper() for c in line if c.isalpha()]

            if len(letters) < 3:
                details.append(f"Line {i + 1}: too short")
                continue

            # Check middle letters (not first or last)
            middle_letters = letters[1:-1]

            if target_letter in middle_letters:
                matches += 1
                pos = middle_letters.index(target_letter) + 1  # +1 for human-readable
                details.append(f"Line {i + 1}: ✓ '{target_letter}' at position {pos + 1}")
            else:
                details.append(f"Line {i + 1}: ✗ missing '{target_letter}' in middle")

        # Quadratic penalty for stricter GRPO training
        linear_mesostic = matches / max(1, min(len(self.target_word), len(structure.lines)))
        mesostic_score = linear_mesostic**2

        # Combine scores
        score = line_result.score * 0.1 + mesostic_score * 0.9
        passed = score >= 0.8

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "target_word": self.target_word,
                "matches": matches,
                "expected": len(self.target_word),
                "line_details": details,
            },
        )

    def describe(self) -> str:
        return f"Mesostic: middle letters spell '{self.target_word}'"


class Anaphora(Constraint):
    """
    Anaphora: Repeated word/phrase at the start of successive lines.

    A rhetorical device where lines begin with the same words.
    "I have a dream..." repeated in MLK's speech is famous anaphora.

    Examples:
        >>> ana = Anaphora(phrase="I have a dream")
        >>> result = ana.verify(poem)

        >>> # Auto-detect repeated opening
        >>> ana = Anaphora()
    """

    name = "Anaphora"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        phrase: str | None = None,
        min_repeats: int = 3,
        min_lines: int = 3,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize anaphora constraint.

        Args:
            phrase: The phrase to repeat (None for auto-detect)
            min_repeats: Minimum times the phrase should appear
            min_lines: Minimum total lines
            weight: Relative weight for composition
        """
        super().__init__(weight)
        self.target_phrase = phrase.lower() if phrase else None
        self.min_repeats = min_repeats
        self.min_lines = min_lines

        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=0.5)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check line count
        line_result = self._line_count.verify(poem)

        lines_lower = [line.lower().strip() for line in structure.lines]

        if self.target_phrase:
            # Count lines starting with target phrase
            repeats = sum(1 for line in lines_lower if line.startswith(self.target_phrase))
            detected_phrase = self.target_phrase
        else:
            # Auto-detect: find most common opening word(s)
            openings = []
            for line in lines_lower:
                if line:
                    # Get first 1-3 words
                    words = line.split()[:3]
                    for i in range(1, len(words) + 1):
                        openings.append(" ".join(words[:i]))

            if not openings:
                return VerificationResult(
                    score=0.0,
                    passed=False,
                    rubric=[],
                    constraint_name=self.name,
                    constraint_type=self.constraint_type,
                    details={"error": "No valid line openings found"},
                )

            # Find most common
            from collections import Counter

            opening_counts = Counter(openings)
            detected_phrase, repeats = opening_counts.most_common(1)[0]

        # Score based on repeats
        if repeats >= self.min_repeats:
            anaphora_score = 1.0
        else:
            anaphora_score = repeats / self.min_repeats

        # Combine scores
        score = line_result.score * 0.1 + anaphora_score * 0.9
        passed = repeats >= self.min_repeats and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "phrase": detected_phrase,
                "repeats": repeats,
                "min_repeats": self.min_repeats,
            },
        )

    def describe(self) -> str:
        if self.target_phrase:
            return f"Anaphora: '{self.target_phrase}' repeated {self.min_repeats}+ times"
        return f"Anaphora: repeated opening phrase ({self.min_repeats}+ times)"


class PalindromePoem(Constraint):
    """
    Palindrome Poem: Lines read the same forward and backward.

    Can be:
    - Word palindrome: word order reverses (line 1 = line N)
    - Letter palindrome: each line is a letter palindrome

    Examples:
        >>> pal = PalindromePoem(level="word")
        >>> result = pal.verify(poem)
    """

    name = "Palindrome Poem"
    constraint_type = ConstraintType.SEMANTIC

    def __init__(
        self,
        level: str = "word",  # "word" or "letter"
        min_lines: int = 4,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize palindrome poem constraint.

        Args:
            level: "word" (line order reverses) or "letter" (each line is palindrome)
            min_lines: Minimum number of lines
            weight: Relative weight for composition
        """
        super().__init__(weight)
        self.level = level
        self.min_lines = min_lines

        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=0.5)

    def _is_letter_palindrome(self, text: str) -> bool:
        """Check if text is a letter palindrome."""
        letters = [c.lower() for c in text if c.isalpha()]
        return letters == letters[::-1]

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        # Check line count
        line_result = self._line_count.verify(poem)

        if self.level == "letter":
            # Each line should be a palindrome
            palindrome_lines = sum(
                1 for line in structure.lines if self._is_letter_palindrome(line)
            )
            # Quadratic penalty for stricter GRPO training
            linear_palindrome = palindrome_lines / max(1, len(structure.lines))
            palindrome_score = linear_palindrome**2
        else:
            # Word level: first half of lines mirror second half
            lines = [line.strip().lower() for line in structure.lines]
            n = len(lines)

            if n < 2:
                palindrome_score = 0.0
            else:
                # Compare lines: line[i] should equal line[n-1-i]
                matches = 0
                comparisons = n // 2

                for i in range(comparisons):
                    if lines[i] == lines[n - 1 - i]:
                        matches += 1

                # Quadratic penalty for stricter GRPO training
                linear_palindrome = matches / max(1, comparisons)
                palindrome_score = linear_palindrome**2

        # Combine scores
        score = line_result.score * 0.1 + palindrome_score * 0.9
        passed = palindrome_score >= 0.8 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "level": self.level,
                "palindrome_score": palindrome_score,
            },
        )

    def describe(self) -> str:
        if self.level == "letter":
            return "Palindrome Poem: each line is a letter palindrome"
        return "Palindrome Poem: line order mirrors (word-level)"
