"""
Novel experimental poetic forms for testing instruction-following.

These forms are intentionally unusual and do not correspond to any
traditional or historical poetic conventions. They test a model's
ability to follow arbitrary structural rules rather than relying
on memorized patterns.

Each form has a fictional backstory for flavor, but the constraints
are what matter for evaluation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from abide.constraints import (
    Constraint,
    ConstraintType,
    LineCount,
    NumericBound,
    VerificationResult,
)
from abide.constraints.lexical import (
    AllWordsUnique,
    MonosyllabicOnly,
    WordCount,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class HourglassVerse(Constraint):
    """
    Hourglass Verse: Word count expands then contracts.

    Origin: Said to have emerged from a meditation practice in 1970s
    California, where practitioners wrote poems matching their
    breathing patterns - inhale (expand) and exhale (contract).

    Structure:
    - Line 1: 1 word
    - Line 2: 2 words
    - Line 3: 3 words
    - Line 4: 4 words
    - Line 5: 5 words (widest point)
    - Line 6: 4 words
    - Line 7: 3 words
    - Line 8: 2 words
    - Line 9: 1 word

    Examples:
        >>> form = HourglassVerse()
        >>> result = form.verify(poem)
    """

    name = "Hourglass Verse"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.word_pattern = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        self._line_count = LineCount(9, weight=2.0)
        self._word_count = WordCount(self.word_pattern, tolerance=0, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        word_result = self._word_count.verify(poem)

        score = line_result.score * 0.3 + word_result.score * 0.7
        passed = line_result.passed and word_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "line_count": len(structure.lines),
                "word_pattern": self.word_pattern,
                "word_result": word_result.details,
            },
        )

    def describe(self) -> str:
        return "Hourglass Verse: 9 lines with 1-2-3-4-5-4-3-2-1 words per line"


class PrimeVerse(Constraint):
    """
    Prime Verse: Each line has a prime number of words.

    Origin: Attributed to a mathematician-poet collective in 1980s
    Budapest who believed prime numbers held mystical significance.
    Poems must use only prime word counts.

    Structure:
    - 6 lines minimum
    - Each line must have 2, 3, 5, 7, 11, or 13 words
    - Line N uses prime(N): Line 1 = 2 words, Line 2 = 3, Line 3 = 5...

    Examples:
        >>> form = PrimeVerse()
        >>> result = form.verify(poem)
    """

    name = "Prime Verse"
    constraint_type = ConstraintType.COMPOSITE

    PRIMES: ClassVar[list[int]] = [2, 3, 5, 7, 11, 13]

    def __init__(self, num_lines: int = 6, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_lines = num_lines
        self._line_count = LineCount(num_lines, weight=2.0)
        self._word_count = WordCount(self.PRIMES[:num_lines], tolerance=0, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        word_result = self._word_count.verify(poem)

        score = line_result.score * 0.3 + word_result.score * 0.7
        passed = line_result.passed and word_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "expected_pattern": self.PRIMES[: self.num_lines],
                "word_result": word_result.details,
            },
        )

    def describe(self) -> str:
        primes = self.PRIMES[: self.num_lines]
        return f"Prime Verse: {self.num_lines} lines with {primes} words per line"


class VowelPilgrimage(Constraint):
    """
    Vowel Pilgrimage: Each line begins with successive vowels.

    Origin: Created by a linguist studying phoneme distribution who
    wanted poems that "journeyed" through the vowel space, visiting
    each vowel territory in turn.

    Structure:
    - 5 lines (one for each vowel)
    - Line 1 starts with word beginning 'A'
    - Line 2 starts with word beginning 'E'
    - Line 3 starts with word beginning 'I'
    - Line 4 starts with word beginning 'O'
    - Line 5 starts with word beginning 'U'

    Examples:
        >>> form = VowelPilgrimage()
        >>> result = form.verify(poem)
    """

    name = "Vowel Pilgrimage"
    constraint_type = ConstraintType.COMPOSITE

    VOWELS: ClassVar[list[str]] = ["A", "E", "I", "O", "U"]

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._line_count = LineCount(5, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        # Check each line starts with correct vowel
        matches = 0
        details = []

        for i, line in enumerate(structure.lines[:5]):
            expected = self.VOWELS[i]
            words = line.strip().split()
            if words:
                first_letter = words[0][0].upper() if words[0] else ""
                if first_letter == expected:
                    matches += 1
                    details.append(f"Line {i+1}: ✓ starts with '{expected}'")
                else:
                    details.append(f"Line {i+1}: ✗ expected '{expected}', got '{first_letter}'")
            else:
                details.append(f"Line {i+1}: ✗ empty line")

        vowel_score = matches / 5

        score = line_result.score * 0.3 + vowel_score * 0.7
        passed = matches == 5 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"vowel_matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        return "Vowel Pilgrimage: 5 lines starting with A, E, I, O, U"


class MirrorFrame(Constraint):
    """
    Mirror Frame: First and last lines identical, enclosing the poem.

    Origin: Inspired by the concept of "frame narrative" in literature,
    this form creates a closed loop where the poem ends where it began,
    like a snake eating its tail.

    Structure:
    - Minimum 5 lines
    - First line and last line must be identical
    - Middle lines can vary

    Examples:
        >>> form = MirrorFrame()
        >>> result = form.verify(poem)
    """

    name = "Mirror Frame"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, min_lines: int = 5, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        # Check first and last lines match
        if len(structure.lines) >= 2:
            first = structure.lines[0].strip().lower()
            last = structure.lines[-1].strip().lower()
            frame_match = first == last
            frame_score = 1.0 if frame_match else 0.0
        else:
            frame_match = False
            frame_score = 0.0

        score = line_result.score * 0.3 + frame_score * 0.7
        passed = frame_match and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "first_line": structure.lines[0] if structure.lines else "",
                "last_line": structure.lines[-1] if structure.lines else "",
                "frame_match": frame_match,
            },
        )

    def describe(self) -> str:
        return f"Mirror Frame: {self.min_lines}+ lines, first and last lines identical"


class DescendingStaircase(Constraint):
    """
    Descending Staircase: Each line has one fewer word than the previous.

    Origin: Developed by minimalist poets in 1990s who sought to
    capture the feeling of loss and diminishment in structural form.

    Structure:
    - Line 1: 7 words
    - Line 2: 6 words
    - Line 3: 5 words
    - Line 4: 4 words
    - Line 5: 3 words
    - Line 6: 2 words
    - Line 7: 1 word

    Examples:
        >>> form = DescendingStaircase()
        >>> result = form.verify(poem)
    """

    name = "Descending Staircase"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, start_words: int = 7, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.start_words = start_words
        self.word_pattern = list(range(start_words, 0, -1))
        self._line_count = LineCount(start_words, weight=2.0)
        self._word_count = WordCount(self.word_pattern, tolerance=0, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        word_result = self._word_count.verify(poem)

        score = line_result.score * 0.3 + word_result.score * 0.7
        passed = line_result.passed and word_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "word_pattern": self.word_pattern,
                "word_result": word_result.details,
            },
        )

    def describe(self) -> str:
        return (
            f"Descending Staircase: {self.start_words} lines with {self.start_words} down to 1 word"
        )


class QuestionQuest(Constraint):
    """
    Question Quest: Every line must be a question.

    Origin: Attributed to Socratic poetry circles who believed that
    knowledge could only be expressed through inquiry, never statement.

    Structure:
    - Minimum 4 lines
    - Every line must end with a question mark
    - Each line should be phrased as a question

    Examples:
        >>> form = QuestionQuest()
        >>> result = form.verify(poem)
    """

    name = "Question Quest"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, min_lines: int = 4, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        # Check each line ends with ?
        questions = 0
        details = []
        for i, line in enumerate(structure.lines):
            stripped = line.strip()
            if stripped.endswith("?"):
                questions += 1
                details.append(f"Line {i+1}: ✓ is question")
            else:
                details.append(f"Line {i+1}: ✗ not a question")

        question_score = questions / max(1, len(structure.lines))

        score = line_result.score * 0.3 + question_score * 0.7
        passed = question_score == 1.0 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"questions": questions, "line_details": details},
        )

    def describe(self) -> str:
        return f"Question Quest: {self.min_lines}+ lines, every line ends with '?'"


class WhisperPoem(Constraint):
    """
    Whisper Poem: Very short lines, as if whispered.

    Origin: From a tradition of "constraint minimalism" where poets
    tried to express profound ideas in the smallest possible space,
    like passing secrets.

    Structure:
    - 6 lines minimum
    - Each line maximum 20 characters (including spaces)
    - Creates intimate, secretive feeling

    Examples:
        >>> form = WhisperPoem()
        >>> result = form.verify(poem)
    """

    name = "Whisper Poem"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, max_chars: int = 20, min_lines: int = 6, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.max_chars = max_chars
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        # Check each line is short enough
        short_lines = 0
        details = []
        for i, line in enumerate(structure.lines):
            length = len(line.strip())
            if length <= self.max_chars:
                short_lines += 1
                details.append(f"Line {i+1}: ✓ {length} chars")
            else:
                details.append(f"Line {i+1}: ✗ {length} chars (max {self.max_chars})")

        char_score = short_lines / max(1, len(structure.lines))

        score = line_result.score * 0.3 + char_score * 0.7
        passed = char_score == 1.0 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"short_lines": short_lines, "line_details": details},
        )

    def describe(self) -> str:
        return f"Whisper Poem: {self.min_lines}+ lines, each max {self.max_chars} characters"


class ThunderVerse(Constraint):
    """
    Thunder Verse: Long, booming lines that fill the space.

    Origin: The opposite of Whisper Poem, Thunder Verse emerged from
    performance poets who wanted lines that demanded to be shouted.

    Structure:
    - 4 lines minimum
    - Each line minimum 50 characters
    - Creates overwhelming, powerful feeling

    Examples:
        >>> form = ThunderVerse()
        >>> result = form.verify(poem)
    """

    name = "Thunder Verse"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, min_chars: int = 50, min_lines: int = 4, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.min_chars = min_chars
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        # Check each line is long enough
        long_lines = 0
        details = []
        for i, line in enumerate(structure.lines):
            length = len(line.strip())
            if length >= self.min_chars:
                long_lines += 1
                details.append(f"Line {i+1}: ✓ {length} chars")
            else:
                details.append(f"Line {i+1}: ✗ {length} chars (min {self.min_chars})")

        char_score = long_lines / max(1, len(structure.lines))

        score = line_result.score * 0.3 + char_score * 0.7
        passed = char_score == 1.0 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"long_lines": long_lines, "line_details": details},
        )

    def describe(self) -> str:
        return f"Thunder Verse: {self.min_lines}+ lines, each min {self.min_chars} characters"


class AlphabeticTerminus(Constraint):
    """
    Alphabetic Terminus: Last word of each line follows the alphabet.

    Origin: A reverse acrostic created by a typesetter who was bored
    and decided to hide messages in the final words of lines.

    Structure:
    - 26 lines (or fewer for partial alphabet)
    - Last word of line 1 starts with 'A'
    - Last word of line 2 starts with 'B'
    - And so on...

    Examples:
        >>> form = AlphabeticTerminus()
        >>> result = form.verify(poem)

        >>> # Partial alphabet
        >>> form = AlphabeticTerminus(letters="ABCDEF")
    """

    name = "Alphabetic Terminus"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, letters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", weight: float = 1.0) -> None:
        super().__init__(weight)
        self.letters = letters.upper()
        self._line_count = LineCount(len(self.letters), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        matches = 0
        details = []

        for i, (letter, line) in enumerate(zip(self.letters, structure.lines)):
            words = re.findall(r"\b[a-zA-Z]+\b", line)
            if words:
                last_word = words[-1]
                first_char = last_word[0].upper()
                if first_char == letter:
                    matches += 1
                    details.append(f"Line {i+1}: ✓ ends with '{last_word}'")
                else:
                    details.append(f"Line {i+1}: ✗ expected '{letter}', got '{first_char}'")
            else:
                details.append(f"Line {i+1}: ✗ no words found")

        alpha_score = matches / max(1, len(self.letters))

        score = line_result.score * 0.3 + alpha_score * 0.7
        passed = matches == len(self.letters) and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        if len(self.letters) == 26:
            return "Alphabetic Terminus: last words follow A-Z"
        return f"Alphabetic Terminus: last words start {self.letters[:3]}... ({len(self.letters)} lines)"


class OddEvenDance(Constraint):
    """
    Odd-Even Dance: Alternating line lengths based on position.

    Origin: Inspired by the mathematical beauty of parity, this form
    creates a rhythmic alternation between short and long lines.

    Structure:
    - Odd lines (1, 3, 5...): exactly 3 words
    - Even lines (2, 4, 6...): exactly 6 words
    - Minimum 6 lines

    Examples:
        >>> form = OddEvenDance()
        >>> result = form.verify(poem)
    """

    name = "Odd-Even Dance"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self, odd_words: int = 3, even_words: int = 6, min_lines: int = 6, weight: float = 1.0
    ) -> None:
        super().__init__(weight)
        self.odd_words = odd_words
        self.even_words = even_words
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        matches = 0
        details = []

        for i, line in enumerate(structure.lines):
            words = line.split()
            expected = self.odd_words if (i + 1) % 2 == 1 else self.even_words
            actual = len(words)

            if actual == expected:
                matches += 1
                details.append(f"Line {i+1} ({'odd' if (i+1)%2==1 else 'even'}): ✓ {actual} words")
            else:
                details.append(
                    f"Line {i+1} ({'odd' if (i+1)%2==1 else 'even'}): ✗ {actual} (expected {expected})"
                )

        dance_score = matches / max(1, len(structure.lines))

        score = line_result.score * 0.3 + dance_score * 0.7
        passed = dance_score == 1.0 and line_result.passed

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
            f"Odd-Even Dance: odd lines {self.odd_words} words, "
            f"even lines {self.even_words} words"
        )


class NumericalEcho(Constraint):
    """
    Numerical Echo: Word counts follow digits of pi (3-1-4-1-5-9-2-6).

    Origin: Created by a mathematical poet who believed pi's infinite
    non-repeating decimal contained all possible poems encoded within.

    Structure:
    - 8 lines
    - Words per line: 3, 1, 4, 1, 5, 9, 2, 6

    Examples:
        >>> form = NumericalEcho()
        >>> result = form.verify(poem)
    """

    name = "Numerical Echo"
    constraint_type = ConstraintType.COMPOSITE

    PI_DIGITS: ClassVar[list[int]] = [3, 1, 4, 1, 5, 9, 2, 6]

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._line_count = LineCount(8, weight=2.0)
        self._word_count = WordCount(self.PI_DIGITS, tolerance=0, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        word_result = self._word_count.verify(poem)

        score = line_result.score * 0.3 + word_result.score * 0.7
        passed = line_result.passed and word_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "pi_pattern": self.PI_DIGITS,
                "word_result": word_result.details,
            },
        )

    def describe(self) -> str:
        return "Numerical Echo: 8 lines with 3-1-4-1-5-9-2-6 words (pi digits)"


class ColorSpectrum(Constraint):
    """
    Color Spectrum: Each line must contain a color from the rainbow.

    Origin: Developed by synesthetic poets who experienced colors when
    reading words, wanting to create poems that painted rainbows.

    Structure:
    - 7 lines
    - Line 1 contains "red"
    - Line 2 contains "orange"
    - Line 3 contains "yellow"
    - Line 4 contains "green"
    - Line 5 contains "blue"
    - Line 6 contains "indigo"
    - Line 7 contains "violet"

    Examples:
        >>> form = ColorSpectrum()
        >>> result = form.verify(poem)
    """

    name = "Color Spectrum"
    constraint_type = ConstraintType.COMPOSITE

    COLORS: ClassVar[list[str]] = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._line_count = LineCount(7, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        matches = 0
        details = []

        for i, (color, line) in enumerate(zip(self.COLORS, structure.lines)):
            line_lower = line.lower()
            if color in line_lower:
                matches += 1
                details.append(f"Line {i+1}: ✓ contains '{color}'")
            else:
                details.append(f"Line {i+1}: ✗ missing '{color}'")

        color_score = matches / 7

        score = line_result.score * 0.3 + color_score * 0.7
        passed = matches == 7 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"color_matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        return "Color Spectrum: 7 lines containing red, orange, yellow, green, blue, indigo, violet"


class ElementalVerse(Constraint):
    """
    Elemental Verse: Each line must contain a periodic table element.

    Origin: From a chemistry professor who wrote poems as mnemonics
    for students, each line encoding an element name.

    Structure:
    - 5 lines minimum
    - Each line must contain a different element name
    - Common elements: gold, silver, iron, copper, lead, zinc...

    Examples:
        >>> form = ElementalVerse()
        >>> result = form.verify(poem)
    """

    name = "Elemental Verse"
    constraint_type = ConstraintType.COMPOSITE

    ELEMENTS: ClassVar[set[str]] = {
        "hydrogen",
        "helium",
        "lithium",
        "carbon",
        "nitrogen",
        "oxygen",
        "neon",
        "sodium",
        "magnesium",
        "aluminum",
        "silicon",
        "phosphorus",
        "sulfur",
        "chlorine",
        "argon",
        "potassium",
        "calcium",
        "iron",
        "copper",
        "zinc",
        "silver",
        "tin",
        "gold",
        "mercury",
        "lead",
        "uranium",
        "platinum",
        "nickel",
        "cobalt",
        "titanium",
    }

    def __init__(self, min_lines: int = 5, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        matches = 0
        details = []
        found_elements = set()

        for i, line in enumerate(structure.lines):
            line_lower = line.lower()
            found = None
            for element in self.ELEMENTS:
                if element in line_lower and element not in found_elements:
                    found = element
                    found_elements.add(element)
                    break

            if found:
                matches += 1
                details.append(f"Line {i+1}: ✓ contains '{found}'")
            else:
                details.append(f"Line {i+1}: ✗ no unique element found")

        element_score = matches / max(1, len(structure.lines))

        score = line_result.score * 0.3 + element_score * 0.7
        passed = element_score >= 0.8 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"element_matches": matches, "found_elements": list(found_elements)},
        )

    def describe(self) -> str:
        return f"Elemental Verse: {self.min_lines}+ lines, each with a periodic element name"


class NumberWord(Constraint):
    """
    Number Word: Each line contains a number word in sequence.

    Origin: A children's poet created this form to help kids learn
    to count while reading poetry.

    Structure:
    - 10 lines
    - Line 1 contains "one"
    - Line 2 contains "two"
    - ...up to "ten"

    Examples:
        >>> form = NumberWord()
        >>> result = form.verify(poem)
    """

    name = "Number Word"
    constraint_type = ConstraintType.COMPOSITE

    NUMBERS: ClassVar[list[str]] = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ]

    def __init__(self, num_lines: int = 10, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_lines = min(num_lines, 10)
        self._line_count = LineCount(self.num_lines, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        matches = 0
        details = []

        for i, line in enumerate(structure.lines[: self.num_lines]):
            number = self.NUMBERS[i]
            # Use word boundary to avoid partial matches
            pattern = r"\b" + number + r"\b"
            if re.search(pattern, line.lower()):
                matches += 1
                details.append(f"Line {i+1}: ✓ contains '{number}'")
            else:
                details.append(f"Line {i+1}: ✗ missing '{number}'")

        number_score = matches / self.num_lines

        score = line_result.score * 0.3 + number_score * 0.7
        passed = matches == self.num_lines and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"number_matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        return f"Number Word: {self.num_lines} lines containing 'one' through '{self.NUMBERS[self.num_lines-1]}'"


class TemporalVerse(Constraint):
    """
    Temporal Verse: Each line contains a time-related word.

    Origin: Created by a poet obsessed with the passage of time,
    each line must acknowledge temporality.

    Structure:
    - 6 lines minimum
    - Each line contains a time word: hour, minute, second, day, week,
      month, year, moment, instant, eternity, forever, always, never, etc.

    Examples:
        >>> form = TemporalVerse()
        >>> result = form.verify(poem)
    """

    name = "Temporal Verse"
    constraint_type = ConstraintType.COMPOSITE

    TIME_WORDS: ClassVar[set[str]] = {
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "moment",
        "instant",
        "eternity",
        "forever",
        "always",
        "never",
        "time",
        "clock",
        "dawn",
        "dusk",
        "morning",
        "evening",
        "night",
        "noon",
        "midnight",
        "today",
        "tomorrow",
        "yesterday",
        "now",
        "then",
        "past",
        "present",
        "future",
        "century",
        "decade",
        "age",
        "era",
    }

    def __init__(self, min_lines: int = 6, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        matches = 0
        details = []

        for i, line in enumerate(structure.lines):
            line_lower = line.lower()
            found = None
            for word in self.TIME_WORDS:
                pattern = r"\b" + word + r"\b"
                if re.search(pattern, line_lower):
                    found = word
                    break

            if found:
                matches += 1
                details.append(f"Line {i+1}: ✓ contains '{found}'")
            else:
                details.append(f"Line {i+1}: ✗ no time word found")

        time_score = matches / max(1, len(structure.lines))

        score = line_result.score * 0.3 + time_score * 0.7
        passed = time_score == 1.0 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"time_matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        return f"Temporal Verse: {self.min_lines}+ lines, each with a time-related word"


class ExclamationEcho(Constraint):
    """
    Exclamation Echo: Every line ends with exclamation.

    Origin: From slam poets who wanted every line to demand attention,
    to be shouted rather than whispered.

    Structure:
    - 5 lines minimum
    - Every line must end with "!"

    Examples:
        >>> form = ExclamationEcho()
        >>> result = form.verify(poem)
    """

    name = "Exclamation Echo"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, min_lines: int = 5, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        exclamations = 0
        details = []

        for i, line in enumerate(structure.lines):
            if line.strip().endswith("!"):
                exclamations += 1
                details.append(f"Line {i+1}: ✓ ends with '!'")
            else:
                details.append(f"Line {i+1}: ✗ missing '!'")

        exclaim_score = exclamations / max(1, len(structure.lines))

        score = line_result.score * 0.3 + exclaim_score * 0.7
        passed = exclaim_score == 1.0 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"exclamations": exclamations, "line_details": details},
        )

    def describe(self) -> str:
        return f"Exclamation Echo: {self.min_lines}+ lines, all ending with '!'"


class MonotoneMountain(Constraint):
    """
    Monotone Mountain: All one-syllable words forming ascending pattern.

    Origin: A minimalist experiment combining monosyllabic constraint
    with ascending word counts.

    Structure:
    - 5 lines
    - Line 1: 2 words, Line 2: 3 words, Line 3: 4 words, Line 4: 3 words, Line 5: 2 words
    - ALL words must be one syllable

    Examples:
        >>> form = MonotoneMountain()
        >>> result = form.verify(poem)
    """

    name = "Monotone Mountain"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.word_pattern = [2, 3, 4, 3, 2]
        self._line_count = LineCount(5, weight=2.0)
        self._word_count = WordCount(self.word_pattern, tolerance=0, weight=1.5)
        self._monosyllabic = MonosyllabicOnly(min_words=10, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        word_result = self._word_count.verify(poem)
        mono_result = self._monosyllabic.verify(poem)

        score = line_result.score * 0.2 + word_result.score * 0.4 + mono_result.score * 0.4
        passed = line_result.passed and word_result.passed and mono_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "word_pattern": self.word_pattern,
                "word_result": word_result.details,
                "mono_result": mono_result.details,
            },
        )

    def describe(self) -> str:
        return "Monotone Mountain: 5 lines (2-3-4-3-2 words), ALL words one syllable"


class UniqueUtterance(Constraint):
    """
    Unique Utterance: No word may appear twice in the entire poem.

    Origin: From Oulipo experiments in constraint-based writing, where
    repetition was forbidden to force creative vocabulary use.

    Structure:
    - 6 lines minimum
    - No word appears more than once
    - Tests vocabulary breadth

    Examples:
        >>> form = UniqueUtterance()
        >>> result = form.verify(poem)
    """

    name = "Unique Utterance"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, min_lines: int = 6, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)
        self._unique = AllWordsUnique(min_words=20, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        unique_result = self._unique.verify(poem)

        score = line_result.score * 0.3 + unique_result.score * 0.7
        passed = line_result.passed and unique_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "line_count": len(structure.lines),
                "unique_result": unique_result.details,
            },
        )

    def describe(self) -> str:
        return f"Unique Utterance: {self.min_lines}+ lines, no word repeats"


class BinaryBeat(Constraint):
    """
    Binary Beat: Word counts follow binary pattern.

    Origin: From digital poets who wanted to encode binary data
    in poetic form, 1s and 0s becoming words and pauses.

    Structure:
    - 8 lines
    - Word counts: 1, 0, 1, 1, 0, 1, 0, 0 (binary for 172)
    - 0-word lines are empty or just punctuation

    Note: This form uses the pattern 2-1-2-2-1-2-1-1 since
    0-word lines are unusual. Each digit +1 gives word count.

    Examples:
        >>> form = BinaryBeat()
        >>> result = form.verify(poem)
    """

    name = "Binary Beat"
    constraint_type = ConstraintType.COMPOSITE

    # Original binary: 10110100, we add 1 to each for word count
    PATTERN: ClassVar[list[int]] = [2, 1, 2, 2, 1, 2, 1, 1]

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._line_count = LineCount(8, weight=2.0)
        self._word_count = WordCount(self.PATTERN, tolerance=0, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        word_result = self._word_count.verify(poem)

        score = line_result.score * 0.3 + word_result.score * 0.7
        passed = line_result.passed and word_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "binary_pattern": self.PATTERN,
                "word_result": word_result.details,
            },
        )

    def describe(self) -> str:
        return "Binary Beat: 8 lines with 2-1-2-2-1-2-1-1 words (binary pattern)"


class ConsonantCascade(Constraint):
    """
    Consonant Cascade: Consecutive lines cannot start with same consonant.

    Origin: A phonetics exercise turned poetic form, designed to
    create maximum consonant variety in line openings.

    Structure:
    - 8 lines minimum
    - No two consecutive lines can start with words beginning
      with the same consonant
    - Vowel starts are allowed to repeat

    Examples:
        >>> form = ConsonantCascade()
        >>> result = form.verify(poem)
    """

    name = "Consonant Cascade"
    constraint_type = ConstraintType.COMPOSITE

    VOWELS: ClassVar[set[str]] = {"A", "E", "I", "O", "U"}

    def __init__(self, min_lines: int = 8, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        # Get first consonant of each line
        first_consonants = []
        for line in structure.lines:
            words = re.findall(r"\b[a-zA-Z]+\b", line)
            if words:
                first_letter = words[0][0].upper()
                # Only track consonants
                if first_letter not in self.VOWELS:
                    first_consonants.append(first_letter)
                else:
                    first_consonants.append(None)  # Vowel, no constraint
            else:
                first_consonants.append(None)

        # Check no consecutive consonants match
        violations = 0
        details = []

        for i in range(1, len(first_consonants)):
            prev = first_consonants[i - 1]
            curr = first_consonants[i]

            if prev is not None and curr is not None and prev == curr:
                violations += 1
                details.append(f"Lines {i}/{i+1}: ✗ both start with '{prev}'")
            else:
                details.append(f"Lines {i}/{i+1}: ✓ no consonant repeat")

        if len(first_consonants) <= 1:
            cascade_score = 0.0
        else:
            cascade_score = 1.0 - (violations / (len(first_consonants) - 1))

        score = line_result.score * 0.3 + cascade_score * 0.7
        passed = violations == 0 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"violations": violations, "line_details": details},
        )

    def describe(self) -> str:
        return f"Consonant Cascade: {self.min_lines}+ lines, no consecutive consonant repeats"


class SandwichSonnet(Constraint):
    """
    Sandwich Sonnet: First and last couplets are identical.

    Origin: A structural experiment where the poem is "sandwiched"
    between identical bookends, like bread around a filling.

    Structure:
    - 8 lines minimum
    - Lines 1-2 must equal lines (N-1)-N
    - Middle lines can vary

    Examples:
        >>> form = SandwichSonnet()
        >>> result = form.verify(poem)
    """

    name = "Sandwich Sonnet"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, min_lines: int = 8, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        n = len(structure.lines)
        if n < 4:
            sandwich_score = 0.0
            matches = 0
        else:
            # Compare first two with last two
            first_couplet = [line.strip().lower() for line in structure.lines[:2]]
            last_couplet = [line.strip().lower() for line in structure.lines[-2:]]

            matches = sum(1 for a, b in zip(first_couplet, last_couplet) if a == b)
            sandwich_score = matches / 2

        score = line_result.score * 0.3 + sandwich_score * 0.7
        passed = matches == 2 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "sandwich_matches": matches,
                "first_two": structure.lines[:2] if len(structure.lines) >= 2 else [],
                "last_two": structure.lines[-2:] if len(structure.lines) >= 2 else [],
            },
        )

    def describe(self) -> str:
        return f"Sandwich Sonnet: {self.min_lines}+ lines, first 2 lines = last 2 lines"


class VoidVerse(Constraint):
    """
    Void Verse: Every line contains an absence word.

    Origin: From nihilist poets exploring themes of emptiness and
    absence through structural constraint.

    Structure:
    - 5 lines minimum
    - Every line must contain: nothing, empty, void, absence, hollow,
      blank, zero, none, null, lack, missing, gone, lost, etc.

    Examples:
        >>> form = VoidVerse()
        >>> result = form.verify(poem)
    """

    name = "Void Verse"
    constraint_type = ConstraintType.COMPOSITE

    VOID_WORDS: ClassVar[set[str]] = {
        "nothing",
        "empty",
        "void",
        "absence",
        "hollow",
        "blank",
        "zero",
        "none",
        "null",
        "lack",
        "missing",
        "gone",
        "lost",
        "vacant",
        "barren",
        "bare",
        "without",
        "nil",
        "nought",
        "emptiness",
        "vacuum",
        "abyss",
        "darkness",
        "shadow",
    }

    def __init__(self, min_lines: int = 5, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        matches = 0
        details = []

        for i, line in enumerate(structure.lines):
            line_lower = line.lower()
            found = None
            for word in self.VOID_WORDS:
                pattern = r"\b" + word + r"\b"
                if re.search(pattern, line_lower):
                    found = word
                    break

            if found:
                matches += 1
                details.append(f"Line {i+1}: ✓ contains '{found}'")
            else:
                details.append(f"Line {i+1}: ✗ no void word found")

        void_score = matches / max(1, len(structure.lines))

        score = line_result.score * 0.3 + void_score * 0.7
        passed = void_score == 1.0 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"void_matches": matches, "line_details": details},
        )

    def describe(self) -> str:
        return f"Void Verse: {self.min_lines}+ lines, each with an absence word (nothing, empty, void...)"


class GoldenRatio(Constraint):
    """
    Golden Ratio: Word counts approximate Fibonacci sequence.

    Origin: Inspired by the mathematical beauty found in nature,
    this form uses Fibonacci numbers (1, 1, 2, 3, 5, 8) for word counts.

    Structure:
    - 6 lines
    - Words per line: 1, 1, 2, 3, 5, 8

    Examples:
        >>> form = GoldenRatio()
        >>> result = form.verify(poem)
    """

    name = "Golden Ratio"
    constraint_type = ConstraintType.COMPOSITE

    FIB: ClassVar[list[int]] = [1, 1, 2, 3, 5, 8]

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._line_count = LineCount(6, weight=2.0)
        self._word_count = WordCount(self.FIB, tolerance=0, weight=2.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)
        word_result = self._word_count.verify(poem)

        score = line_result.score * 0.3 + word_result.score * 0.7
        passed = line_result.passed and word_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "fibonacci_pattern": self.FIB,
                "word_result": word_result.details,
            },
        )

    def describe(self) -> str:
        return "Golden Ratio: 6 lines with 1-1-2-3-5-8 words (Fibonacci)"


class EchoEnd(Constraint):
    """
    Echo End: Last word of each line starts with same letter.

    Origin: A constraint form where all line endings share an
    initial letter, creating a subtle echo effect.

    Structure:
    - 5 lines minimum
    - Last word of every line starts with same letter
    - Letter can be specified or detected

    Examples:
        >>> form = EchoEnd(letter="S")
        >>> result = form.verify(poem)
    """

    name = "Echo End"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, letter: str | None = None, min_lines: int = 5, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.letter = letter.upper() if letter else None
        self.min_lines = min_lines
        self._line_count = LineCount(NumericBound.at_least(min_lines), weight=1.0)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        line_result = self._line_count.verify(poem)

        # Get last word first letters
        last_word_firsts = []
        for line in structure.lines:
            words = re.findall(r"\b[a-zA-Z]+\b", line)
            if words:
                last_word_firsts.append(words[-1][0].upper())

        if not last_word_firsts:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": "No words found"},
            )

        # Determine target letter
        if self.letter:
            target = self.letter
        else:
            # Auto-detect most common
            from collections import Counter

            target = Counter(last_word_firsts).most_common(1)[0][0]

        matches = sum(1 for ltr in last_word_firsts if ltr == target)
        echo_score = matches / len(last_word_firsts)

        score = line_result.score * 0.3 + echo_score * 0.7
        passed = echo_score == 1.0 and line_result.passed

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "target_letter": target,
                "matches": matches,
                "total": len(last_word_firsts),
            },
        )

    def describe(self) -> str:
        if self.letter:
            return f"Echo End: {self.min_lines}+ lines, all end-words start with '{self.letter}'"
        return f"Echo End: {self.min_lines}+ lines, all end-words start with same letter"
