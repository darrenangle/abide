"""
Mathematical poetic forms for poet-mathematicians.

These forms embed mathematical sequences, number theory concepts,
and geometric relationships into poetic structure. They test both
instruction-following AND mathematical reasoning.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from abide.constraints import (
    Constraint,
    ConstraintType,
    VerificationResult,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    return all(n % i != 0 for i in range(3, int(math.sqrt(n)) + 1, 2))


def fibonacci_sequence(n: int) -> list[int]:
    """Generate first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs[:n]


def triangular_number(n: int) -> int:
    """Return nth triangular number (1, 3, 6, 10, 15...)."""
    return n * (n + 1) // 2


def gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


class FibonacciVerse(Constraint):
    """
    Each line has Fibonacci-many words: 1, 1, 2, 3, 5, 8, 13...

    A form that embodies the golden spiral in its structure.
    The word counts grow according to nature's favorite sequence.
    """

    name = "FibonacciVerse"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, num_lines: int = 7, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_lines = num_lines
        self.expected = fibonacci_sequence(num_lines)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if len(structure.lines) != self.num_lines:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": f"Expected {self.num_lines} lines, got {len(structure.lines)}"},
            )

        matches = 0
        details = []
        for i, line in enumerate(structure.lines):
            words = len(line.split())
            expected = self.expected[i]
            if words == expected:
                matches += 1
                details.append(f"Line {i + 1}: ✓ {words} words (Fib[{i + 1}])")
            else:
                details.append(f"Line {i + 1}: ✗ {words} words (expected {expected})")

        # Steep penalty for GRPO training: 0 errors=1.0, 1 error=0.5, 2 errors=0.25, 3+=0.05
        violations = self.num_lines - matches
        if violations == 0:
            score = 1.0
        elif violations == 1:
            score = 0.5
        elif violations == 2:
            score = 0.25
        else:
            score = 0.05
        return VerificationResult(
            score=score,
            passed=violations == 0,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"expected": self.expected, "line_details": details},
        )

    def describe(self) -> str:
        seq = ", ".join(str(x) for x in self.expected)
        return (
            f"FibonacciVerse: Write {self.num_lines} lines where line N has "
            f"the Nth Fibonacci number of words. Sequence: {seq}"
        )


class GoldenRatioVerse(Constraint):
    """
    Consecutive line lengths approximate the golden ratio (φ ≈ 1.618).

    Each line should have approximately φ times as many characters
    as the previous line, creating a golden spiral in text.
    """

    name = "GoldenRatioVerse"
    constraint_type = ConstraintType.COMPOSITE

    PHI: ClassVar[float] = 1.618033988749895

    def __init__(self, num_lines: int = 5, tolerance: float = 0.15, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_lines = num_lines
        self.tolerance = tolerance  # Allow 15% deviation from phi

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if len(structure.lines) < self.num_lines:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": f"Expected at least {self.num_lines} lines"},
            )

        matches = 0
        details = []
        lengths = [len(line) for line in structure.lines[: self.num_lines]]

        details.append(f"Line 1: {lengths[0]} chars (base)")

        for i in range(1, len(lengths)):
            if lengths[i - 1] == 0:
                details.append(f"Line {i + 1}: ✗ previous line empty")
                continue

            actual_ratio = lengths[i] / lengths[i - 1]
            deviation = abs(actual_ratio - self.PHI) / self.PHI

            if deviation <= self.tolerance:
                matches += 1
                details.append(f"Line {i + 1}: ✓ {lengths[i]} chars (ratio {actual_ratio:.3f} ≈ φ)")
            else:
                details.append(
                    f"Line {i + 1}: ✗ {lengths[i]} chars (ratio {actual_ratio:.3f}, "
                    f"expected ~{self.PHI:.3f})"
                )

        # Steep penalty for GRPO training: 0 errors=1.0, 1 error=0.5, 2 errors=0.25, 3+=0.05
        total_checks = self.num_lines - 1  # n-1 ratios to check
        violations = total_checks - matches
        if violations == 0:
            score = 1.0
        elif violations == 1:
            score = 0.5
        elif violations == 2:
            score = 0.25
        else:
            score = 0.05
        return VerificationResult(
            score=score,
            passed=violations == 0,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"lengths": lengths, "line_details": details},
        )

    def describe(self) -> str:
        return (
            f"GoldenRatioVerse: Write {self.num_lines} lines where each line is "
            f"approximately φ (1.618) times longer than the previous. "
            f"Creates a golden spiral in text."
        )


class TriangularVerse(Constraint):
    """
    Word counts follow triangular numbers: 1, 3, 6, 10, 15, 21...

    Each line N has N(N+1)/2 words, creating a triangular structure.
    """

    name = "TriangularVerse"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, num_lines: int = 5, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_lines = num_lines
        self.expected = [triangular_number(i + 1) for i in range(num_lines)]

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if len(structure.lines) != self.num_lines:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": f"Expected {self.num_lines} lines"},
            )

        matches = 0
        details = []
        for i, line in enumerate(structure.lines):
            words = len(line.split())
            expected = self.expected[i]
            if words == expected:
                matches += 1
                details.append(f"Line {i + 1}: ✓ {words} words (T[{i + 1}])")
            else:
                details.append(f"Line {i + 1}: ✗ {words} words (expected {expected})")

        # Steep penalty for GRPO training: 0 errors=1.0, 1 error=0.5, 2 errors=0.25, 3+=0.05
        violations = self.num_lines - matches
        if violations == 0:
            score = 1.0
        elif violations == 1:
            score = 0.5
        elif violations == 2:
            score = 0.25
        else:
            score = 0.05
        return VerificationResult(
            score=score,
            passed=violations == 0,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"expected": self.expected, "line_details": details},
        )

    def describe(self) -> str:
        seq = ", ".join(str(x) for x in self.expected)
        return (
            f"TriangularVerse: Write {self.num_lines} lines where line N has "
            f"the Nth triangular number of words. Sequence: {seq}"
        )


class PythagoreanTercet(Constraint):
    """
    Three-line stanzas where word counts form Pythagorean triples.

    Each stanza's word counts (a, b, c) must satisfy a² + b² = c².
    Classic triples: (3,4,5), (5,12,13), (8,15,17), (7,24,25).
    """

    name = "PythagoreanTercet"
    constraint_type = ConstraintType.COMPOSITE

    # Common Pythagorean triples (sorted ascending)
    TRIPLES: ClassVar[list[tuple[int, int, int]]] = [
        (3, 4, 5),
        (5, 12, 13),
        (8, 15, 17),
        (7, 24, 25),
    ]

    def __init__(self, num_stanzas: int = 1, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_stanzas = num_stanzas

    def _is_pythagorean(self, a: int, b: int, c: int) -> bool:
        """Check if (a,b,c) form a Pythagorean triple (order-independent)."""
        nums = sorted([a, b, c])
        return nums[0] ** 2 + nums[1] ** 2 == nums[2] ** 2

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        expected_lines = self.num_stanzas * 3
        if len(structure.lines) < expected_lines:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": f"Expected at least {expected_lines} lines"},
            )

        matches = 0
        details = []

        for s in range(self.num_stanzas):
            start = s * 3
            words = [len(structure.lines[start + i].split()) for i in range(3)]
            a, b, c = words

            if self._is_pythagorean(a, b, c):
                matches += 1
                details.append(
                    f"Stanza {s + 1}: ✓ ({a}, {b}, {c}) - "
                    f"{min(words)}² + {sorted(words)[1]}² = {max(words)}²"
                )
            else:
                sorted_w = sorted(words)
                lhs = sorted_w[0] ** 2 + sorted_w[1] ** 2
                rhs = sorted_w[2] ** 2
                details.append(
                    f"Stanza {s + 1}: ✗ ({a}, {b}, {c}) - "
                    f"{sorted_w[0]}² + {sorted_w[1]}² = {lhs} ≠ {rhs}"
                )

        # Steep penalty for GRPO training: 0 errors=1.0, 1 error=0.5, 2 errors=0.25, 3+=0.05
        violations = self.num_stanzas - matches
        if violations == 0:
            score = 1.0
        elif violations == 1:
            score = 0.5
        elif violations == 2:
            score = 0.25
        else:
            score = 0.05
        return VerificationResult(
            score=score,
            passed=violations == 0,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"stanza_details": details},
        )

    def describe(self) -> str:
        examples = ", ".join(f"({a},{b},{c})" for a, b, c in self.TRIPLES[:3])
        return (
            f"PythagoreanTercet: Write {self.num_stanzas} three-line stanza(s) where "
            f"the word counts form a Pythagorean triple (a² + b² = c²). "
            f"Examples: {examples}"
        )


class CoprimeVerse(Constraint):
    """
    Adjacent lines must have coprime word counts (GCD = 1).

    Forces careful selection of word counts that share no common factors.
    """

    name = "CoprimeVerse"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, num_lines: int = 5, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_lines = num_lines

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if len(structure.lines) < self.num_lines:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": f"Expected at least {self.num_lines} lines"},
            )

        matches = 0
        details = []
        word_counts = [len(line.split()) for line in structure.lines[: self.num_lines]]

        details.append(f"Line 1: {word_counts[0]} words")

        for i in range(1, len(word_counts)):
            a, b = word_counts[i - 1], word_counts[i]
            g = gcd(a, b)

            if g == 1:
                matches += 1
                details.append(f"Line {i + 1}: ✓ {b} words (coprime with {a})")
            else:
                details.append(f"Line {i + 1}: ✗ {b} words (GCD({a},{b}) = {g})")

        # Steep penalty for GRPO training: 0 errors=1.0, 1 error=0.5, 2 errors=0.25, 3+=0.05
        total_checks = self.num_lines - 1
        violations = total_checks - matches
        if violations == 0:
            score = 1.0
        elif violations == 1:
            score = 0.5
        elif violations == 2:
            score = 0.25
        else:
            score = 0.05
        return VerificationResult(
            score=score,
            passed=violations == 0,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"word_counts": word_counts, "line_details": details},
        )

    def describe(self) -> str:
        return (
            f"CoprimeVerse: Write {self.num_lines} lines where adjacent lines have "
            f"COPRIME word counts (GCD = 1). Example: 3,5,7,4,9 works; 4,6 fails (GCD=2)."
        )


class PiKu(Constraint):
    """
    Like haiku, but syllables follow digits of pi: 3-1-4-1-5-9-2-6...

    A meditation on the infinite and irrational through verse.
    """

    name = "PiKu"
    constraint_type = ConstraintType.COMPOSITE

    # Digits of pi (first 20)
    PI_DIGITS: ClassVar[list[int]] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4]

    def __init__(self, num_lines: int = 5, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_lines = min(num_lines, len(self.PI_DIGITS))
        self.expected = self.PI_DIGITS[: self.num_lines]

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counter based on vowel groups."""
        word = word.lower().strip(".,!?;:'\"")
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
        # Handle silent e
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if len(structure.lines) != self.num_lines:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": f"Expected {self.num_lines} lines"},
            )

        matches = 0
        details = []
        for i, line in enumerate(structure.lines):
            words = line.split()
            syllables = sum(self._count_syllables(w) for w in words)
            expected = self.expected[i]

            if syllables == expected:
                matches += 1
                details.append(f"Line {i + 1}: ✓ {syllables} syllables (π[{i + 1}])")
            else:
                details.append(f"Line {i + 1}: ✗ {syllables} syllables (expected {expected})")

        # Steep penalty for GRPO training: 0 errors=1.0, 1 error=0.5, 2 errors=0.25, 3+=0.05
        violations = self.num_lines - matches
        if violations == 0:
            score = 1.0
        elif violations == 1:
            score = 0.5
        elif violations == 2:
            score = 0.25
        else:
            score = 0.05
        return VerificationResult(
            score=score,
            passed=violations == 0,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"expected": self.expected, "line_details": details},
        )

    def describe(self) -> str:
        seq = ", ".join(str(x) for x in self.expected)
        return (
            f"PiKu: Write {self.num_lines} lines where the syllable counts follow "
            f"the digits of π. Pattern: {seq}"
        )


class SquareStanzas(Constraint):
    """
    Stanza N has N² words.

    Stanza 1: 1 word, Stanza 2: 4 words, Stanza 3: 9 words, etc.
    Creates a visual representation of perfect squares.
    """

    name = "SquareStanzas"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, num_stanzas: int = 4, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_stanzas = num_stanzas
        self.expected = [(i + 1) ** 2 for i in range(num_stanzas)]

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if len(structure.stanzas) < self.num_stanzas:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": f"Expected at least {self.num_stanzas} stanzas"},
            )

        matches = 0
        details = []
        for i, stanza in enumerate(structure.stanzas[: self.num_stanzas]):
            words = sum(len(line.split()) for line in stanza)
            expected = self.expected[i]

            if words == expected:
                matches += 1
                details.append(f"Stanza {i + 1}: ✓ {words} words ({i + 1}²)")
            else:
                details.append(f"Stanza {i + 1}: ✗ {words} words (expected {expected} = {i + 1}²)")

        # Steep penalty for GRPO training: 0 errors=1.0, 1 error=0.5, 2 errors=0.25, 3+=0.05
        violations = self.num_stanzas - matches
        if violations == 0:
            score = 1.0
        elif violations == 1:
            score = 0.5
        elif violations == 2:
            score = 0.25
        else:
            score = 0.05
        return VerificationResult(
            score=score,
            passed=violations == 0,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"expected": self.expected, "stanza_details": details},
        )

    def describe(self) -> str:
        seq = ", ".join(str(x) for x in self.expected)
        return (
            f"SquareStanzas: Write {self.num_stanzas} stanzas where stanza N has "
            f"N² words. Pattern: {seq}"
        )


class SelfReferential(Constraint):
    """
    Line N contains exactly N instances of the digit N (or word for N).

    Line 1 must contain exactly one "1" or "one"
    Line 2 must contain exactly two "2"s or "two"s
    Etc.
    """

    name = "SelfReferential"
    constraint_type = ConstraintType.COMPOSITE

    WORDS: ClassVar[dict[int, str]] = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }

    def __init__(self, num_lines: int = 5, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.num_lines = min(num_lines, 9)  # Only 1-9 make sense

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        if len(structure.lines) < self.num_lines:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": f"Expected at least {self.num_lines} lines"},
            )

        matches = 0
        details = []
        for i in range(self.num_lines):
            line = structure.lines[i].lower()
            n = i + 1
            digit_count = line.count(str(n))
            word_count = line.count(self.WORDS[n])
            total = digit_count + word_count

            if total == n:
                matches += 1
                details.append(f"Line {n}: ✓ contains {total}x '{n}'/'{self.WORDS[n]}'")
            else:
                details.append(
                    f"Line {n}: ✗ contains {total}x '{n}'/'{self.WORDS[n]}' (expected {n})"
                )

        # Steep penalty for GRPO training: 0 errors=1.0, 1 error=0.5, 2 errors=0.25, 3+=0.05
        violations = self.num_lines - matches
        if violations == 0:
            score = 1.0
        elif violations == 1:
            score = 0.5
        elif violations == 2:
            score = 0.25
        else:
            score = 0.05
        return VerificationResult(
            score=score,
            passed=violations == 0,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={"line_details": details},
        )

    def describe(self) -> str:
        return (
            f"SelfReferential: Write {self.num_lines} lines where line N contains "
            f"exactly N instances of the number N (digit or word). "
            f"Line 1 has one '1' or 'one', line 2 has two '2's or 'two's, etc."
        )


class ModularVerse(Constraint):
    """
    All word lengths must be congruent to K (mod N).

    E.g., if K=1 and N=3, every word must have length 1, 4, 7, 10... letters.
    Tests understanding of modular arithmetic.
    """

    name = "ModularVerse"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(self, k: int = 1, n: int = 3, min_words: int = 15, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.k = k % n  # Normalize k
        self.n = n
        self.min_words = min_words

    def verify(self, poem: str | PoemStructure) -> VerificationResult:
        structure = self._ensure_structure(poem)

        all_words = []
        for line in structure.lines:
            all_words.extend(line.split())

        if len(all_words) < self.min_words:
            return VerificationResult(
                score=0.0,
                passed=False,
                rubric=[],
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                details={"error": f"Expected at least {self.min_words} words"},
            )

        matches = 0
        failures = []
        for word in all_words:
            # Strip punctuation for length calculation
            clean = "".join(c for c in word if c.isalpha())
            if len(clean) % self.n == self.k:
                matches += 1
            else:
                failures.append(f"{word}({len(clean)})")

        # Steep penalty for GRPO training: 0 errors=1.0, 1 error=0.5, 2 errors=0.25, 3+=0.05
        violations = len(all_words) - matches
        if violations == 0:
            score = 1.0
        elif violations == 1:
            score = 0.5
        elif violations == 2:
            score = 0.25
        else:
            score = 0.05
        passed = violations == 0

        return VerificationResult(
            score=score,
            passed=passed,
            rubric=[],
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            details={
                "matches": matches,
                "total_words": len(all_words),
                "failed_words": failures[:10],  # First 10 failures
            },
        )

    def describe(self) -> str:
        examples = [self.k + self.n * i for i in range(4)]
        return (
            f"ModularVerse: Write at least {self.min_words} words where every word's "
            f"length ≡ {self.k} (mod {self.n}). Valid lengths: {examples}..."
        )
