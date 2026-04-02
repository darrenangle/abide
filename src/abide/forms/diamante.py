"""
Diamante and other shape poem forms.

Diamante is a diamond-shaped poem with 7 lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    LineShape,
    MeasureMode,
    ShapeType,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


def _compose_shape_form(
    line_count: LineCount,
    shape: LineShape,
    *,
    strict: bool,
) -> Constraint:
    """Compose a shape-form verifier with hard pass gating in lenient mode."""
    if strict:
        return And([line_count, shape])
    return WeightedSum(
        [(line_count, 1.0), (shape, 3.0)],
        threshold=0.7,
        required_indices=[0, 1],
    )


class Diamante(Constraint):
    """
    Diamante: 7-line diamond-shaped poem.

    This verifier checks the 1-2-3-4-3-2-1 line-length shape only, not the
    part-of-speech or semantic-role conventions sometimes associated with the
    traditional form.

    Examples:
        >>> diamante = Diamante()
        >>> result = diamante.verify(poem)

        >>> # With syllable measurement instead of words
        >>> diamante = Diamante(measure_mode=MeasureMode.SYLLABLES)
    """

    name = "Diamante"
    constraint_type = ConstraintType.COMPOSITE

    # Classic word counts per line
    WORD_PATTERN: ClassVar[list[int]] = [1, 2, 3, 4, 3, 2, 1]

    def __init__(
        self,
        measure_mode: MeasureMode = MeasureMode.WORDS,
        tolerance: int = 0,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        """
        Initialize diamante constraint.

        Args:
            measure_mode: How to measure line length (WORDS, SYLLABLES, CHARACTERS)
            tolerance: Allow +/- this many units per line
            weight: Relative weight for composition
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.measure_mode = measure_mode
        self.tolerance = tolerance
        self.strict_mode = strict

        self._line_count = LineCount(7, weight=2.0)

        # Use LineShape with diamond pattern
        self._shape = LineShape(
            shape_type=ShapeType.DIAMOND,
            num_lines=7,
            mode=measure_mode,
            tolerance=tolerance,
            relative=False,  # Exact counts
            weight=2.0,
        )

        # Override with specific pattern if using words
        if measure_mode == MeasureMode.WORDS:
            self._shape = LineShape(
                lengths=self.WORD_PATTERN,
                mode=measure_mode,
                tolerance=tolerance,
                relative=False,
                weight=2.0,
            )

        self._constraint = _compose_shape_form(self._line_count, self._shape, strict=strict)

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
        return f"Diamante: 7 lines with 1-2-3-4-3-2-1 {self.measure_mode.value} pattern"


class Cinquain(Constraint):
    """
    Cinquain: 5-line poem with syllable pattern 2-4-6-8-2.

    Invented by Adelaide Crapsey. Related to haiku/tanka.

    Structure:
    - Line 1: 2 syllables
    - Line 2: 4 syllables
    - Line 3: 6 syllables
    - Line 4: 8 syllables
    - Line 5: 2 syllables

    Examples:
        >>> cinquain = Cinquain()
        >>> result = cinquain.verify(poem)
    """

    name = "Cinquain"
    constraint_type = ConstraintType.COMPOSITE

    SYLLABLE_PATTERN: ClassVar[list[int]] = [2, 4, 6, 8, 2]

    def __init__(
        self,
        tolerance: int = 0,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        """
        Initialize cinquain constraint.

        Args:
            tolerance: Allow +/- this many syllables per line
            weight: Relative weight for composition
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.tolerance = tolerance
        self.strict_mode = strict

        self._line_count = LineCount(5, weight=2.0)
        self._shape = LineShape(
            lengths=self.SYLLABLE_PATTERN,
            mode=MeasureMode.SYLLABLES,
            tolerance=tolerance,
            relative=False,
            weight=2.0,
        )

        self._constraint = _compose_shape_form(self._line_count, self._shape, strict=strict)

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
        return "Cinquain: 5 lines with 2-4-6-8-2 syllables"


class WordCinquain(Constraint):
    """
    Word cinquain: 5-line poem with word pattern 1-2-3-4-1.

    Variant of cinquain measured by words instead of syllables.

    This verifier checks only the 1-2-3-4-1 word-count pattern.

    Examples:
        >>> cinquain = WordCinquain()
        >>> result = cinquain.verify(poem)
    """

    name = "Word Cinquain"
    constraint_type = ConstraintType.COMPOSITE

    WORD_PATTERN: ClassVar[list[int]] = [1, 2, 3, 4, 1]

    def __init__(
        self,
        tolerance: int = 0,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        super().__init__(weight)
        self.tolerance = tolerance
        self.strict_mode = strict

        self._line_count = LineCount(5, weight=2.0)
        self._shape = LineShape(
            lengths=self.WORD_PATTERN,
            mode=MeasureMode.WORDS,
            tolerance=tolerance,
            relative=False,
            weight=2.0,
        )

        self._constraint = _compose_shape_form(self._line_count, self._shape, strict=strict)

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
        return "Word Cinquain: 5 lines with 1-2-3-4-1 words"


class Etheree(Constraint):
    """
    Etheree: 10-line poem with syllable pattern 1-2-3-4-5-6-7-8-9-10.

    Named after Arkansas poet Etheree Taylor Armstrong.
    Each line adds one syllable, creating a triangular shape.

    Examples:
        >>> etheree = Etheree()
        >>> result = etheree.verify(poem)
    """

    name = "Etheree"
    constraint_type = ConstraintType.COMPOSITE

    SYLLABLE_PATTERN: ClassVar[list[int]] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __init__(
        self,
        tolerance: int = 0,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        super().__init__(weight)
        self.tolerance = tolerance
        self.strict_mode = strict

        self._line_count = LineCount(10, weight=2.0)
        self._shape = LineShape(
            shape_type=ShapeType.TRIANGLE_UP,
            num_lines=10,
            mode=MeasureMode.SYLLABLES,
            tolerance=tolerance,
            relative=False,
            weight=2.0,
        )

        # Use exact syllable counts
        self._shape = LineShape(
            lengths=self.SYLLABLE_PATTERN,
            mode=MeasureMode.SYLLABLES,
            tolerance=tolerance,
            relative=False,
            weight=2.0,
        )

        self._constraint = _compose_shape_form(self._line_count, self._shape, strict=strict)

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
        return "Etheree: 10 lines with 1-2-3-4-5-6-7-8-9-10 syllables"


class ReverseEtheree(Constraint):
    """
    Reverse Etheree: 10-line poem with syllable pattern 10-9-8-7-6-5-4-3-2-1.

    Inverted etheree - each line removes one syllable.

    Examples:
        >>> etheree = ReverseEtheree()
        >>> result = etheree.verify(poem)
    """

    name = "Reverse Etheree"
    constraint_type = ConstraintType.COMPOSITE

    SYLLABLE_PATTERN: ClassVar[list[int]] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    def __init__(
        self,
        tolerance: int = 0,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        super().__init__(weight)
        self.tolerance = tolerance
        self.strict_mode = strict

        self._line_count = LineCount(10, weight=2.0)
        self._shape = LineShape(
            lengths=self.SYLLABLE_PATTERN,
            mode=MeasureMode.SYLLABLES,
            tolerance=tolerance,
            relative=False,
            weight=2.0,
        )

        self._constraint = _compose_shape_form(self._line_count, self._shape, strict=strict)

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
        return "Reverse Etheree: 10 lines with 10-9-8-7-6-5-4-3-2-1 syllables"
