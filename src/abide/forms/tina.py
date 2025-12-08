"""
Sestina-related forms: Tritina, Quatina, Quintina, Terzanelle.

These are smaller versions of the sestina, using end-word rotation.
Note: Full end-word rotation checking requires manual verification.
These constraints check structural requirements only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    RhymeScheme,
    StanzaCount,
    StanzaSizes,
    VerificationResult,
    WeightedSum,
)

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Tritina(Constraint):
    """
    Tritina: Simplified sestina with 3 end-words rotating across 3 tercets.

    Structure:
    - 3 stanzas of 3 lines each = 9 lines
    - Plus 1-line envoi (tornada) containing all 3 words = 10 lines total
    - End-word rotation: ABC -> CAB -> BCA -> (envoi with A, B, C)

    The rotation pattern is the same as sestina (reversed insertion).

    Examples:
        >>> tritina = Tritina()
        >>> result = tritina.verify(poem)
    """

    name = "Tritina"
    constraint_type = ConstraintType.COMPOSITE

    # Rotation pattern for 3 words (like sestina's pattern for 6)
    # ABC -> CAB -> BCA
    ROTATION_PATTERN: ClassVar[list[list[int]]] = [
        [0, 1, 2],  # Stanza 1: ABC
        [2, 0, 1],  # Stanza 2: CAB
        [1, 2, 0],  # Stanza 3: BCA
    ]

    def __init__(
        self,
        include_envoi: bool = True,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        """
        Initialize tritina constraint.

        Args:
            include_envoi: If True, expects 10 lines (9 + envoi)
            weight: Relative weight for composition
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.include_envoi = include_envoi
        self.strict_mode = strict

        total_lines = 10 if include_envoi else 9
        stanza_sizes = [3, 3, 3, 1] if include_envoi else [3, 3, 3]
        stanza_count = 4 if include_envoi else 3

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.0)
        self._stanza_sizes = StanzaSizes(stanza_sizes, weight=1.0)

        # Note: End-word rotation is a semantic constraint that requires
        # checking actual word content, not just structure.
        # For now, we verify structural requirements only.

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.0),
            (self._stanza_sizes, 1.0),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.6)

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
        lines = "10 (9 + envoi)" if self.include_envoi else "9"
        return f"Tritina: {lines} lines with 3 rotating end-words"


class Quatina(Constraint):
    """
    Quatina: 4-word variant of sestina.

    Structure:
    - 4 stanzas of 4 lines = 16 lines
    - Plus 2-line envoi = 18 lines total
    - End-word rotation: ABCD -> DABC -> CDAB -> BCDA

    Examples:
        >>> quatina = Quatina()
        >>> result = quatina.verify(poem)
    """

    name = "Quatina"
    constraint_type = ConstraintType.COMPOSITE

    # Rotation pattern for 4 words
    ROTATION_PATTERN: ClassVar[list[list[int]]] = [
        [0, 1, 2, 3],  # Stanza 1: ABCD
        [3, 0, 1, 2],  # Stanza 2: DABC
        [2, 3, 0, 1],  # Stanza 3: CDAB
        [1, 2, 3, 0],  # Stanza 4: BCDA
    ]

    def __init__(
        self,
        include_envoi: bool = True,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        """
        Initialize quatina constraint.

        Args:
            include_envoi: If True, expects 18 lines (16 + 2-line envoi)
            weight: Relative weight for composition
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.include_envoi = include_envoi
        self.strict_mode = strict

        total_lines = 18 if include_envoi else 16
        stanza_sizes = [4, 4, 4, 4, 2] if include_envoi else [4, 4, 4, 4]
        stanza_count = 5 if include_envoi else 4

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.0)
        self._stanza_sizes = StanzaSizes(stanza_sizes, weight=1.0)

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.0),
            (self._stanza_sizes, 1.0),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.6)

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
        lines = "18 (16 + envoi)" if self.include_envoi else "16"
        return f"Quatina: {lines} lines with 4 rotating end-words"


class Quintina(Constraint):
    """
    Quintina: 5-word variant of sestina.

    Structure:
    - 5 stanzas of 5 lines = 25 lines
    - Plus coda (variable length)
    - End-word rotation following sestina's pattern

    Examples:
        >>> quintina = Quintina()
        >>> result = quintina.verify(poem)
    """

    name = "Quintina"
    constraint_type = ConstraintType.COMPOSITE

    # Rotation pattern for 5 words (sestina-style reversed insertion)
    # ABCDE -> EABCD -> DEABC -> CDABC -> BCDEA
    ROTATION_PATTERN: ClassVar[list[list[int]]] = [
        [0, 1, 2, 3, 4],  # Stanza 1: ABCDE
        [4, 0, 1, 2, 3],  # Stanza 2: EABCD
        [3, 4, 0, 1, 2],  # Stanza 3: DEABC
        [2, 3, 4, 0, 1],  # Stanza 4: CDEAB
        [1, 2, 3, 4, 0],  # Stanza 5: BCDEA
    ]

    def __init__(
        self,
        include_coda: bool = True,
        coda_lines: int = 3,
        weight: float = 1.0,
        strict: bool = False,
    ) -> None:
        """
        Initialize quintina constraint.

        Args:
            include_coda: If True, includes coda at end
            coda_lines: Number of lines in coda
            weight: Relative weight for composition
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.include_coda = include_coda
        self.coda_lines = coda_lines
        self.strict_mode = strict

        total_lines = 25 + coda_lines if include_coda else 25
        stanza_sizes = [5, 5, 5, 5, 5] + ([coda_lines] if include_coda else [])
        stanza_count = 6 if include_coda else 5

        self._line_count = LineCount(total_lines, weight=2.0)
        self._stanza_count = StanzaCount(stanza_count, weight=1.0)
        self._stanza_sizes = StanzaSizes(stanza_sizes, weight=1.0)

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.0),
            (self._stanza_sizes, 1.0),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.6)

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
        lines = f"{25 + self.coda_lines} (25 + coda)" if self.include_coda else "25"
        return f"Quintina: {lines} lines with 5 rotating end-words"


class Terzanelle(Constraint):
    """
    Terzanelle: Hybrid of villanelle and terza rima.

    Structure:
    - Tercets with interlocking rhyme like terza rima (ABA BCB CDC...)
    - But with villanelle-like refrains
    - Usually 19 lines like a villanelle
    - First and third lines of first stanza return as refrains

    Examples:
        >>> terzanelle = Terzanelle()
        >>> result = terzanelle.verify(poem)
    """

    name = "Terzanelle"
    constraint_type = ConstraintType.COMPOSITE

    def __init__(
        self,
        stanza_count: int = 6,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        strict: bool = False,
    ) -> None:
        """
        Initialize terzanelle constraint.

        Args:
            stanza_count: Number of tercets (typically 6)
            weight: Relative weight for composition
            rhyme_threshold: Minimum rhyme score
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.stanza_count_val = stanza_count
        self.rhyme_threshold = rhyme_threshold
        self.strict_mode = strict

        # 5 tercets + 1 quatrain = 19 lines (like villanelle)
        # Or just tercets: stanza_count * 3
        total_lines = 19  # Standard villanelle-like structure

        self._line_count = LineCount(total_lines, weight=2.0)

        # Terza rima pattern: ABA BCB CDC DED EFE FF (for villanelle-length)

        self._rhyme = RhymeScheme(
            "ABA BCB CDC DED EFE FF",
            weight=2.0,
            threshold=rhyme_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._rhyme, 2.0),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([c for c, _ in constraints])
        else:
            self._constraint = WeightedSum(constraints, threshold=0.6)

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
        return "Terzanelle: 19 lines with terza rima rhyme scheme"
