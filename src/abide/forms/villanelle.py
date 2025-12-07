"""
Villanelle form template.

A 19-line poem with two refrains and a specific rhyme scheme.
"""

from __future__ import annotations

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    LineCount,
    Refrain,
    RhymeScheme,
    StanzaCount,
    StanzaSizes,
    VerificationResult,
    WeightedSum,
)
from abide.primitives import PoemStructure


class Villanelle(Constraint):
    """
    Villanelle: 19 lines in 6 stanzas with refrains.

    Structure:
    - 5 tercets (3-line stanzas) + 1 quatrain (4-line stanza)
    - Line 1 repeats at lines 6, 12, 18
    - Line 3 repeats at lines 9, 15, 19
    - Rhyme scheme: ABA ABA ABA ABA ABA ABAA

    Famous examples:
    - "Do not go gentle into that good night" by Dylan Thomas
    - "One Art" by Elizabeth Bishop

    Examples:
        >>> villanelle = Villanelle()
        >>> result = villanelle.verify(poem)
    """

    name = "Villanelle"
    constraint_type = ConstraintType.COMPOSITE

    # Standard villanelle rhyme scheme (A=1,3 position rhymes, B=2 position)
    RHYME_SCHEME = "ABAABABAABABAABAA"

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        refrain_threshold: float = 0.85,
        strict: bool = False,
    ) -> None:
        """
        Initialize villanelle constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum score for rhymes to count
            refrain_threshold: Minimum similarity for refrains (lower allows variation)
            strict: If True, all constraints must pass; if False, uses weighted scoring
        """
        super().__init__(weight)
        self.rhyme_threshold = rhyme_threshold
        self.refrain_threshold = refrain_threshold
        self.strict = strict

        # Structural constraints
        self._line_count = LineCount(19, weight=2.0)
        self._stanza_count = StanzaCount(6, weight=1.0)
        self._stanza_sizes = StanzaSizes([3, 3, 3, 3, 3, 4], weight=1.5)

        # Relational constraints
        # Refrain A: line 0 (line 1) repeats at lines 5, 11, 17 (lines 6, 12, 18)
        self._refrain_a = Refrain(
            reference_line=0,
            repeat_at=[5, 11, 17],
            weight=2.0,
            threshold=refrain_threshold,
        )

        # Refrain B: line 2 (line 3) repeats at lines 8, 14, 18 (lines 9, 15, 19)
        self._refrain_b = Refrain(
            reference_line=2,
            repeat_at=[8, 14, 18],
            weight=2.0,
            threshold=refrain_threshold,
        )

        # Rhyme scheme
        self._rhyme_scheme = RhymeScheme(
            self.RHYME_SCHEME,
            weight=1.5,
            threshold=rhyme_threshold,
        )

        # Compose constraints
        if strict:
            self._constraint = And([
                self._line_count,
                self._stanza_count,
                self._stanza_sizes,
                self._refrain_a,
                self._refrain_b,
                self._rhyme_scheme,
            ])
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 2.0),
                    (self._stanza_count, 1.0),
                    (self._stanza_sizes, 1.5),
                    (self._refrain_a, 2.0),
                    (self._refrain_b, 2.0),
                    (self._rhyme_scheme, 1.5),
                ],
                threshold=0.6,
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
        return "Villanelle: 19 lines (5 tercets + quatrain) with ABA rhyme and two refrains"
