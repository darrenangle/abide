"""
Ballade form template.

A 28-line poem with envoi and refrain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


class Ballade(Constraint):
    """
    Ballade: 28 lines with 3 octaves + envoi, refrain at stanza ends.

    Structure:
    - 3 stanzas of 8 lines (octaves) + 1 envoi of 4 lines
    - Rhyme scheme per octave: ABABBCBC
    - Envoi rhyme: BCBC
    - Last line of each stanza is refrain (same line)
    - Only 3 rhyme sounds used throughout (A, B, C)
    - Envoi traditionally addressed to "Prince"

    Famous example:
        "Ballade des Dames du Temps Jadis" by François Villon

    Examples:
        >>> ballade = Ballade()
        >>> result = ballade.verify(poem)
    """

    name = "Ballade"
    constraint_type = ConstraintType.COMPOSITE

    LINE_COUNT = 28
    OCTAVE_SCHEME = "ABABBCBC"
    ENVOI_SCHEME = "BCBC"
    RHYME_SCHEME = OCTAVE_SCHEME * 3 + ENVOI_SCHEME

    def __init__(
        self,
        weight: float = 1.0,
        rhyme_threshold: float = 0.6,
        refrain_threshold: float = 0.9,
        strict: bool = False,
    ) -> None:
        """
        Initialize ballade constraint.

        Args:
            weight: Relative weight for composition
            rhyme_threshold: Minimum score for rhymes
            refrain_threshold: Minimum similarity for refrain lines
            strict: If True, all constraints must pass
        """
        super().__init__(weight)
        self.rhyme_threshold = rhyme_threshold
        self.refrain_threshold = refrain_threshold
        self.strict = strict
        self._line_count = LineCount(self.LINE_COUNT, weight=2.0)
        self._stanza_count = StanzaCount(4, weight=1.0)
        self._stanza_sizes = StanzaSizes([8, 8, 8, 4], weight=1.0)
        self._rhyme = RhymeScheme(
            self.RHYME_SCHEME,
            weight=2.0,
            threshold=rhyme_threshold,
            allow_identical=True,
        )
        self._refrain = Refrain(
            reference_line=7,
            repeat_at=[15, 23, 27],
            weight=2.0,
            threshold=refrain_threshold,
        )

        constraints = [
            (self._line_count, 2.0),
            (self._stanza_count, 1.0),
            (self._stanza_sizes, 1.0),
            (self._rhyme, 2.0),
            (self._refrain, 2.0),
        ]

        self._constraint: Constraint
        if strict:
            self._constraint = And([constraint for constraint, _ in constraints])
        else:
            self._constraint = WeightedSum(
                constraints,
                threshold=0.6,
                required_indices=list(range(len(constraints))),
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
        return "Ballade: 28 lines (3 octaves + envoi) with ABABBCBC rhyme and refrain"
