"""
Sestina form template.

A 39-line poem with complex end-word rotation pattern.
"""

from __future__ import annotations

from abide.constraints import (
    And,
    Constraint,
    ConstraintType,
    EndWordPattern,
    LineCount,
    StanzaCount,
    StanzaSizes,
    VerificationResult,
    WeightedSum,
)
from abide.primitives import PoemStructure


class Sestina(Constraint):
    """
    Sestina: 39 lines with end-word rotation pattern.

    Structure:
    - 6 sestets (6-line stanzas) + 1 tercet (3-line envoi)
    - Same 6 end words rotate through all stanzas
    - Rotation pattern (retrogradatio cruciata): 6,1,5,2,4,3

    The end words from stanza 1 appear in subsequent stanzas
    according to the rotation pattern. The envoi uses all 6
    words, typically 2 per line.

    Examples:
        >>> sestina = Sestina()
        >>> result = sestina.verify(poem)
    """

    name = "Sestina"
    constraint_type = ConstraintType.COMPOSITE

    # Sestina rotation pattern (0-indexed)
    ROTATION = [5, 0, 4, 1, 3, 2]

    def __init__(
        self,
        weight: float = 1.0,
        word_match_threshold: float = 0.8,
        strict: bool = False,
    ) -> None:
        """
        Initialize sestina constraint.

        Args:
            weight: Relative weight for composition
            word_match_threshold: Minimum similarity for end-word matching
            strict: If True, all constraints must pass; if False, uses weighted scoring
        """
        super().__init__(weight)
        self.word_match_threshold = word_match_threshold
        self.strict = strict

        # Structural constraints
        self._line_count = LineCount(39, weight=1.5)
        self._stanza_count = StanzaCount(7, weight=1.0)
        self._stanza_sizes = StanzaSizes(
            [6, 6, 6, 6, 6, 6, 3],  # 6 sestets + envoi
            weight=1.5,
        )

        # End-word rotation pattern (main 6 stanzas)
        self._end_word_pattern = EndWordPattern(
            num_words=6,
            num_stanzas=6,
            rotation=self.ROTATION,
            weight=3.0,
            threshold=word_match_threshold,
        )

        # Compose constraints
        if strict:
            self._constraint = And([
                self._line_count,
                self._stanza_count,
                self._stanza_sizes,
                self._end_word_pattern,
            ])
        else:
            self._constraint = WeightedSum(
                [
                    (self._line_count, 1.5),
                    (self._stanza_count, 1.0),
                    (self._stanza_sizes, 1.5),
                    (self._end_word_pattern, 3.0),
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
        return "Sestina: 39 lines (6 sestets + envoi) with end-word rotation"
