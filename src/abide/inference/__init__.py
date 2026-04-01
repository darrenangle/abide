"""
Form inference - reverse-engineer constraints from poems.

This module analyzes poems and derives self-consistent FormSpec constraints
that the source poem itself passes with 100% score. Useful for:
- Creating custom forms from example poems
- Testing framework expressiveness
- Recovering reproducible structural and prosodic patterns from a poem
"""

from abide.inference.analyzer import (
    FormAnalysis,
    InferredConstraint,
    analyze_poem,
    infer_form,
)

__all__ = [
    "FormAnalysis",
    "InferredConstraint",
    "analyze_poem",
    "infer_form",
]
