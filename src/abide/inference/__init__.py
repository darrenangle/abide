"""
Form inference - reverse-engineer constraints from poems.

This module analyzes poems and derives FormSpec constraints that
the poem passes with 100% score. Useful for:
- Creating custom forms from example poems
- Testing framework expressiveness
- Understanding a poem's formal structure
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
