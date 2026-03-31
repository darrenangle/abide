"""
Constraint types and base classes.

This module provides the constraint algebra:
- Base Constraint class with verify/describe/to_rubric methods
- Structural constraints (line count, stanza count, etc.)
- Relational constraints (rhyme scheme, refrains, etc.)
- Prosodic constraints (meter, syllables)
- Sonic constraints (assonance, consonance)
- Composition operators (AND, OR, NOT, WeightedSum)
"""

from abide.constraints.base import Constraint, NumericConstraint
from abide.constraints.lexical import (
    Alliteration,
    AllWordsUnique,
    CharacterCount,
    CharacterPalindrome,
    CrossLineVowelWordCount,
    DoubleAcrostic,
    ExactCharacterBudget,
    ExactTotalCharacters,
    ExactTotalVowels,
    ExactWordCount,
    ForcedWords,
    LetterFrequency,
    LineEndsWith,
    LineStartsWith,
    MonosyllabicOnly,
    NoConsecutiveRepeats,
    NoSharedLetters,
    PositionalCharacter,
    VowelConsonantPattern,
    WordCount,
    WordLengthPattern,
    WordLengthStaircase,
)
from abide.constraints.meter import Meter, MeterPattern
from abide.constraints.operators import And, AtLeast, AtMost, Not, Or, WeightedSum
from abide.constraints.relational import (
    Acrostic,
    EndRhymeDensity,
    EndRhymePairs,
    EndWordPattern,
    LinePairSimilarity,
    OpeningPhraseRefrain,
    Refrain,
    RhymeScheme,
)
from abide.constraints.shape import (
    LineLengthRange,
    LineShape,
    MeasureMode,
    ShapeType,
)
from abide.constraints.structural import (
    GroupedStanzas,
    LineCount,
    StanzaCount,
    StanzaSizes,
    SyllablesPerLine,
    TotalSyllables,
)
from abide.constraints.types import (
    BoundType,
    ConstraintType,
    NumericBound,
    RubricItem,
    VerificationResult,
)

__all__ = [
    "Acrostic",
    "AllWordsUnique",
    "Alliteration",
    "And",
    "AtLeast",
    "AtMost",
    "BoundType",
    "CharacterCount",
    "CharacterPalindrome",
    "Constraint",
    "ConstraintType",
    "CrossLineVowelWordCount",
    "DoubleAcrostic",
    "EndRhymeDensity",
    "EndRhymePairs",
    "EndWordPattern",
    "ExactCharacterBudget",
    "ExactTotalCharacters",
    "ExactTotalVowels",
    "ExactWordCount",
    "ForcedWords",
    "GroupedStanzas",
    "LetterFrequency",
    "LineCount",
    "LineEndsWith",
    "LineLengthRange",
    "LinePairSimilarity",
    "LineShape",
    "LineStartsWith",
    "MeasureMode",
    "Meter",
    "MeterPattern",
    "MonosyllabicOnly",
    "NoConsecutiveRepeats",
    "NoSharedLetters",
    "Not",
    "NumericBound",
    "NumericConstraint",
    "OpeningPhraseRefrain",
    "Or",
    "PositionalCharacter",
    "Refrain",
    "RhymeScheme",
    "RubricItem",
    "ShapeType",
    "StanzaCount",
    "StanzaSizes",
    "SyllablesPerLine",
    "TotalSyllables",
    "VerificationResult",
    "VowelConsonantPattern",
    "WeightedSum",
    "WordCount",
    "WordLengthPattern",
    "WordLengthStaircase",
]
