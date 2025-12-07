"""
Pre-composed poetic form templates.

Each form is built entirely from primitive constraints, demonstrating
the composability of the constraint algebra.

Available forms:
- Sestina: 39 lines with end-word rotation
- Villanelle: 19 lines with refrains and ABA rhyme
- Sonnet: Shakespearean, Petrarchan, Spenserian variants
- Haiku: 5-7-5 syllable structure
- Tanka: 5-7-5-7-7 syllable structure
- Limerick: 5 lines with AABBA rhyme
"""

from abide.forms.haiku import Haiku, Tanka
from abide.forms.limerick import Limerick
from abide.forms.sestina import Sestina
from abide.forms.sonnet import (
    PetrarchanSonnet,
    ShakespeareanSonnet,
    Sonnet,
    SpenserianSonnet,
)
from abide.forms.villanelle import Villanelle

__all__ = [
    "Haiku",
    "Limerick",
    "PetrarchanSonnet",
    "Sestina",
    "ShakespeareanSonnet",
    "Sonnet",
    "SpenserianSonnet",
    "Tanka",
    "Villanelle",
]
