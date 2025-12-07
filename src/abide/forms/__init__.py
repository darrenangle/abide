"""
Pre-composed poetic form templates.

Each form is built entirely from primitive constraints, demonstrating
the composability of the constraint algebra.

Available forms:
- Haiku: 5-7-5 syllable structure
- Tanka: 5-7-5-7-7 syllable structure
- Limerick: 5 lines with AABBA rhyme
- Sonnet: Shakespearean, Petrarchan, Spenserian variants
- Villanelle: 19 lines with refrains and ABA rhyme
- Sestina: 39 lines with end-word rotation
- Triolet: 8 lines with ABaAabAB rhyme and refrains
- Pantoum: Quatrains with interlocking line repetition
- Terza Rima: Tercets with ABA BCB chain rhyme
- Ghazal: Couplets with radif and qafiya
- Rondeau: 15 lines with rentrement refrain
- Ballade: 28 lines with 3 octaves + envoi
- Blues Poem: AAB tercets with line repetition
- Clerihew: 4-line humorous biographical poem
"""

from abide.forms.ballade import Ballade
from abide.forms.blues import BluesPoem
from abide.forms.clerihew import Clerihew
from abide.forms.ghazal import Ghazal
from abide.forms.haiku import Haiku, Tanka
from abide.forms.limerick import Limerick
from abide.forms.pantoum import Pantoum
from abide.forms.rondeau import Rondeau
from abide.forms.sestina import Sestina
from abide.forms.sonnet import (
    PetrarchanSonnet,
    ShakespeareanSonnet,
    Sonnet,
    SpenserianSonnet,
)
from abide.forms.terza_rima import TerzaRima
from abide.forms.triolet import Triolet
from abide.forms.villanelle import Villanelle

__all__ = [
    "Ballade",
    "BluesPoem",
    "Clerihew",
    "Ghazal",
    "Haiku",
    "Limerick",
    "Pantoum",
    "PetrarchanSonnet",
    "Rondeau",
    "Sestina",
    "ShakespeareanSonnet",
    "Sonnet",
    "SpenserianSonnet",
    "Tanka",
    "TerzaRima",
    "Triolet",
    "Villanelle",
]
