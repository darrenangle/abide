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
- Quatrain: 4-line stanza with various rhyme schemes
- Couplet: 2 rhyming lines
- Blank Verse: Unrhymed iambic pentameter
- Ode: Pindaric, Horatian, Irregular variants
- Ballad: Narrative quatrains with ABCB rhyme
- Diamante: 7-line diamond shape poem
- Cinquain: 5-line syllabic poem
- Ottava Rima: 8-line stanza with ABABABCC rhyme
- Rhyme Royal: 7-line stanza with ABABBCC rhyme
- Kyrielle: Quatrains with refrain
- Epigram: Short witty poem
- Rubaiyat: Persian quatrains with AABA rhyme
- Free Verse: Poetry without fixed meter
"""

from abide.forms.ballade import Ballade
from abide.forms.ballad import Ballad, BroadBallad, LiteraryBallad
from abide.forms.blank_verse import BlankVerse, DramaticVerse
from abide.forms.blues import BluesPoem
from abide.forms.clerihew import Clerihew
from abide.forms.couplet import Couplet, Elegiac, HeroicCouplet, ShortCouplet
from abide.forms.diamante import (
    Cinquain,
    Diamante,
    Etheree,
    ReverseEtheree,
    WordCinquain,
)
from abide.forms.epigram import Distich, Epigram, Monostich, Tercet, Triplet
from abide.forms.free_verse import FreeVerse, ProsePoem
from abide.forms.ghazal import Ghazal
from abide.forms.haiku import Haiku, Tanka
from abide.forms.kyrielle import Kyrielle, KyrielleSonnet
from abide.forms.limerick import Limerick
from abide.forms.ode import HoratianOde, IrregularOde, Ode, PindaricOde
from abide.forms.ottava_rima import OttavaRima, RhymeRoyal, SpenserianStanza
from abide.forms.pantoum import Pantoum
from abide.forms.quatrain import (
    BalladStanza,
    EnvelopeQuatrain,
    HeroicQuatrain,
    Quatrain,
)
from abide.forms.rondeau import Rondeau
from abide.forms.rubaiyat import Rubai, Rubaiyat
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
    "Ballad",
    "BalladStanza",
    "BlankVerse",
    "BluesPoem",
    "BroadBallad",
    "Cinquain",
    "Clerihew",
    "Couplet",
    "Diamante",
    "Distich",
    "DramaticVerse",
    "Elegiac",
    "EnvelopeQuatrain",
    "Epigram",
    "Etheree",
    "FreeVerse",
    "Ghazal",
    "Haiku",
    "HeroicCouplet",
    "HeroicQuatrain",
    "HoratianOde",
    "IrregularOde",
    "Kyrielle",
    "KyrielleSonnet",
    "Limerick",
    "LiteraryBallad",
    "Monostich",
    "Ode",
    "OttavaRima",
    "Pantoum",
    "PetrarchanSonnet",
    "PindaricOde",
    "ProsePoem",
    "Quatrain",
    "ReverseEtheree",
    "RhymeRoyal",
    "Rondeau",
    "Rubai",
    "Rubaiyat",
    "Sestina",
    "ShakespeareanSonnet",
    "ShortCouplet",
    "Sonnet",
    "SpenserianSonnet",
    "SpenserianStanza",
    "Tanka",
    "Tercet",
    "TerzaRima",
    "Triolet",
    "Triplet",
    "Villanelle",
    "WordCinquain",
]
