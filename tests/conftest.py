"""
Pytest configuration and shared fixtures for abide tests.

Testing Philosophy:
===================
Our primary test suite consists of REAL POEMS that serve as ground truth.
Each poem is paired with its expected constraint specification (the "abide spec").
Tests verify that:
1. Our constraints correctly PASS for poems that follow the form
2. Our constraints correctly FAIL (with appropriate scores) for poems that don't
3. The rubric output accurately explains what matched/didn't match

This approach ensures we're testing against reality, not synthetic data.
"""

from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import settings
from hypothesis import strategies as st

# ============================================================================
# Hypothesis Configuration
# ============================================================================

settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=None)
settings.register_profile("debug", max_examples=5, deadline=None, verbosity=2)


# ============================================================================
# Ground Truth Poem Specifications
# ============================================================================


@dataclass
class PoemSpec:
    """A poem paired with its expected constraint specification.

    This is the core of our e2e testing: real poems with their known forms,
    used to verify that abide correctly identifies and scores adherence.
    """

    name: str
    author: str
    form: str  # e.g., "haiku", "villanelle", "shakespearean_sonnet"
    text: str

    # Expected structural properties (verified by tests)
    expected_line_count: int
    expected_stanza_count: int
    expected_stanza_sizes: list[int]

    # Expected pattern properties (form-specific)
    expected_syllables_per_line: list[int] | None = None
    expected_rhyme_scheme: str | None = None
    expected_end_words: list[str] | None = None  # For sestinas
    expected_refrains: dict[str, list[int]] | None = None  # line_text -> positions

    # Expected scores (what our system should produce)
    expected_min_score: float = 0.9  # Should score at least this high

    # Notes about the poem for test documentation
    notes: str = ""


# ============================================================================
# HAIKU - Ground Truth Examples
# ============================================================================

BASHO_FROG = PoemSpec(
    name="The Old Pond",
    author="Matsuo Bashō",
    form="haiku",
    text="""An old silent pond
A frog jumps into the pond
Splash! Silence again""",
    expected_line_count=3,
    expected_stanza_count=1,
    expected_stanza_sizes=[3],
    expected_syllables_per_line=[5, 7, 5],
    notes="Classic haiku. English translation maintains 5-7-5 structure.",
)

BASHO_AUTUMN = PoemSpec(
    name="Autumn Moonlight",
    author="Matsuo Bashō",
    form="haiku",
    text="""Autumn moonlight—
a worm digs silently
into the chestnut""",
    expected_line_count=3,
    expected_stanza_count=1,
    expected_stanza_sizes=[3],
    expected_syllables_per_line=[4, 6, 5],  # Looser translation
    expected_min_score=0.7,  # Won't be perfect 5-7-5
    notes="Translation doesn't preserve strict 5-7-5.",
)


# ============================================================================
# VILLANELLE - Ground Truth Examples
# ============================================================================

THOMAS_GENTLE_NIGHT = PoemSpec(
    name="Do not go gentle into that good night",
    author="Dylan Thomas",
    form="villanelle",
    text="""Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light.

Though wise men at their end know dark is right,
Because their words had forked no lightning they
Do not go gentle into that good night.

Good men, the last wave by, crying how bright
Their frail deeds might have danced in a green bay,
Rage, rage against the dying of the light.

Wild men who caught and sang the sun in flight,
And learn, too late, they grieved it on its way,
Do not go gentle into that good night.

Grave men, near death, who see with blinding sight
Blind eyes could blaze like meteors and be gay,
Rage, rage against the dying of the light.

And you, my father, there on the sad height,
Curse, bless, me now with your fierce tears, I pray.
Do not go gentle into that good night.
Rage, rage against the dying of the light.""",
    expected_line_count=19,
    expected_stanza_count=6,
    expected_stanza_sizes=[3, 3, 3, 3, 3, 4],
    expected_rhyme_scheme="ABA ABA ABA ABA ABA ABAA",
    expected_refrains={
        "Do not go gentle into that good night.": [0, 5, 11, 17],
        "Rage, rage against the dying of the light.": [2, 8, 14, 18],
    },
    notes="The definitive English villanelle. Perfect form adherence.",
)

BISHOP_ONE_ART = PoemSpec(
    name="One Art",
    author="Elizabeth Bishop",
    form="villanelle",
    text="""The art of losing isn't hard to master;
so many things seem filled with the intent
to be lost that their loss is no disaster.

Lose something every day. Accept the fluster
of lost door keys, the hour badly spent.
The art of losing isn't hard to master.

Then practice losing farther, losing faster:
places, and names, and where it was you meant
to travel. None of these will bring disaster.

I lost my mother's watch. And look! my last, or
next-to-last, of three loved houses went.
The art of losing isn't hard to master.

I lost two cities, lovely ones. And, vaster,
some realms I owned, two rivers, a continent.
I miss them, but it wasn't a disaster.

—Even losing you (the joking voice, a gesture
I love) I shan't have lied. It's evident
the art of losing's not too hard to master
though it may look like (Write it!) like disaster.""",
    expected_line_count=19,
    expected_stanza_count=6,
    expected_stanza_sizes=[3, 3, 3, 3, 3, 4],
    expected_rhyme_scheme="ABA ABA ABA ABA ABA ABAA",
    expected_refrains={
        "The art of losing isn't hard to master": [0, 5, 11, 17],  # Varies slightly
    },
    expected_min_score=0.85,  # Slight variations in refrains
    notes="Famous villanelle with slight refrain variations ('master' vs 'master;').",
)


# ============================================================================
# SHAKESPEAREAN SONNET - Ground Truth Examples
# ============================================================================

SHAKESPEARE_18 = PoemSpec(
    name="Sonnet 18",
    author="William Shakespeare",
    form="shakespearean_sonnet",
    text="""Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance, or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe, or eyes can see,
So long lives this, and this gives life to thee.""",
    expected_line_count=14,
    expected_stanza_count=1,  # Traditionally no stanza breaks
    expected_stanza_sizes=[14],
    expected_rhyme_scheme="ABABCDCDEFEFGG",
    expected_syllables_per_line=[10] * 14,  # Iambic pentameter
    notes="The most famous English sonnet. Perfect Shakespearean form.",
)

SHAKESPEARE_130 = PoemSpec(
    name="Sonnet 130",
    author="William Shakespeare",
    form="shakespearean_sonnet",
    text="""My mistress' eyes are nothing like the sun;
Coral is far more red than her lips' red;
If snow be white, why then her breasts are dun;
If hairs be wires, black wires grow on her head.
I have seen roses damask'd, red and white,
But no such roses see I in her cheeks;
And in some perfumes is there more delight
Than in the breath that from my mistress reeks.
I love to hear her speak, yet well I know
That music hath a far more pleasing sound;
I grant I never saw a goddess go;
My mistress, when she walks, treads on the ground:
And yet, by heaven, I think my love as rare
As any she belied with false compare.""",
    expected_line_count=14,
    expected_stanza_count=1,
    expected_stanza_sizes=[14],
    expected_rhyme_scheme="ABABCDCDEFEFGG",
    expected_syllables_per_line=[10] * 14,
    notes="Anti-Petrarchan sonnet. Perfect Shakespearean form.",
)


# ============================================================================
# PETRARCHAN SONNET - Ground Truth Examples
# ============================================================================

WORDSWORTH_WORLD = PoemSpec(
    name="The World Is Too Much with Us",
    author="William Wordsworth",
    form="petrarchan_sonnet",
    text="""The world is too much with us; late and soon,
Getting and spending, we lay waste our powers;—
Little we see in Nature that is ours;
We have given our hearts away, a sordid boon!
This Sea that bares her bosom to the moon;
The winds that will be howling at all hours,
And are up-gathered now like sleeping flowers;
For this, for everything, we are out of tune;
It moves us not. Great God! I'd rather be
A Pagan suckled in a creed outworn;
So might I, standing on this pleasant lea,
Have glimpses that would make me less forlorn;
Have sight of Proteus rising from the sea;
Or hear old Triton blow his wreathèd horn.""",
    expected_line_count=14,
    expected_stanza_count=1,
    expected_stanza_sizes=[14],
    expected_rhyme_scheme="ABBAABBACDCDCD",
    expected_syllables_per_line=[10] * 14,
    notes="Petrarchan/Italian sonnet form with ABBAABBA octave.",
)


# ============================================================================
# LIMERICK - Ground Truth Examples
# ============================================================================

LEAR_OLD_MAN_BEARD = PoemSpec(
    name="There was an Old Man with a beard",
    author="Edward Lear",
    form="limerick",
    text="""There was an Old Man with a beard,
Who said, 'It is just as I feared!
Two Owls and a Hen,
Four Larks and a Wren,
Have all built their nests in my beard!'""",
    expected_line_count=5,
    expected_stanza_count=1,
    expected_stanza_sizes=[5],
    expected_rhyme_scheme="AABBA",
    notes="Classic limerick form with AABBA rhyme scheme.",
)


# ============================================================================
# NEGATIVE EXAMPLES (Should NOT match forms)
# ============================================================================

FREE_VERSE_WILLIAMS = PoemSpec(
    name="This Is Just to Say",
    author="William Carlos Williams",
    form="free_verse",
    text="""I have eaten
the plums
that were in
the icebox

and which
you were probably
saving
for breakfast

Forgive me
they were delicious
so sweet
and so cold""",
    expected_line_count=12,
    expected_stanza_count=3,
    expected_stanza_sizes=[4, 4, 4],
    expected_rhyme_scheme=None,  # No rhyme scheme
    expected_min_score=0.0,  # Should NOT match formal verse
    notes="Free verse - should fail haiku, sonnet, villanelle checks.",
)


# ============================================================================
# Poem Collections by Form
# ============================================================================

HAIKU_POEMS = [BASHO_FROG, BASHO_AUTUMN]
VILLANELLE_POEMS = [THOMAS_GENTLE_NIGHT, BISHOP_ONE_ART]
SHAKESPEAREAN_SONNETS = [SHAKESPEARE_18, SHAKESPEARE_130]
PETRARCHAN_SONNETS = [WORDSWORTH_WORLD]
LIMERICKS = [LEAR_OLD_MAN_BEARD]
FREE_VERSE_POEMS = [FREE_VERSE_WILLIAMS]

ALL_POEMS = (
    HAIKU_POEMS
    + VILLANELLE_POEMS
    + SHAKESPEAREAN_SONNETS
    + PETRARCHAN_SONNETS
    + LIMERICKS
    + FREE_VERSE_POEMS
)


# ============================================================================
# Hypothesis Strategies for Fuzz Testing
# ============================================================================


@st.composite
def words(draw: st.DrawFn, min_length: int = 1, max_length: int = 12) -> str:
    """Generate random words (lowercase letters only)."""
    import string

    length = draw(st.integers(min_value=min_length, max_value=max_length))
    return draw(st.text(alphabet=string.ascii_lowercase, min_size=length, max_size=length))


@st.composite
def whitespace_variations(draw: st.DrawFn, text: str) -> str:
    """Add random whitespace variations to text for fuzz testing."""
    result = text

    # Randomly convert some newlines to CRLF
    if draw(st.booleans()):
        result = result.replace("\n", "\r\n")

    # Randomly add trailing whitespace
    if draw(st.booleans()):
        result_lines = result.split("\n")
        for i in range(len(result_lines)):
            if draw(st.booleans()):
                spaces = draw(st.integers(min_value=1, max_value=5))
                result_lines[i] += " " * spaces
        result = "\n".join(result_lines)

    return result


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(params=HAIKU_POEMS, ids=lambda p: p.name)
def haiku_poem(request: pytest.FixtureRequest) -> PoemSpec:
    """Parametrized fixture providing haiku ground truth."""
    return request.param


@pytest.fixture(params=VILLANELLE_POEMS, ids=lambda p: p.name)
def villanelle_poem(request: pytest.FixtureRequest) -> PoemSpec:
    """Parametrized fixture providing villanelle ground truth."""
    return request.param


@pytest.fixture(params=SHAKESPEAREAN_SONNETS, ids=lambda p: p.name)
def shakespearean_sonnet(request: pytest.FixtureRequest) -> PoemSpec:
    """Parametrized fixture providing Shakespearean sonnet ground truth."""
    return request.param


@pytest.fixture(params=ALL_POEMS, ids=lambda p: f"{p.author}:{p.name}")
def any_poem(request: pytest.FixtureRequest) -> PoemSpec:
    """Parametrized fixture providing any poem ground truth."""
    return request.param


@pytest.fixture
def thomas_villanelle() -> PoemSpec:
    """Dylan Thomas's villanelle for detailed testing."""
    return THOMAS_GENTLE_NIGHT


@pytest.fixture
def shakespeare_18() -> PoemSpec:
    """Shakespeare's Sonnet 18 for detailed testing."""
    return SHAKESPEARE_18


@pytest.fixture
def basho_frog() -> PoemSpec:
    """Bashō's frog haiku for detailed testing."""
    return BASHO_FROG


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests requiring external resources")
    config.addinivalue_line("markers", "e2e: end-to-end tests using real poem ground truth")
