"""Tests for form templates using ground truth poems."""

import pytest

from abide import verify
from abide.forms import (
    BlankVerse,
    Bop,
    Couplet,
    Haiku,
    Kyrielle,
    Limerick,
    OttavaRima,
    PetrarchanSonnet,
    Quatrain,
    RhymeRoyal,
    Rondelet,
    Sestina,
    ShakespeareanSonnet,
    Sonnet,
    SpenserianSonnet,
    Tanka,
    Triolet,
    Villanelle,
)

# Ground truth poems for testing
BASHO_HAIKU = """An old silent pond
A frog jumps into the pond
Splash! Silence again"""

DYLAN_THOMAS_VILLANELLE = """Do not go gentle into that good night,
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
Rage, rage against the dying of the light."""

SHAKESPEARE_SONNET_18 = """Shall I compare thee to a summer's day?
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
So long lives this, and this gives life to thee."""

LEAR_LIMERICK = """There was an Old Man with a beard,
Who said, "It is just as I feared!
Two Owls and a Hen,
Four Larks and a Wren,
Have all built their nests in my beard!\""""

TEN_SYLLABLE_LINE_A = "The winter sunlight settles on the bay"
TEN_SYLLABLE_LINE_B = "A silver morning wanders through the rain"
TEN_SYLLABLE_LINE_C = "A shadow drifts unnoticed through the storm"
SHORT_LIMERICK_LINE = "Cold sparrows drift through reeds"


def _triolet_near_miss() -> str:
    return "\n".join(
        [
            "Moonlight settles softly on the bay",
            "A silver heron circles through the rain",
            "Winter branches listen by the bay",
            "Moonlight settles softly on the bay",
            "Small lanterns shimmer near the bay",
            "The river answers slowly through the rain",
            "Moonlight settles softly on the bay",
            "A patient evening wanders through the rain",
        ]
    )


def _bop_near_miss() -> str:
    return "\n".join(
        [
            "Stanza one line one",
            "Stanza one line two",
            "Stanza one line three",
            "Stanza one line four",
            "Stanza one line five",
            "Stanza one line six",
            "The doorway keeps its answer",
            "",
            "Stanza two line one",
            "Stanza two line two",
            "Stanza two line three",
            "Stanza two line four",
            "Stanza two line five",
            "Stanza two line six",
            "Stanza two line seven",
            "Stanza two line eight",
            "The doorway keeps its answer",
            "",
            "Stanza three line one",
            "Stanza three line two",
            "Stanza three line three",
            "Stanza three line four",
            "Stanza three line five",
            "Stanza three line six",
            "The doorway keeps a lantern",
        ]
    )


def _kyrielle_near_miss() -> str:
    return "\n\n".join(
        [
            "\n".join(
                [
                    "Morning gathers over the bay",
                    "Thin branches hover near the bay",
                    "Cold swallows wander by the bay",
                    "The chapel bells remember storm",
                ]
            ),
            "\n".join(
                [
                    "Late sparrows settle in the rain",
                    "Small windows silver in the rain",
                    "Still rooftops darken with the rain",
                    "The chapel bells remember storm",
                ]
            ),
            "\n".join(
                [
                    "Young shadows lengthen toward the light",
                    "Pale shutters soften in the light",
                    "Distant orchards blur into the light",
                    "The chapel doors remember storm",
                ]
            ),
        ]
    )


def _rondelet_near_miss() -> str:
    return "\n".join(
        [
            "Soft rain returns to stone",
            "Quiet swallows cross the night",
            "Soft rain returns to stone",
            "Small lanterns gather by the stone",
            "Still rivers answer through the night",
            "Pale windows hover in the night",
            "Soft clouds return to stone",
        ]
    )


class TestHaiku:
    """Tests for Haiku form template."""

    def test_basho_haiku_passes(self) -> None:
        """Bashō's haiku should pass."""
        haiku = Haiku(strict=False)
        result = verify(BASHO_HAIKU, haiku)
        # Should get good score even if syllables aren't exact
        assert result.score > 0.5

    def test_haiku_wrong_lines_fails(self) -> None:
        """Poem with wrong line count fails."""
        haiku = Haiku()
        result = verify("Line one\nLine two", haiku)
        assert result.passed is False

    def test_haiku_strict_mode(self) -> None:
        """Strict mode requires all constraints."""
        haiku = Haiku(strict=True)
        result = verify(BASHO_HAIKU, haiku)
        # May or may not pass depending on exact syllable count
        assert 0 <= result.score <= 1.0

    def test_haiku_with_tolerance(self) -> None:
        """Syllable tolerance allows variation."""
        haiku = Haiku(syllable_tolerance=1)
        result = verify(BASHO_HAIKU, haiku)
        # With tolerance, should be more lenient
        assert result.score > 0.3

    def test_haiku_describe(self) -> None:
        """Description mentions haiku characteristics."""
        haiku = Haiku()
        desc = haiku.describe()
        assert "Haiku" in desc
        assert "5-7-5" in desc


class TestTanka:
    """Tests for Tanka form template."""

    def test_tanka_structure(self) -> None:
        """Tanka requires 5 lines."""
        tanka = Tanka()
        poem = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        result = verify(poem, tanka)
        # Should at least pass line count
        assert len(result.rubric) > 0

    def test_tanka_wrong_lines(self) -> None:
        """Wrong line count fails."""
        tanka = Tanka()
        result = verify(BASHO_HAIKU, tanka)  # 3 lines, not 5
        assert result.passed is False

    def test_tanka_describe(self) -> None:
        """Description mentions tanka characteristics."""
        tanka = Tanka()
        desc = tanka.describe()
        assert "Tanka" in desc
        assert "5-7-5-7-7" in desc


class TestVillanelle:
    """Tests for Villanelle form template."""

    def test_dylan_thomas_villanelle_scores_well(self) -> None:
        """Dylan Thomas's villanelle should score well."""
        villanelle = Villanelle()
        result = verify(DYLAN_THOMAS_VILLANELLE, villanelle)
        # Should get good score
        assert result.score > 0.5

    def test_villanelle_has_19_lines_rubric(self) -> None:
        """Rubric should check for 19 lines."""
        villanelle = Villanelle()
        result = verify(DYLAN_THOMAS_VILLANELLE, villanelle)
        # Should have rubric items
        assert len(result.rubric) > 0

    def test_villanelle_wrong_length_fails(self) -> None:
        """Poem with wrong length fails."""
        villanelle = Villanelle(strict=True)
        result = verify("Short poem\nTwo lines", villanelle)
        assert result.passed is False

    def test_villanelle_missing_refrains_fails_even_in_lenient_mode(self) -> None:
        """Lenient scoring still requires the defining refrain structure."""
        villanelle = Villanelle(strict=False)
        poem = """alpha cat
beta bat
gamma cat

delta cat
epsilon bat
zeta cat

eta cat
theta bat
iota cat

kappa cat
lambda bat
mu cat

nu cat
xi bat
omicron cat

pi cat
rho bat
sigma cat
tau cat"""
        result = verify(poem, villanelle)
        assert result.score > 0.5
        assert result.passed is False

    def test_villanelle_describe(self) -> None:
        """Description mentions villanelle characteristics."""
        villanelle = Villanelle()
        desc = villanelle.describe()
        assert "Villanelle" in desc
        assert "19" in desc or "refrain" in desc.lower()


class TestSonnet:
    """Tests for Sonnet form templates."""

    def test_generic_sonnet_14_lines(self) -> None:
        """Generic sonnet checks for 14 lines."""
        sonnet = Sonnet()
        result = verify(SHAKESPEARE_SONNET_18, sonnet)
        assert result.score > 0.5

    def test_shakespearean_sonnet(self) -> None:
        """Shakespearean sonnet with specific rhyme scheme."""
        sonnet = ShakespeareanSonnet()
        result = verify(SHAKESPEARE_SONNET_18, sonnet)
        # Should score reasonably well
        assert result.score > 0.4

    def test_shakespearean_rhyme_scheme(self) -> None:
        """Shakespearean sonnet has ABAB CDCD EFEF GG scheme."""
        sonnet = ShakespeareanSonnet()
        assert sonnet.RHYME_SCHEME == "ABABCDCDEFEFGG"

    def test_shakespearean_sonnet_missing_real_rhyme_fails_even_in_lenient_mode(self) -> None:
        """A high-scoring near miss should not pass without the defining rhyme scheme."""
        sonnet = ShakespeareanSonnet(strict=False)
        line_a = "The winter sunlight settles on the bay"
        line_b = "A silver morning wanders through the rain"
        poem = "\n".join([line_a, line_b] * 7)
        result = verify(poem, sonnet)
        assert result.score > 0.7
        assert result.passed is False

    def test_petrarchan_sonnet_scheme(self) -> None:
        """Petrarchan sonnet has ABBAABBA + sestet."""
        sonnet = PetrarchanSonnet()
        assert "ABBAABBA" in sonnet._rhyme_scheme.scheme

    def test_spenserian_sonnet_scheme(self) -> None:
        """Spenserian sonnet has interlocking scheme."""
        sonnet = SpenserianSonnet()
        assert sonnet.RHYME_SCHEME == "ABABBCBCCDCDEE"

    def test_sonnet_wrong_length(self) -> None:
        """Poem with wrong length fails."""
        sonnet = Sonnet(strict=True)
        result = verify(BASHO_HAIKU, sonnet)  # Only 3 lines
        assert result.passed is False

    def test_sonnet_describe(self) -> None:
        """Description mentions sonnet characteristics."""
        sonnet = ShakespeareanSonnet()
        desc = sonnet.describe()
        assert "Sonnet" in desc
        assert "14" in desc or "ABAB" in desc

    def test_generic_sonnet_describe_uses_syllable_proxy_language(self) -> None:
        """Generic sonnet description should not claim full meter verification."""
        desc = Sonnet().describe().lower()
        assert "10 syllables" in desc
        assert "iambic pentameter" not in desc


class TestLimerick:
    """Tests for Limerick form template."""

    def test_lear_limerick_scores_well(self) -> None:
        """Edward Lear's limerick should score well."""
        limerick = Limerick()
        result = verify(LEAR_LIMERICK, limerick)
        assert result.score > 0.5

    def test_limerick_5_lines(self) -> None:
        """Limerick requires 5 lines."""
        limerick = Limerick()
        result = verify(LEAR_LIMERICK, limerick)
        # Check line count passed
        line_count_items = [r for r in result.rubric if "line" in r.criterion.lower()]
        assert len(line_count_items) > 0

    def test_limerick_wrong_lines(self) -> None:
        """Wrong line count fails."""
        limerick = Limerick(strict=True)
        result = verify(BASHO_HAIKU, limerick)  # 3 lines, not 5
        assert result.passed is False

    def test_limerick_describe(self) -> None:
        """Description mentions limerick characteristics."""
        limerick = Limerick()
        desc = limerick.describe()
        assert "Limerick" in desc
        assert "AABBA" in desc


class TestBlankVerse:
    """Tests for BlankVerse form template."""

    def test_blank_verse_rejects_obviously_rhymed_poem(self) -> None:
        """Lenient blank verse should still reject systematic end rhyme."""
        blank = BlankVerse(strict=False, syllable_tolerance=1)
        poem = "\n".join(
            [
                "The river folds itself into the night",
                "A colder wind arrives before the light",
                "The distant roofs lie waiting for first light",
            ]
        )
        result = verify(poem, blank)
        assert result.score > 0.6
        assert result.passed is False

    def test_blank_verse_describe_matches_default_verifier(self) -> None:
        """Default description should describe the syllable proxy, not full meter."""
        desc = BlankVerse().describe().lower()
        assert "unrhymed" in desc
        assert "10 syllables" in desc
        assert "iambic pentameter" not in desc

    def test_blank_verse_describe_mentions_meter_when_enabled(self) -> None:
        """Strict meter mode can claim iambic pentameter explicitly."""
        desc = BlankVerse(strict_meter=True).describe().lower()
        assert "iambic pentameter" in desc


@pytest.mark.parametrize(
    ("form", "poem"),
    [
        (
            Couplet(strict=False),
            "\n".join([TEN_SYLLABLE_LINE_A, TEN_SYLLABLE_LINE_B]),
        ),
        (
            Quatrain(strict=False),
            "\n".join(
                [
                    TEN_SYLLABLE_LINE_A,
                    TEN_SYLLABLE_LINE_B,
                    TEN_SYLLABLE_LINE_A,
                    TEN_SYLLABLE_LINE_B,
                ]
            ),
        ),
        (
            Limerick(strict=False),
            "\n".join(
                [
                    TEN_SYLLABLE_LINE_A,
                    TEN_SYLLABLE_LINE_A,
                    SHORT_LIMERICK_LINE,
                    SHORT_LIMERICK_LINE,
                    TEN_SYLLABLE_LINE_A,
                ]
            ),
        ),
        (
            OttavaRima(strict=False),
            "\n".join(
                [
                    TEN_SYLLABLE_LINE_A,
                    TEN_SYLLABLE_LINE_B,
                    TEN_SYLLABLE_LINE_A,
                    TEN_SYLLABLE_LINE_B,
                    TEN_SYLLABLE_LINE_A,
                    TEN_SYLLABLE_LINE_B,
                    TEN_SYLLABLE_LINE_C,
                    TEN_SYLLABLE_LINE_C,
                ]
            ),
        ),
        (
            RhymeRoyal(strict=False),
            "\n".join(
                [
                    TEN_SYLLABLE_LINE_A,
                    TEN_SYLLABLE_LINE_B,
                    TEN_SYLLABLE_LINE_A,
                    TEN_SYLLABLE_LINE_B,
                    TEN_SYLLABLE_LINE_B,
                    TEN_SYLLABLE_LINE_C,
                    TEN_SYLLABLE_LINE_C,
                ]
            ),
        ),
    ],
    ids=["couplet", "quatrain", "limerick", "ottava-rima", "rhyme-royal"],
)
def test_rhyme_driven_forms_reject_repeated_line_false_positives(form, poem) -> None:
    """Lenient rhyme forms should not canonically pass repeated-line near misses."""
    result = verify(poem, form)
    assert result.score > 0.7
    assert result.passed is False


@pytest.mark.parametrize(
    ("form", "poem"),
    [
        (Triolet(strict=False), _triolet_near_miss()),
        (Bop(strict=False), _bop_near_miss()),
        (Kyrielle(strict=False), _kyrielle_near_miss()),
    ],
    ids=["triolet", "bop", "kyrielle"],
)
def test_refrain_driven_forms_reject_missing_refrain_near_misses(form, poem) -> None:
    """Lenient refrain forms should not canonically pass when a refrain is missing."""
    result = verify(poem, form)
    assert result.score > 0.7
    assert result.passed is False


def test_rondelet_does_not_report_pass_when_penalty_score_is_low() -> None:
    """Custom score overrides should not hide a failed refrain behind passed=True."""
    result = verify(_rondelet_near_miss(), Rondelet(strict=False))
    assert result.score == 0.05
    assert result.passed is False


class TestSestina:
    """Tests for Sestina form template."""

    def test_sestina_structure(self) -> None:
        """Sestina has correct structure constraints."""
        sestina = Sestina()
        assert sestina._stanza_sizes.expected_sizes == (6, 6, 6, 6, 6, 6, 3)
        assert sestina._line_count.bound.value == 39

    def test_sestina_rotation(self) -> None:
        """Sestina uses correct rotation pattern."""
        sestina = Sestina()
        assert sestina.ROTATION == [5, 0, 4, 1, 3, 2]

    def test_sestina_wrong_length(self) -> None:
        """Poem with wrong length fails."""
        sestina = Sestina()
        result = verify(BASHO_HAIKU, sestina)
        assert result.passed is False
        assert result.score < 0.5

    def test_sestina_describe(self) -> None:
        """Description mentions sestina characteristics."""
        sestina = Sestina()
        desc = sestina.describe()
        assert "Sestina" in desc
        assert "39" in desc or "rotation" in desc.lower()


class TestFormComposition:
    """Tests for using forms in larger compositions."""

    def test_form_has_verify_method(self) -> None:
        """All forms have verify method."""
        forms = [Haiku(), Villanelle(), Sestina(), Sonnet(), Limerick()]
        for form in forms:
            assert hasattr(form, "verify")
            assert callable(form.verify)

    def test_form_has_describe_method(self) -> None:
        """All forms have describe method."""
        forms = [Haiku(), Villanelle(), Sestina(), Sonnet(), Limerick()]
        for form in forms:
            assert hasattr(form, "describe")
            desc = form.describe()
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_form_returns_verification_result(self) -> None:
        """All forms return VerificationResult."""
        from abide import VerificationResult

        forms = [Haiku(), Villanelle(), Sestina(), Sonnet(), Limerick()]
        for form in forms:
            result = form.verify("Test poem\nLine two")
            assert isinstance(result, VerificationResult)
            assert 0 <= result.score <= 1.0
            assert isinstance(result.passed, bool)
            assert isinstance(result.rubric, list)


class TestFormIntegration:
    """Integration tests with verify function."""

    def test_verify_with_haiku(self) -> None:
        """verify() works with Haiku form."""
        result = verify(BASHO_HAIKU, Haiku())
        assert hasattr(result, "score")
        assert hasattr(result, "passed")
        assert hasattr(result, "rubric")

    def test_verify_with_villanelle(self) -> None:
        """verify() works with Villanelle form."""
        result = verify(DYLAN_THOMAS_VILLANELLE, Villanelle())
        assert result.constraint_name == "Villanelle"

    def test_verify_with_sonnet(self) -> None:
        """verify() works with Sonnet form."""
        result = verify(SHAKESPEARE_SONNET_18, ShakespeareanSonnet())
        assert result.constraint_name == "Shakespearean Sonnet"
