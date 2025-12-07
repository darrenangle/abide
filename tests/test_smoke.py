"""
Smoke tests to verify basic package setup and ground truth data.

These tests verify that:
- The package can be imported
- Ground truth poem fixtures are correctly structured
- Basic structural properties of test poems are accurate
"""

import pytest

from tests.conftest import (
    ALL_POEMS,
    HAIKU_POEMS,
    SHAKESPEAREAN_SONNETS,
    VILLANELLE_POEMS,
    PoemSpec,
)


class TestPackageImports:
    """Test that package structure is correct."""

    def test_import_abide(self) -> None:
        """Can import main package."""
        import abide

        assert hasattr(abide, "__version__")

    def test_import_primitives(self) -> None:
        """Can import primitives subpackage."""
        from abide import primitives

        assert primitives is not None

    def test_import_constraints(self) -> None:
        """Can import constraints subpackage."""
        from abide import constraints

        assert constraints is not None

    def test_import_forms(self) -> None:
        """Can import forms subpackage."""
        from abide import forms

        assert forms is not None

    def test_import_integration(self) -> None:
        """Can import integration subpackage."""
        from abide import integration

        assert integration is not None


class TestGroundTruthStructure:
    """Verify ground truth poem specifications are correctly structured."""

    @pytest.mark.parametrize("poem", ALL_POEMS, ids=lambda p: f"{p.author}:{p.name}")
    def test_poem_has_required_fields(self, poem: PoemSpec) -> None:
        """All poems have required fields."""
        assert poem.name
        assert poem.author
        assert poem.form
        assert poem.text
        assert poem.expected_line_count > 0
        assert poem.expected_stanza_count > 0
        assert len(poem.expected_stanza_sizes) == poem.expected_stanza_count

    @pytest.mark.parametrize("poem", ALL_POEMS, ids=lambda p: f"{p.author}:{p.name}")
    def test_poem_line_count_matches(self, poem: PoemSpec) -> None:
        """Verify expected_line_count matches actual text."""
        lines = [line for line in poem.text.strip().split("\n") if line.strip()]
        assert len(lines) == poem.expected_line_count, (
            f"{poem.name}: expected {poem.expected_line_count} lines, "
            f"got {len(lines)}"
        )

    @pytest.mark.parametrize("poem", ALL_POEMS, ids=lambda p: f"{p.author}:{p.name}")
    def test_poem_stanza_count_matches(self, poem: PoemSpec) -> None:
        """Verify expected_stanza_count matches actual text."""
        # Split on blank lines
        stanzas = [s.strip() for s in poem.text.strip().split("\n\n") if s.strip()]
        assert len(stanzas) == poem.expected_stanza_count, (
            f"{poem.name}: expected {poem.expected_stanza_count} stanzas, "
            f"got {len(stanzas)}"
        )

    @pytest.mark.parametrize("poem", ALL_POEMS, ids=lambda p: f"{p.author}:{p.name}")
    def test_poem_stanza_sizes_match(self, poem: PoemSpec) -> None:
        """Verify expected_stanza_sizes matches actual text."""
        stanzas = [s.strip() for s in poem.text.strip().split("\n\n") if s.strip()]
        actual_sizes = [len([ln for ln in s.split("\n") if ln.strip()]) for s in stanzas]
        assert actual_sizes == poem.expected_stanza_sizes, (
            f"{poem.name}: expected sizes {poem.expected_stanza_sizes}, "
            f"got {actual_sizes}"
        )


class TestHaikuGroundTruth:
    """Verify haiku ground truth specifics."""

    @pytest.mark.parametrize("poem", HAIKU_POEMS, ids=lambda p: p.name)
    def test_haiku_has_three_lines(self, poem: PoemSpec) -> None:
        """Haiku should have exactly 3 lines."""
        assert poem.expected_line_count == 3

    @pytest.mark.parametrize("poem", HAIKU_POEMS, ids=lambda p: p.name)
    def test_haiku_has_one_stanza(self, poem: PoemSpec) -> None:
        """Haiku should have exactly 1 stanza."""
        assert poem.expected_stanza_count == 1

    @pytest.mark.parametrize("poem", HAIKU_POEMS, ids=lambda p: p.name)
    def test_haiku_has_syllable_spec(self, poem: PoemSpec) -> None:
        """Haiku should specify syllables per line."""
        assert poem.expected_syllables_per_line is not None
        assert len(poem.expected_syllables_per_line) == 3


class TestVillanelleGroundTruth:
    """Verify villanelle ground truth specifics."""

    @pytest.mark.parametrize("poem", VILLANELLE_POEMS, ids=lambda p: p.name)
    def test_villanelle_has_19_lines(self, poem: PoemSpec) -> None:
        """Villanelle should have exactly 19 lines."""
        assert poem.expected_line_count == 19

    @pytest.mark.parametrize("poem", VILLANELLE_POEMS, ids=lambda p: p.name)
    def test_villanelle_has_six_stanzas(self, poem: PoemSpec) -> None:
        """Villanelle should have exactly 6 stanzas."""
        assert poem.expected_stanza_count == 6

    @pytest.mark.parametrize("poem", VILLANELLE_POEMS, ids=lambda p: p.name)
    def test_villanelle_stanza_pattern(self, poem: PoemSpec) -> None:
        """Villanelle should have [3,3,3,3,3,4] stanza pattern."""
        assert poem.expected_stanza_sizes == [3, 3, 3, 3, 3, 4]

    @pytest.mark.parametrize("poem", VILLANELLE_POEMS, ids=lambda p: p.name)
    def test_villanelle_has_refrains(self, poem: PoemSpec) -> None:
        """Villanelle should specify refrains."""
        assert poem.expected_refrains is not None
        assert len(poem.expected_refrains) >= 1  # At least A1 refrain


class TestSonnetGroundTruth:
    """Verify sonnet ground truth specifics."""

    @pytest.mark.parametrize("poem", SHAKESPEAREAN_SONNETS, ids=lambda p: p.name)
    def test_sonnet_has_14_lines(self, poem: PoemSpec) -> None:
        """Sonnet should have exactly 14 lines."""
        assert poem.expected_line_count == 14

    @pytest.mark.parametrize("poem", SHAKESPEAREAN_SONNETS, ids=lambda p: p.name)
    def test_sonnet_has_rhyme_scheme(self, poem: PoemSpec) -> None:
        """Sonnet should specify rhyme scheme."""
        assert poem.expected_rhyme_scheme is not None
        assert poem.expected_rhyme_scheme == "ABABCDCDEFEFGG"

    @pytest.mark.parametrize("poem", SHAKESPEAREAN_SONNETS, ids=lambda p: p.name)
    def test_sonnet_has_syllable_spec(self, poem: PoemSpec) -> None:
        """Sonnet should specify 10 syllables per line (iambic pentameter)."""
        assert poem.expected_syllables_per_line is not None
        assert poem.expected_syllables_per_line == [10] * 14
