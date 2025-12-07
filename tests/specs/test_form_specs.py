"""
Tests for FormSpec and instruction generation.

Tests verify that:
1. FormSpec correctly composes constraints
2. Instructions are generated in plain English
3. Verification results map back to instruction items
"""


from abide.specs import (
    FormSpec,
    haiku_spec,
    sestina_spec,
    shakespearean_sonnet_spec,
    tanka_spec,
    triolet_spec,
    villanelle_spec,
)
from tests.fixtures.poems import (
    HAIKU_SYNTHETIC_PERFECT,
    LIMERICK_SYNTHETIC_PERFECT,
    TANKA_SYNTHETIC_PERFECT,
    TRIOLET_SYNTHETIC_PERFECT,
)


class TestFormSpecBasics:
    """Test basic FormSpec functionality."""

    def test_haiku_spec_has_correct_items(self):
        """Haiku spec should have line_count and syllables."""
        spec = haiku_spec()
        assert len(spec) == 2
        item_ids = [item.id for item in spec]
        assert "line_count" in item_ids
        assert "syllables" in item_ids

    def test_shakespearean_spec_has_correct_items(self):
        """Shakespearean sonnet spec should have line_count, syllables, rhyme_scheme."""
        spec = shakespearean_sonnet_spec()
        assert len(spec) == 3
        item_ids = [item.id for item in spec]
        assert "line_count" in item_ids
        assert "syllables" in item_ids
        assert "rhyme_scheme" in item_ids

    def test_villanelle_spec_has_refrains(self):
        """Villanelle spec should have refrain constraints."""
        spec = villanelle_spec()
        item_ids = [item.id for item in spec]
        assert "refrain_a" in item_ids
        assert "refrain_b" in item_ids


class TestInstructionGeneration:
    """Test plain English instruction generation."""

    def test_haiku_full_instruction(self):
        """Haiku full instruction should be clear and complete."""
        spec = haiku_spec()
        instruction = spec.full_instruction()

        assert "Haiku" in instruction
        assert "3 lines" in instruction or "exactly 3" in instruction
        assert "5-7-5" in instruction or "syllable" in instruction

    def test_shakespearean_full_instruction(self):
        """Shakespearean sonnet instruction should mention key elements."""
        spec = shakespearean_sonnet_spec()
        instruction = spec.full_instruction()

        assert "Shakespearean Sonnet" in instruction
        assert "14" in instruction
        assert "ABAB" in instruction or "rhyme" in instruction.lower()
        assert "10 syllables" in instruction or "pentameter" in instruction.lower()

    def test_instruction_for_subset(self):
        """Should generate instructions for specific constraints only."""
        spec = shakespearean_sonnet_spec()
        instruction = spec.instruction_for("line_count")

        assert "14" in instruction
        assert "rhyme" not in instruction.lower()  # Should not include rhyme scheme

    def test_instructions_by_category(self):
        """Should filter instructions by category."""
        spec = shakespearean_sonnet_spec()

        structural = spec.instructions_by_category("structural")
        assert "14" in structural

        prosodic = spec.instructions_by_category("prosodic")
        assert "syllable" in prosodic.lower()

    def test_instruction_without_intro(self):
        """Should generate instruction without intro sentence."""
        spec = haiku_spec()
        instruction = spec.full_instruction(include_intro=False)

        assert not instruction.startswith("Write a Haiku")
        assert "- " in instruction  # Should still have bullet points


class TestFormSpecVerification:
    """Test verification through FormSpec."""

    def test_haiku_verify_synthetic(self):
        """Haiku spec should verify synthetic perfect poem."""
        spec = haiku_spec()
        results = spec.verify(HAIKU_SYNTHETIC_PERFECT)

        assert "line_count" in results
        assert "syllables" in results
        assert results["line_count"].score == 1.0
        assert results["syllables"].score == 1.0

    def test_tanka_verify_synthetic(self):
        """Tanka spec should verify synthetic perfect poem."""
        spec = tanka_spec()
        results = spec.verify(TANKA_SYNTHETIC_PERFECT)

        assert results["line_count"].score == 1.0
        assert results["syllables"].score == 1.0

    def test_verify_subset(self):
        """Should verify only specified constraints."""
        spec = shakespearean_sonnet_spec()
        results = spec.verify_subset(HAIKU_SYNTHETIC_PERFECT, "line_count")

        assert "line_count" in results
        assert "syllables" not in results
        assert "rhyme_scheme" not in results

    def test_weighted_score_perfect(self):
        """Perfect haiku should have weighted score of 1.0."""
        spec = haiku_spec()
        score = spec.weighted_score(HAIKU_SYNTHETIC_PERFECT)
        assert score == 1.0

    def test_weighted_score_wrong_form(self):
        """Wrong form should have lower weighted score than correct form."""
        spec = haiku_spec()
        # Use a limerick (5 lines) for haiku spec (3 lines)
        wrong_score = spec.weighted_score(LIMERICK_SYNTHETIC_PERFECT)
        correct_score = spec.weighted_score(HAIKU_SYNTHETIC_PERFECT)
        # Wrong form should score lower than correct form
        assert wrong_score < correct_score


class TestFormSpecSerialization:
    """Test FormSpec serialization."""

    def test_to_dict(self):
        """FormSpec should serialize to dictionary."""
        spec = haiku_spec()
        data = spec.to_dict()

        assert data["name"] == "Haiku"
        assert "items" in data
        assert len(data["items"]) == 2

        item = data["items"][0]
        assert "id" in item
        assert "instruction" in item
        assert "weight" in item
        assert "category" in item


class TestConstraintInstructions:
    """Test that individual constraints generate proper instructions."""

    def test_line_count_instruction(self):
        """LineCount should generate clear instruction."""
        from abide.constraints import LineCount

        constraint = LineCount(14)
        instruction = constraint.instruction()

        assert "14" in instruction
        assert "line" in instruction.lower()

    def test_syllables_per_line_uniform(self):
        """SyllablesPerLine with uniform count should be clear."""
        from abide.constraints import SyllablesPerLine

        constraint = SyllablesPerLine([10] * 14)
        instruction = constraint.instruction()

        assert "10" in instruction
        assert "syllable" in instruction.lower()

    def test_syllables_per_line_pattern(self):
        """SyllablesPerLine with pattern should list pattern."""
        from abide.constraints import SyllablesPerLine

        constraint = SyllablesPerLine([5, 7, 5])
        instruction = constraint.instruction()

        assert "5-7-5" in instruction

    def test_rhyme_scheme_instruction(self):
        """RhymeScheme should explain the pattern."""
        from abide.constraints import RhymeScheme

        constraint = RhymeScheme("ABABCDCDEFEFGG")
        instruction = constraint.instruction()

        assert "ABABCDCDEFEFGG" in instruction
        assert "rhyme" in instruction.lower()

    def test_refrain_instruction(self):
        """Refrain should specify which lines repeat."""
        from abide.constraints import Refrain

        constraint = Refrain(reference_line=0, repeat_at=[5, 11, 17])
        instruction = constraint.instruction()

        assert "1" in instruction  # Line 1 (0-indexed becomes 1)
        assert "6" in instruction  # Line 6 (5-indexed becomes 6)
        assert "refrain" in instruction.lower() or "repeat" in instruction.lower()


class TestComplexFormSpecs:
    """Test complex form specifications."""

    def test_villanelle_instruction_completeness(self):
        """Villanelle instruction should cover all requirements."""
        spec = villanelle_spec()
        instruction = spec.full_instruction()

        # Should mention structure
        assert "19" in instruction
        # Should mention rhyme
        assert "rhyme" in instruction.lower() or "ABA" in instruction
        # Should mention refrains
        assert "repeat" in instruction.lower() or "refrain" in instruction.lower()

    def test_sestina_instruction_completeness(self):
        """Sestina instruction should explain end-word rotation."""
        spec = sestina_spec()
        instruction = spec.full_instruction()

        # Should mention structure
        assert "39" in instruction or "6" in instruction
        # Should mention end words
        assert "end word" in instruction.lower() or "rotate" in instruction.lower()

    def test_triolet_verify_synthetic(self):
        """Triolet spec should verify synthetic perfect poem."""
        spec = triolet_spec()
        results = spec.verify(TRIOLET_SYNTHETIC_PERFECT)

        # At minimum, line count should pass
        assert results["line_count"].score == 1.0


class TestFormSpecChaining:
    """Test FormSpec method chaining."""

    def test_add_returns_self(self):
        """add() should return self for chaining."""
        from abide.constraints import LineCount

        spec = FormSpec(name="Test", description="Test form")
        result = spec.add("test", LineCount(10))

        assert result is spec
        assert len(spec) == 1

    def test_chained_adds(self):
        """Multiple add() calls should chain."""
        from abide.constraints import LineCount, RhymeScheme

        spec = (
            FormSpec(name="Test", description="Test form")
            .add("lines", LineCount(10))
            .add("rhyme", RhymeScheme("AABB"))
        )

        assert len(spec) == 2
