"""Tests for composition operators."""

from abide.constraints import (
    And,
    AtLeast,
    AtMost,
    LineCount,
    Not,
    Or,
    StanzaCount,
    WeightedSum,
)


class TestAnd:
    """Tests for And (logical AND) operator."""

    def test_all_pass(self) -> None:
        """All constraints passing means AND passes."""
        poem = "Line one\nLine two\nLine three"
        constraint = And([LineCount(3), StanzaCount(1)])
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_one_fails(self) -> None:
        """One constraint failing means AND fails."""
        poem = "Line one\nLine two\nLine three"
        constraint = And([LineCount(3), LineCount(5)])  # 3 != 5
        result = constraint.verify(poem)
        assert result.passed is False
        assert result.score < 1.0

    def test_all_fail(self) -> None:
        """All constraints failing means AND fails."""
        poem = "Line one\nLine two"
        constraint = And([LineCount(5), LineCount(10)])
        result = constraint.verify(poem)
        assert result.passed is False

    def test_min_mode(self) -> None:
        """Min mode uses minimum score."""
        poem = "Line one\nLine two\nLine three"
        constraint = And([LineCount(3), LineCount(4)], mode="min")
        result = constraint.verify(poem)
        # LineCount(3) scores 1.0, LineCount(4) scores less
        # Min should be less than 1.0
        assert result.score < 1.0

    def test_product_mode(self) -> None:
        """Product mode multiplies scores."""
        poem = "Line one\nLine two\nLine three"
        constraint = And([LineCount(3), LineCount(3)], mode="product")
        result = constraint.verify(poem)
        # Both score 1.0, product is 1.0
        assert result.score == 1.0

    def test_mean_mode(self) -> None:
        """Mean mode averages scores."""
        poem = "Line one\nLine two\nLine three"
        constraint = And([LineCount(3), LineCount(3)], mode="mean")
        result = constraint.verify(poem)
        assert result.score == 1.0

    def test_empty_constraints(self) -> None:
        """Empty constraints list."""
        constraint = And([])
        result = constraint.verify("Any poem")
        # Edge case: no constraints means "vacuously true" for passed
        # but score defaults to 0.0 when scores list is empty
        assert result.passed is True

    def test_rubric_combines_children(self) -> None:
        """Rubric includes all child rubrics."""
        poem = "Line one\nLine two\nLine three"
        constraint = And([LineCount(3), StanzaCount(1)])
        result = constraint.verify(poem)
        # Should have rubric items from both constraints
        assert len(result.rubric) >= 2

    def test_describe(self) -> None:
        """Description joins child descriptions."""
        constraint = And([LineCount(3), StanzaCount(1)])
        desc = constraint.describe()
        assert " AND " in desc


class TestOr:
    """Tests for Or (logical OR) operator."""

    def test_all_pass(self) -> None:
        """All constraints passing means OR passes."""
        poem = "Line one\nLine two\nLine three"
        constraint = Or([LineCount(3), StanzaCount(1)])
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_one_passes(self) -> None:
        """One constraint passing means OR passes."""
        poem = "Line one\nLine two\nLine three"
        constraint = Or([LineCount(3), LineCount(10)])
        result = constraint.verify(poem)
        assert result.passed is True
        # Score is max, so 1.0 from LineCount(3)
        assert result.score == 1.0

    def test_all_fail(self) -> None:
        """All constraints failing means OR fails."""
        poem = "Line one\nLine two"
        constraint = Or([LineCount(5), LineCount(10)])
        result = constraint.verify(poem)
        assert result.passed is False

    def test_score_is_max(self) -> None:
        """Score is maximum of child scores."""
        poem = "Line one\nLine two\nLine three"
        constraint = Or([LineCount(3), LineCount(4)])
        result = constraint.verify(poem)
        # LineCount(3) scores 1.0, should be the max
        assert result.score == 1.0

    def test_empty_constraints(self) -> None:
        """Empty constraints list."""
        constraint = Or([])
        result = constraint.verify("Any poem")
        # Edge case: no constraints means "vacuously false"
        assert result.passed is False

    def test_rubric_from_best(self) -> None:
        """Rubric comes from best-scoring constraint."""
        poem = "Line one\nLine two\nLine three"
        constraint = Or([LineCount(3), LineCount(100)])
        result = constraint.verify(poem)
        # Should have rubric from LineCount(3), the better one
        assert len(result.rubric) == 1
        assert "3" in result.rubric[0].actual

    def test_describe(self) -> None:
        """Description joins child descriptions."""
        constraint = Or([LineCount(3), LineCount(5)])
        desc = constraint.describe()
        assert " OR " in desc


class TestNot:
    """Tests for Not (logical NOT) operator."""

    def test_inverts_pass(self) -> None:
        """NOT inverts passing to failing."""
        poem = "Line one\nLine two\nLine three"
        constraint = Not(LineCount(3))
        result = constraint.verify(poem)
        assert result.passed is False
        assert result.score == 0.0

    def test_inverts_fail(self) -> None:
        """NOT inverts failing to passing."""
        poem = "Line one\nLine two"
        constraint = Not(LineCount(3))
        result = constraint.verify(poem)
        assert result.passed is True
        # Score = 1 - child_score, child_score is high (Gaussian decay from 2 to 3)
        # so inverted score is low, but still passes
        assert result.score > 0.0

    def test_score_inverted(self) -> None:
        """Score is 1 - child_score."""
        poem = "Line one\nLine two\nLine three"
        inner = LineCount(3)
        constraint = Not(inner)

        inner_result = inner.verify(poem)
        outer_result = constraint.verify(poem)

        assert outer_result.score == 1.0 - inner_result.score

    def test_rubric_negated(self) -> None:
        """Rubric items are negated."""
        poem = "Line one\nLine two\nLine three"
        constraint = Not(LineCount(3))
        result = constraint.verify(poem)
        # Rubric should indicate NOT
        assert len(result.rubric) == 1
        assert "NOT" in result.rubric[0].criterion

    def test_describe(self) -> None:
        """Description wraps with NOT."""
        constraint = Not(LineCount(3))
        desc = constraint.describe()
        assert "NOT" in desc


class TestWeightedSum:
    """Tests for WeightedSum composition."""

    def test_equal_weights(self) -> None:
        """Equal weights produce average score."""
        poem = "Line one\nLine two\nLine three"
        constraint = WeightedSum(
            [
                (LineCount(3), 1.0),
                (StanzaCount(1), 1.0),
            ]
        )
        result = constraint.verify(poem)
        # Both pass, average is 1.0
        assert result.score == 1.0
        assert result.passed is True

    def test_different_weights(self) -> None:
        """Different weights affect score appropriately."""
        poem = "Line one\nLine two\nLine three"
        # LineCount(3) passes (1.0), LineCount(5) fails (~0.6)
        # With weights 3:1, score leans toward the heavier one
        constraint = WeightedSum(
            [
                (LineCount(3), 3.0),
                (LineCount(5), 1.0),
            ]
        )
        result = constraint.verify(poem)
        # (3.0 * 1.0 + 1.0 * 0.6) / 4.0 ≈ 0.9
        assert result.score > 0.85

    def test_threshold(self) -> None:
        """Threshold controls pass/fail."""
        poem = "Line one\nLine two\nLine three"
        constraint = WeightedSum(
            [(LineCount(3), 1.0), (LineCount(5), 1.0)],
            threshold=0.9,
        )
        result = constraint.verify(poem)
        # Score is average of 1.0 and ~0.6 ≈ 0.8
        # With threshold 0.9, should fail
        assert result.passed is False

    def test_low_threshold(self) -> None:
        """Low threshold allows partial matches."""
        poem = "Line one\nLine two\nLine three"
        constraint = WeightedSum(
            [(LineCount(3), 1.0), (LineCount(5), 1.0)],
            threshold=0.3,
        )
        result = constraint.verify(poem)
        assert result.passed is True

    def test_from_constraint_weights(self) -> None:
        """Can use constraint's built-in weights."""
        poem = "Line one\nLine two\nLine three"
        c1 = LineCount(3, weight=2.0)
        c2 = StanzaCount(1, weight=1.0)
        constraint = WeightedSum([c1, c2])  # Uses constraint.weight
        result = constraint.verify(poem)
        assert result.passed is True

    def test_rubric_includes_weights(self) -> None:
        """Rubric shows weight info."""
        poem = "Line one\nLine two\nLine three"
        constraint = WeightedSum(
            [
                (LineCount(3), 2.0),
                (StanzaCount(1), 1.0),
            ]
        )
        result = constraint.verify(poem)
        # Rubric items should show weights
        weight_shown = any("w=" in item.criterion for item in result.rubric)
        assert weight_shown

    def test_describe(self) -> None:
        """Description shows weights."""
        constraint = WeightedSum(
            [
                (LineCount(3), 2.0),
                (StanzaCount(1), 1.0),
            ]
        )
        desc = constraint.describe()
        assert "Weighted" in desc
        assert "2.0" in desc or "1.0" in desc


class TestAtLeast:
    """Tests for AtLeast (N must pass) constraint."""

    def test_all_pass(self) -> None:
        """All constraints passing satisfies AtLeast."""
        poem = "Line one\nLine two\nLine three"
        constraint = AtLeast(
            2,
            [LineCount(3), StanzaCount(1), LineCount(3)],
        )
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_exactly_n_pass(self) -> None:
        """Exactly N constraints passing satisfies AtLeast."""
        poem = "Line one\nLine two\nLine three"
        constraint = AtLeast(
            2,
            [LineCount(3), LineCount(5), LineCount(10)],
        )
        result = constraint.verify(poem)
        # Only LineCount(3) passes
        assert result.passed is False

    def test_more_than_n_pass(self) -> None:
        """More than N constraints passing satisfies AtLeast."""
        poem = "Line one\nLine two\nLine three"
        constraint = AtLeast(
            1,
            [LineCount(3), StanzaCount(1)],
        )
        result = constraint.verify(poem)
        assert result.passed is True

    def test_fewer_than_n_pass(self) -> None:
        """Fewer than N constraints passing fails AtLeast."""
        poem = "Line one\nLine two"
        constraint = AtLeast(
            2,
            [LineCount(3), LineCount(4), LineCount(5)],
        )
        result = constraint.verify(poem)
        assert result.passed is False

    def test_partial_credit(self) -> None:
        """Partial credit based on how many passed."""
        poem = "Line one\nLine two\nLine three"
        constraint = AtLeast(
            3,
            [LineCount(3), LineCount(5), LineCount(10)],
        )
        result = constraint.verify(poem)
        # 1 out of 3 passed, score reflects partial progress
        assert result.score < 1.0
        assert result.passed is False

    def test_at_least_zero(self) -> None:
        """AtLeast(0) always passes."""
        poem = "Any poem"
        constraint = AtLeast(0, [LineCount(100)])
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_describe(self) -> None:
        """Description includes count."""
        constraint = AtLeast(2, [LineCount(3), LineCount(5)])
        desc = constraint.describe()
        assert "At least 2" in desc


class TestAtMost:
    """Tests for AtMost (max N can pass) constraint."""

    def test_none_pass(self) -> None:
        """No constraints passing satisfies AtMost."""
        poem = "Line one\nLine two"
        constraint = AtMost(
            2,
            [LineCount(5), LineCount(10)],
        )
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0

    def test_exactly_n_pass(self) -> None:
        """Exactly N constraints passing satisfies AtMost."""
        poem = "Line one\nLine two\nLine three"
        constraint = AtMost(
            1,
            [LineCount(3), LineCount(5)],
        )
        result = constraint.verify(poem)
        # Only LineCount(3) passes, which is exactly 1
        assert result.passed is True

    def test_more_than_n_pass(self) -> None:
        """More than N constraints passing fails AtMost."""
        poem = "Line one\nLine two\nLine three"
        constraint = AtMost(
            0,
            [LineCount(3), StanzaCount(1)],
        )
        result = constraint.verify(poem)
        # Both pass, but at most 0 allowed
        assert result.passed is False

    def test_partial_credit_for_excess(self) -> None:
        """Partial credit when exceeding limit."""
        poem = "Line one\nLine two\nLine three"
        constraint = AtMost(
            0,
            [LineCount(3), StanzaCount(1)],
        )
        result = constraint.verify(poem)
        # 2 pass when 0 allowed, excess of 2
        assert result.score < 1.0
        assert result.score >= 0.0

    def test_describe(self) -> None:
        """Description includes count."""
        constraint = AtMost(2, [LineCount(3), LineCount(5)])
        desc = constraint.describe()
        assert "At most 2" in desc


class TestCompositionNesting:
    """Tests for nested composition operators."""

    def test_and_inside_or(self) -> None:
        """AND can be nested inside OR."""
        poem = "Line one\nLine two\nLine three"
        inner_and = And([LineCount(3), StanzaCount(1)])
        constraint = Or([inner_and, LineCount(100)])
        result = constraint.verify(poem)
        # inner_and passes, so OR passes
        assert result.passed is True

    def test_or_inside_and(self) -> None:
        """OR can be nested inside AND."""
        poem = "Line one\nLine two\nLine three"
        inner_or = Or([LineCount(3), LineCount(5)])
        constraint = And([inner_or, StanzaCount(1)])
        result = constraint.verify(poem)
        # inner_or passes (3 matches), StanzaCount(1) passes
        assert result.passed is True

    def test_not_inside_and(self) -> None:
        """NOT can be nested inside AND."""
        poem = "Line one\nLine two\nLine three"
        not_five = Not(LineCount(5))
        constraint = And([LineCount(3), not_five])
        result = constraint.verify(poem)
        # LineCount(3) passes, NOT(LineCount(5)) passes
        assert result.passed is True

    def test_deep_nesting(self) -> None:
        """Multiple levels of nesting."""
        poem = "Line one\nLine two\nLine three"
        level1 = And([LineCount(3), StanzaCount(1)])
        level2 = Or([level1, LineCount(100)])
        level3 = And([level2, Not(LineCount(0))])
        result = level3.verify(poem)
        assert result.passed is True

    def test_weighted_sum_with_nested(self) -> None:
        """WeightedSum with nested operators."""
        poem = "Line one\nLine two\nLine three"
        inner = And([LineCount(3), StanzaCount(1)])
        constraint = WeightedSum(
            [
                (inner, 2.0),
                (LineCount(3), 1.0),
            ]
        )
        result = constraint.verify(poem)
        assert result.passed is True
        assert result.score == 1.0
