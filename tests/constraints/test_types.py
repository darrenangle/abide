"""Tests for constraint types."""

import pytest

from abide.constraints.types import (
    BoundType,
    ConstraintType,
    NumericBound,
    RubricItem,
    VerificationResult,
)


class TestConstraintType:
    """Tests for ConstraintType enum."""

    def test_all_types_exist(self) -> None:
        """All expected constraint types exist."""
        assert ConstraintType.STRUCTURAL is not None
        assert ConstraintType.RELATIONAL is not None
        assert ConstraintType.PROSODIC is not None
        assert ConstraintType.SONIC is not None
        assert ConstraintType.SEMANTIC is not None
        assert ConstraintType.COMPOSITE is not None

    def test_types_are_distinct(self) -> None:
        """Constraint types are distinct."""
        types = list(ConstraintType)
        assert len(types) == len(set(types))


class TestBoundType:
    """Tests for BoundType enum."""

    def test_all_bound_types_exist(self) -> None:
        """All expected bound types exist."""
        assert BoundType.EXACT is not None
        assert BoundType.MIN is not None
        assert BoundType.MAX is not None
        assert BoundType.RANGE is not None


class TestNumericBound:
    """Tests for NumericBound dataclass."""

    def test_exact_factory(self) -> None:
        """exact() creates EXACT bound."""
        bound = NumericBound.exact(14)
        assert bound.bound_type == BoundType.EXACT
        assert bound.value == 14

    def test_at_least_factory(self) -> None:
        """at_least() creates MIN bound."""
        bound = NumericBound.at_least(3)
        assert bound.bound_type == BoundType.MIN
        assert bound.value == 3

    def test_at_most_factory(self) -> None:
        """at_most() creates MAX bound."""
        bound = NumericBound.at_most(10)
        assert bound.bound_type == BoundType.MAX
        assert bound.value == 10

    def test_between_factory(self) -> None:
        """between() creates RANGE bound."""
        bound = NumericBound.between(5, 10)
        assert bound.bound_type == BoundType.RANGE
        assert bound.min_value == 5
        assert bound.max_value == 10

    def test_exact_check(self) -> None:
        """EXACT bound checks correctly."""
        bound = NumericBound.exact(14)
        assert bound.check(14) is True
        assert bound.check(13) is False
        assert bound.check(15) is False

    def test_min_check(self) -> None:
        """MIN bound checks correctly."""
        bound = NumericBound.at_least(5)
        assert bound.check(5) is True
        assert bound.check(10) is True
        assert bound.check(4) is False
        assert bound.check(0) is False

    def test_max_check(self) -> None:
        """MAX bound checks correctly."""
        bound = NumericBound.at_most(10)
        assert bound.check(10) is True
        assert bound.check(5) is True
        assert bound.check(0) is True
        assert bound.check(11) is False

    def test_range_check(self) -> None:
        """RANGE bound checks correctly."""
        bound = NumericBound.between(5, 10)
        assert bound.check(5) is True
        assert bound.check(7) is True
        assert bound.check(10) is True
        assert bound.check(4) is False
        assert bound.check(11) is False

    def test_describe_exact(self) -> None:
        """EXACT bound describes itself."""
        bound = NumericBound.exact(14)
        assert bound.describe() == "exactly 14"

    def test_describe_min(self) -> None:
        """MIN bound describes itself."""
        bound = NumericBound.at_least(3)
        assert bound.describe() == "at least 3"

    def test_describe_max(self) -> None:
        """MAX bound describes itself."""
        bound = NumericBound.at_most(10)
        assert bound.describe() == "at most 10"

    def test_describe_range(self) -> None:
        """RANGE bound describes itself."""
        bound = NumericBound.between(5, 10)
        assert bound.describe() == "between 5 and 10"

    def test_invalid_range_missing_values(self) -> None:
        """RANGE bound requires both min and max."""
        with pytest.raises(ValueError, match="RANGE bounds require"):
            NumericBound(bound_type=BoundType.RANGE, min_value=5)

    def test_invalid_range_wrong_order(self) -> None:
        """RANGE bound requires min <= max."""
        with pytest.raises(ValueError, match="min_value must be"):
            NumericBound.between(10, 5)

    def test_invalid_exact_missing_value(self) -> None:
        """EXACT bound requires value."""
        with pytest.raises(ValueError, match="EXACT bounds require value"):
            NumericBound(bound_type=BoundType.EXACT)

    def test_frozen_dataclass(self) -> None:
        """NumericBound is immutable."""
        bound = NumericBound.exact(14)
        with pytest.raises(AttributeError):
            bound.value = 15  # type: ignore


class TestRubricItem:
    """Tests for RubricItem dataclass."""

    def test_create_passing_item(self) -> None:
        """Create a passing rubric item."""
        item = RubricItem(
            criterion="Line count",
            expected="exactly 14",
            actual="14",
            score=1.0,
            passed=True,
        )
        assert item.criterion == "Line count"
        assert item.passed is True
        assert item.score == 1.0

    def test_create_failing_item(self) -> None:
        """Create a failing rubric item."""
        item = RubricItem(
            criterion="Line count",
            expected="exactly 14",
            actual="12",
            score=0.6,
            passed=False,
            explanation="Poem is too short",
        )
        assert item.passed is False
        assert item.explanation == "Poem is too short"

    def test_str_passing(self) -> None:
        """String representation of passing item."""
        item = RubricItem(
            criterion="Line count",
            expected="exactly 14",
            actual="14",
            score=1.0,
            passed=True,
        )
        s = str(item)
        assert "[PASS]" in s
        assert "Line count" in s

    def test_str_failing(self) -> None:
        """String representation of failing item."""
        item = RubricItem(
            criterion="Line count",
            expected="exactly 14",
            actual="12",
            score=0.6,
            passed=False,
        )
        s = str(item)
        assert "[FAIL]" in s
        assert "Line count" in s


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_create_passing_result(self) -> None:
        """Create a passing result."""
        result = VerificationResult(
            score=1.0,
            passed=True,
            rubric=[
                RubricItem(
                    criterion="Test",
                    expected="pass",
                    actual="pass",
                    score=1.0,
                    passed=True,
                )
            ],
            constraint_name="Test Constraint",
            constraint_type=ConstraintType.STRUCTURAL,
        )
        assert result.score == 1.0
        assert result.passed is True
        assert len(result.rubric) == 1

    def test_str_representation(self) -> None:
        """String representation includes key info."""
        result = VerificationResult(
            score=0.75,
            passed=False,
            constraint_name="Test Constraint",
        )
        s = str(result)
        assert "FAIL" in s
        assert "Test Constraint" in s
        assert "0.75" in s

    def test_to_dict(self) -> None:
        """Serialization to dictionary."""
        result = VerificationResult(
            score=1.0,
            passed=True,
            rubric=[
                RubricItem(
                    criterion="Line count",
                    expected="exactly 14",
                    actual="14",
                    score=1.0,
                    passed=True,
                )
            ],
            constraint_name="Line Count",
            constraint_type=ConstraintType.STRUCTURAL,
            details={"actual": 14},
        )
        d = result.to_dict()
        assert d["score"] == 1.0
        assert d["passed"] is True
        assert d["constraint_name"] == "Line Count"
        assert d["constraint_type"] == "STRUCTURAL"
        assert len(d["rubric"]) == 1
        assert d["details"]["actual"] == 14

    def test_default_rubric(self) -> None:
        """Default rubric is empty list."""
        result = VerificationResult(score=0.5, passed=False)
        assert result.rubric == []

    def test_default_details(self) -> None:
        """Default details is empty dict."""
        result = VerificationResult(score=0.5, passed=False)
        assert result.details == {}
