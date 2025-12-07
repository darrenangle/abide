"""Tests for verifiers framework integration."""

import pytest

from abide.forms import Haiku, ShakespeareanSonnet, Villanelle
from abide.verifiers import (
    AbideMajorPoeticForms,
    PoeticFormReward,
    make_poetic_forms_eval,
    make_reward_function,
)
from abide.verifiers.reward import RewardOutput


# Ground truth poems
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


class TestPoeticFormReward:
    """Tests for PoeticFormReward class."""

    def test_callable_interface(self) -> None:
        """Reward function is callable."""
        reward_fn = PoeticFormReward(Haiku())
        result = reward_fn(BASHO_HAIKU)
        assert isinstance(result, RewardOutput)

    def test_returns_score(self) -> None:
        """Returns score in [0, 1]."""
        reward_fn = PoeticFormReward(Haiku())
        result = reward_fn(BASHO_HAIKU)
        assert 0 <= result.score <= 1.0

    def test_returns_passed(self) -> None:
        """Returns passed boolean."""
        reward_fn = PoeticFormReward(Haiku())
        result = reward_fn(BASHO_HAIKU)
        assert isinstance(result.passed, bool)

    def test_returns_rubric(self) -> None:
        """Returns rubric string."""
        reward_fn = PoeticFormReward(Haiku())
        result = reward_fn(BASHO_HAIKU)
        assert isinstance(result.rubric, str)

    def test_returns_metadata(self) -> None:
        """Returns metadata dict."""
        reward_fn = PoeticFormReward(Haiku())
        result = reward_fn(BASHO_HAIKU)
        assert isinstance(result.metadata, dict)
        assert "constraint_name" in result.metadata

    def test_haiku_scores_well(self) -> None:
        """Haiku reward function scores haiku well."""
        reward_fn = PoeticFormReward(Haiku(strict=False))
        result = reward_fn(BASHO_HAIKU)
        assert result.score > 0.5

    def test_villanelle_scores_well(self) -> None:
        """Villanelle reward function scores villanelle well."""
        reward_fn = PoeticFormReward(Villanelle())
        result = reward_fn(DYLAN_THOMAS_VILLANELLE)
        assert result.score > 0.5

    def test_normalize_score(self) -> None:
        """Score normalization clamps to [0, 1]."""
        reward_fn = PoeticFormReward(Haiku(), normalize_score=True)
        result = reward_fn("Random text that won't match")
        assert 0 <= result.score <= 1.0

    def test_exclude_rubric(self) -> None:
        """Can exclude rubric from output."""
        reward_fn = PoeticFormReward(Haiku(), include_rubric=False)
        result = reward_fn(BASHO_HAIKU)
        assert result.rubric == ""

    def test_custom_name(self) -> None:
        """Can set custom name."""
        reward_fn = PoeticFormReward(Haiku(), name="Custom Haiku Check")
        assert reward_fn.name == "Custom Haiku Check"

    def test_verify_method(self) -> None:
        """verify() returns full VerificationResult."""
        from abide.constraints import VerificationResult

        reward_fn = PoeticFormReward(Haiku())
        result = reward_fn.verify(BASHO_HAIKU)
        assert isinstance(result, VerificationResult)

    def test_describe_method(self) -> None:
        """describe() returns constraint description."""
        reward_fn = PoeticFormReward(Haiku())
        desc = reward_fn.describe()
        assert "Haiku" in desc

    def test_to_dict(self) -> None:
        """RewardOutput can be converted to dict."""
        reward_fn = PoeticFormReward(Haiku())
        result = reward_fn(BASHO_HAIKU)
        d = result.to_dict()
        assert "score" in d
        assert "passed" in d
        assert "rubric" in d
        assert "metadata" in d


class TestMakeRewardFunction:
    """Tests for make_reward_function factory."""

    def test_creates_callable(self) -> None:
        """Factory creates callable."""
        reward_fn = make_reward_function(Haiku())
        assert callable(reward_fn)

    def test_passes_kwargs(self) -> None:
        """Factory passes kwargs to PoeticFormReward."""
        reward_fn = make_reward_function(Haiku(), name="Test Name")
        assert reward_fn.name == "Test Name"


class TestAbideMajorPoeticForms:
    """Tests for AbideMajorPoeticForms eval suite."""

    def test_contains_all_forms(self) -> None:
        """Eval contains all major forms."""
        eval_suite = AbideMajorPoeticForms()
        forms = eval_suite.forms
        assert "haiku" in forms
        assert "tanka" in forms
        assert "villanelle" in forms
        assert "sestina" in forms
        assert "shakespearean_sonnet" in forms
        assert "petrarchan_sonnet" in forms
        assert "spenserian_sonnet" in forms
        assert "limerick" in forms

    def test_len(self) -> None:
        """Eval has correct number of forms."""
        eval_suite = AbideMajorPoeticForms()
        assert len(eval_suite) == 8

    def test_iter(self) -> None:
        """Can iterate over eval items."""
        eval_suite = AbideMajorPoeticForms()
        items = list(eval_suite)
        assert len(items) == 8

    def test_evaluate_haiku(self) -> None:
        """Evaluate identifies haiku correctly."""
        eval_suite = AbideMajorPoeticForms()
        result = eval_suite.evaluate(BASHO_HAIKU)
        # Haiku should be best match for actual haiku
        assert result.best_match == "haiku"
        assert result.best_score > 0.5

    def test_evaluate_villanelle(self) -> None:
        """Evaluate identifies villanelle correctly."""
        eval_suite = AbideMajorPoeticForms()
        result = eval_suite.evaluate(DYLAN_THOMAS_VILLANELLE)
        # Villanelle should be best match for actual villanelle
        assert result.best_match == "villanelle"
        assert result.best_score > 0.5

    def test_evaluate_returns_all_scores(self) -> None:
        """Evaluate returns scores for all forms."""
        eval_suite = AbideMajorPoeticForms()
        result = eval_suite.evaluate(BASHO_HAIKU)
        assert len(result.results) == 8
        for form in eval_suite.forms:
            assert form in result.results

    def test_evaluate_aggregate_score(self) -> None:
        """Evaluate computes aggregate score."""
        eval_suite = AbideMajorPoeticForms()
        result = eval_suite.evaluate(BASHO_HAIKU)
        assert 0 <= result.aggregate_score <= 1.0

    def test_get_reward_function(self) -> None:
        """Can get reward function for specific form."""
        eval_suite = AbideMajorPoeticForms()
        reward_fn = eval_suite.get_reward_function("haiku")
        assert isinstance(reward_fn, PoeticFormReward)
        assert reward_fn.name == "haiku"

    def test_get_reward_function_unknown(self) -> None:
        """Unknown form raises ValueError."""
        eval_suite = AbideMajorPoeticForms()
        with pytest.raises(ValueError, match="Unknown form"):
            eval_suite.get_reward_function("unknown_form")

    def test_get_all_reward_functions(self) -> None:
        """Can get all reward functions."""
        eval_suite = AbideMajorPoeticForms()
        reward_fns = eval_suite.get_all_reward_functions()
        assert len(reward_fns) == 8
        assert all(isinstance(fn, PoeticFormReward) for fn in reward_fns.values())

    def test_describe(self) -> None:
        """Describe returns informative string."""
        eval_suite = AbideMajorPoeticForms()
        desc = eval_suite.describe()
        assert "Abide Major Poetic Forms" in desc
        assert "haiku" in desc
        assert "villanelle" in desc

    def test_strict_mode(self) -> None:
        """Strict mode can be enabled."""
        eval_suite = AbideMajorPoeticForms(strict=True)
        # Should still work, just stricter
        result = eval_suite.evaluate(BASHO_HAIKU)
        assert 0 <= result.best_score <= 1.0


class TestMakePoeticFormsEval:
    """Tests for make_poetic_forms_eval factory."""

    def test_creates_full_eval(self) -> None:
        """Factory creates full eval by default."""
        eval_suite = make_poetic_forms_eval()
        assert len(eval_suite) == 8

    def test_filter_forms(self) -> None:
        """Can filter to specific forms."""
        eval_suite = make_poetic_forms_eval(forms=["haiku", "tanka"])
        assert len(eval_suite) == 2
        assert "haiku" in eval_suite.forms
        assert "tanka" in eval_suite.forms

    def test_passes_kwargs(self) -> None:
        """Factory passes kwargs."""
        eval_suite = make_poetic_forms_eval(strict=True)
        assert eval_suite.strict is True


class TestEvalResult:
    """Tests for EvalResult data structure."""

    def test_contains_poem(self) -> None:
        """Result contains original poem."""
        eval_suite = AbideMajorPoeticForms()
        result = eval_suite.evaluate(BASHO_HAIKU)
        assert result.poem == BASHO_HAIKU

    def test_contains_metadata(self) -> None:
        """Result contains metadata."""
        eval_suite = AbideMajorPoeticForms()
        result = eval_suite.evaluate(BASHO_HAIKU)
        assert "all_scores" in result.metadata
        assert "form_count" in result.metadata
        assert result.metadata["form_count"] == 8


class TestIntegration:
    """Integration tests for verifiers framework."""

    def test_reward_function_workflow(self) -> None:
        """Complete workflow: create reward function, evaluate, get score."""
        # 1. Create eval suite
        eval_suite = AbideMajorPoeticForms()

        # 2. Get reward function for specific form
        haiku_reward = eval_suite.get_reward_function("haiku")

        # 3. Evaluate poem
        result = haiku_reward(BASHO_HAIKU)

        # 4. Get score for RL training
        score = result.score
        assert 0 <= score <= 1.0

        # 5. Get rubric for debugging
        rubric = result.rubric
        assert isinstance(rubric, str)

    def test_multi_form_comparison(self) -> None:
        """Compare poem against multiple forms."""
        eval_suite = AbideMajorPoeticForms()

        # Haiku should match haiku better than villanelle
        haiku_eval = eval_suite.evaluate(BASHO_HAIKU)
        assert haiku_eval.results["haiku"].score > haiku_eval.results["villanelle"].score

        # Villanelle should match villanelle better than haiku
        villanelle_eval = eval_suite.evaluate(DYLAN_THOMAS_VILLANELLE)
        assert villanelle_eval.results["villanelle"].score > villanelle_eval.results["haiku"].score
