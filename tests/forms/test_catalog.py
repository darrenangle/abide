"""Tests for the form support catalog."""

from abide.forms.catalog import (
    TRAINING_SAFE_FORM_NAMES,
    FormSupportTier,
    get_form_support_tier,
    load_training_safe_form_instances,
)


def test_training_safe_forms_are_marked_training_safe() -> None:
    for form_name in TRAINING_SAFE_FORM_NAMES:
        assert get_form_support_tier(form_name) == FormSupportTier.TRAINING_SAFE


def test_unlisted_form_defaults_to_experimental() -> None:
    assert get_form_support_tier("ThunderVerse") == FormSupportTier.EXPERIMENTAL


def test_load_training_safe_form_instances_returns_exact_catalog_set() -> None:
    forms = load_training_safe_form_instances()
    assert tuple(forms.keys()) == TRAINING_SAFE_FORM_NAMES


def test_training_safe_catalog_uses_tuned_shakespeare_defaults() -> None:
    form = load_training_safe_form_instances()["ShakespeareanSonnet"]
    assert getattr(form, "syllable_tolerance", None) == 2
    assert getattr(form, "rhyme_threshold", None) == 0.4
