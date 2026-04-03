"""Integration tests for prompt-generator form-set selection."""

from scripts import prompt_generator


def test_resolve_form_selection_mode_defaults_to_rl_default(monkeypatch) -> None:
    monkeypatch.delenv("ABIDE_FORM_SET", raising=False)
    monkeypatch.delenv("ABIDE_LEARNABLE", raising=False)
    monkeypatch.delenv("ABIDE_TRADITIONAL", raising=False)

    assert prompt_generator.resolve_form_selection_mode() == "rl_default"


def test_resolve_form_selection_mode_honors_explicit_form_set(monkeypatch) -> None:
    monkeypatch.setenv("ABIDE_FORM_SET", "all")
    monkeypatch.setenv("ABIDE_LEARNABLE", "1")

    assert prompt_generator.resolve_form_selection_mode() == "all"


def test_resolve_form_selection_mode_accepts_well_known(monkeypatch) -> None:
    monkeypatch.setenv("ABIDE_FORM_SET", "well_known")
    monkeypatch.delenv("ABIDE_LEARNABLE", raising=False)
    monkeypatch.delenv("ABIDE_TRADITIONAL", raising=False)

    assert prompt_generator.resolve_form_selection_mode() == "well_known"
