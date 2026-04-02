"""Docstring claim-alignment checks for structural form verifiers."""

from __future__ import annotations

from inspect import getdoc

from abide.forms import Ballad, Clerihew, Ghazal, Haiku, Tanaga


def test_clerihew_docstring_no_longer_claims_humor_or_biography_detection() -> None:
    doc = getdoc(Clerihew)
    assert doc is not None
    lowered = doc.lower()
    assert "humorous" not in lowered
    assert "biographical" not in lowered


def test_ballad_docstring_no_longer_claims_narrative_detection() -> None:
    doc = getdoc(Ballad)
    assert doc is not None
    assert "narrative content" not in doc.lower()


def test_representative_form_docstrings_drop_unverified_theme_promises() -> None:
    tanaga_doc = getdoc(Tanaga)
    haiku_doc = getdoc(Haiku)
    ghazal_doc = getdoc(Ghazal)

    assert tanaga_doc is not None
    assert haiku_doc is not None
    assert ghazal_doc is not None

    assert "social commentary" not in tanaga_doc.lower()
    assert "moment in nature" not in haiku_doc.lower()
    assert "thematically self-contained" not in ghazal_doc.lower()
