"""Claim-alignment tests for intentionally generic structural forms."""

from __future__ import annotations

from abide.forms import Aubade, Epigram, IrregularOde, Ode


def test_ode_descriptions_do_not_claim_unverified_lyric_semantics() -> None:
    assert "lyric" not in Ode().describe().lower()
    assert "structural shell" in Ode().describe().lower()
    assert "structural shell" in IrregularOde().describe().lower()


def test_aubade_description_stays_structural_only() -> None:
    desc = Aubade().describe().lower()
    assert "dawn" not in desc
    assert "structural proxy" in desc


def test_epigram_description_does_not_claim_wit_detection() -> None:
    desc = Epigram().describe().lower()
    assert "witty" not in desc
    assert "pointed" not in desc
    assert "rhyme proxy" in desc
