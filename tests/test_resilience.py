"""Draft resilience tests (PLAN.md §18): template sentences interpolate evidence
only, sentences with missing cited ids are dropped, and a failed draft degrades
without taking down the door list."""

from __future__ import annotations

from karum_service_ai.agent.draft import render_sentence


def test_render_sentence_interpolates_evidence_fields_only() -> None:
    """A sentence whose cited evidence ids are all present is interpolated."""
    evidence = {"ev-1": "first of 25", "ev-4": "Bars and cocktail culture"}
    assert (
        render_sentence("Whisler's ranks {ev-1} for this brand ({ev-4}).", evidence)
        == "Whisler's ranks first of 25 for this brand (Bars and cocktail culture)."
    )


def test_render_sentence_drops_when_cited_id_missing() -> None:
    """A sentence whose cited evidence id is missing is dropped, never fabricated."""
    evidence = {"ev-1": "first of 25"}
    assert render_sentence("Whisler's ranks {ev-1} ({ev-99}).", evidence) is None


def test_render_sentence_no_citations_passes_through() -> None:
    """A sentence with no evidence citations is returned unchanged."""
    assert render_sentence("A plain sentence with no ids.", {}) == "A plain sentence with no ids."
