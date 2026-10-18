"""Menu audit tests (PLAN.md §16): grounding, incumbents, shelf inputs, sources."""

from __future__ import annotations

from tests.conftest import make_client

from karum_service_ai.agent.audit import audit_menu
from karum_service_ai.vertex.models import ErrorCode

AUDIT_JSON = """
{
  "results": [
    {
      "door": {"id": "1F0E9D8C-7B6A-4593-8C2D-1E0F2A3B4C10", "type": "urn:entity:place", "name": "Whisler's", "city": "Austin"},
      "succeeded": true,
      "shelfUnknown": false,
      "incumbents": [{"name": "Del Maguey Single Village Mezcal", "sources": [{"url": "https://www.whislersaustin.com/menu", "title": "Whisler's — Menu"}]}],
      "competitorCount": 1,
      "shelfOpportunity": 0.75,
      "menuSummary": "Agave-forward cocktail menu; Del Maguey and Montelobos anchor the mezcal shelf.",
      "sources": [{"url": "https://www.whislersaustin.com/menu", "title": "Whisler's — Menu"}]
    },
    {
      "door": {"id": "2A1B0C3D-4E5F-4A6B-9C8D-7E6F5A4B3C20", "type": "urn:entity:place", "name": "Here Nor There", "city": "Austin"},
      "succeeded": true,
      "shelfUnknown": true,
      "incumbents": [],
      "competitorCount": 0,
      "shelfOpportunity": 0.0,
      "menuSummary": "No readable public menu (social pages only). Shelf unknown.",
      "sources": [{"url": "https://www.here-nor-there.com", "title": "Here Nor There"}]
    }
  ],
  "attribution": {"modelId": "gemini-2.5-flash", "promptVersion": "audit-menu@v1", "evidenceIds": ["ev-1"], "ungrounded": false},
  "errorCode": "ERROR_CODE_UNSPECIFIED"
}
"""

TARGETS = [
    {
        "door": {"id": "1", "type": "urn:entity:place", "name": "Whisler's", "city": "Austin"},
        "website": "https://www.whislersaustin.com/menu",
        "taste_tags": [{"urn": "urn:tag:genre:place:restaurant:bar", "label": "Bars"}],
    },
    {
        "door": {"id": "2", "type": "urn:entity:place", "name": "Here Nor There", "city": "Austin"},
        "website": "https://www.here-nor-there.com",
        "taste_tags": [],
    },
]


def test_audit_menu_parses_grounded_results() -> None:
    """The model returns one MenuAuditResult per target with incumbents and sources."""
    client, fake, _ = make_client(AUDIT_JSON)
    response = audit_menu(client, TARGETS)

    assert len(response.results) == 2
    first, second = response.results
    assert first.succeeded and not first.shelf_unknown
    assert first.incumbents[0].name == "Del Maguey Single Village Mezcal"
    assert first.competitor_count == 1
    assert first.shelf_opportunity == 0.75
    assert response.attribution.model_id == "gemini-2.5-flash"
    assert response.error_code == ErrorCode.UNSPECIFIED


def test_audit_menu_shelf_unknown_keeps_rank() -> None:
    """An unverifiable venue keeps shelf_unknown=true and shelf_opportunity=0.0 so
    the composite renormalises over affinity and demographics."""
    client, fake, _ = make_client(AUDIT_JSON)
    response = audit_menu(client, TARGETS)

    second = response.results[1]
    assert second.shelf_unknown is True
    assert second.shelf_opportunity == 0.0
    assert second.competitor_count == 0
    assert second.incumbents == []


def test_audit_menu_degraded_on_vertex_failure() -> None:
    """When Vertex fails the audit returns a degraded response with error_code UPSTREAM
    and no fabricated menu data."""

    def boom(model: str, contents: str, config: object) -> object:
        raise RuntimeError("Vertex is down")

    client, fake, _ = make_client("{}")
    fake.generate_content = boom  # type: ignore[method-assign]
    response = audit_menu(client, TARGETS)

    assert response.results == []
    assert response.error_code == ErrorCode.UPSTREAM
    assert "Vertex is down" in response.error_message
    assert response.attribution.ungrounded is False


def test_audit_menu_is_capped_to_shortlist() -> None:
    """Grounding runs only over the targets passed in (the working shortlist),
    never the full take-25 pool."""
    client, fake, _ = make_client(AUDIT_JSON)
    audit_menu(client, TARGETS[:1])

    assert len(fake.calls) == 1
    prompt = fake.calls[0][1]
    assert "Whisler's" in prompt
    assert "Here Nor There" not in prompt
