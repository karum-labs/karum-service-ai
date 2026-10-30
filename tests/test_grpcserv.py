"""The AI gRPC adapter (PLAN.md §15, §16, §17).

The adapter is where the contract is either kept or broken: the run's evidence
set must reach the citation filter, the Qloo payload's names must reach the
incumbent check, and every response must carry its attribution (PLAN.md §3).
"""

from karum.agent.v1 import agent_pb2
from karum.common.v1 import common_pb2
from karum_service_ai.core.config import Settings
from karum_service_ai.grpcserv import AgentServicer
from karum_service_ai.vertex.models import (
    AuditMenuResponse,
    DraftDoorPitchResponse,
    ErrorCode,
    LlmOnlyBaselineResponse,
    ModelAttribution,
    Sku,
)

AUDIT_JSON = """
{
  "results": [
    {
      "door": {"id": "d-1", "type": "urn:entity:place", "name": "Whisler's", "city": "Austin"},
      "succeeded": true,
      "shelfUnknown": false,
      "incumbents": [
        {"name": "Aesop", "sources": []},
        {"name": "Never Heard Of It", "sources": []}
      ],
      "competitorCount": 2,
      "shelfOpportunity": 0.5,
      "menuSummary": "Aesop on the back bar.",
      "sources": []
    }
  ],
  "attribution": {"modelId": "gemini-3.8-flash", "promptVersion": "menu-audit@v1", "evidenceIds": ["ev-0002"], "ungrounded": false},
  "errorCode": "ERROR_CODE_UNSPECIFIED"
}
"""

SOURCED_AUDIT_JSON = """
{
  "results": [
    {
      "door": {"id": "d-1", "type": "urn:entity:place", "name": "Whisler's", "city": "Austin"},
      "succeeded": true,
      "shelfUnknown": false,
      "incumbents": [
        {"name": "Aesop", "sources": [{"url": "https://example.com/menu", "title": "Aesop on the Whisler's menu"}]}
      ],
      "competitorCount": 1,
      "shelfOpportunity": 0.75,
      "menuSummary": "Aesop on the back bar.",
      "sources": []
    }
  ],
  "attribution": {"modelId": "gemini-3.8-flash", "promptVersion": "menu-audit@v1", "evidenceIds": [], "ungrounded": false},
  "errorCode": "ERROR_CODE_UNSPECIFIED"
}
"""


class _FakeVertex:
    """A fake model that answers the audit, the draft, and the baseline RPCs."""

    model_id = "fake"
    grounding_shortlist_cap = 5

    def __init__(
        self,
        audit_json: str | None = None,
        script: str = "Whisler's ranks first of 25 (affinity 0.87; ev-0001).",
    ) -> None:
        self._audit_json = audit_json
        self._script = script

    def generate(
        self, prompt, schema, prompt_version, model_id, grounded=False, request_payload=None
    ):
        if schema is AuditMenuResponse:
            return AuditMenuResponse.model_validate_json(self._audit_json)
        if schema is LlmOnlyBaselineResponse:
            return LlmOnlyBaselineResponse(
                text="ungrounded advice",
                venue_names=["La Condesa"],
                attribution=ModelAttribution(
                    model_id="fake", prompt_version=prompt_version, ungrounded=True
                ),
                error_code=ErrorCode.UNSPECIFIED,
            )
        return DraftDoorPitchResponse(
            script=self._script,
            battlecard=None,
            recommended_sku=Sku(id="sk-ilegal", name="Ilegal Mezcal Espadin Joven"),
            mode_paragraph="",
            attribution=ModelAttribution(
                model_id="fake", prompt_version="draft-door-pitch@v1", ungrounded=False
            ),
            error_code=ErrorCode.UNSPECIFIED,
        )


def _door_context() -> agent_pb2.DoorContext:
    return agent_pb2.DoorContext(
        shared_tags=[common_pb2.Tag()],
        affinity=common_pb2.SubScore(name="cultural_affinity", value=1.0, raw_value=0.87),
        audit=agent_pb2.MenuAuditResult(
            door=common_pb2.EntityRef(id="d-1", name="Whisler's"),
            succeeded=True,
        ),
    )


def _draft_request(evidence_ids: list[str]) -> agent_pb2.DraftDoorPitchRequest:
    return agent_pb2.DraftDoorPitchRequest(
        mode=agent_pb2.CAMPAIGN_MODE_WHOLESALE,
        brand=agent_pb2.BrandContext(name="Ilegal Mezcal"),
        door=_door_context(),
        city="Austin",
        evidence_ids=evidence_ids,
    )


def test_baseline_rpc_returns_the_venue_names():
    servicer = AgentServicer(Settings(), client=_FakeVertex())
    response = servicer.LlmOnlyBaseline(
        agent_pb2.LlmOnlyBaselineRequest(
            brand=agent_pb2.BrandContext(name="Sola"),
            city="Austin",
            mode=agent_pb2.CAMPAIGN_MODE_WHOLESALE,
        ),
        None,
    )
    assert response.text == "ungrounded advice"
    assert list(response.venue_names) == ["La Condesa"]


def test_baseline_rpc_carries_its_attribution():
    """Every response carries the model id, the prompt version, and the ids."""
    servicer = AgentServicer(Settings(), client=_FakeVertex())
    response = servicer.LlmOnlyBaseline(
        agent_pb2.LlmOnlyBaselineRequest(
            brand=agent_pb2.BrandContext(name="Sola"),
            city="Austin",
            mode=agent_pb2.CAMPAIGN_MODE_WHOLESALE,
        ),
        None,
    )
    assert response.attribution.model_id == "fake"
    assert response.attribution.prompt_version == "llm-only-baseline@v1"
    assert response.attribution.ungrounded is True
    assert list(response.attribution.evidence_ids) == []


def test_draft_rpc_keeps_a_cited_id_inside_the_run_evidence_set():
    """A cited id the run's evidence set carries survives the filter."""
    servicer = AgentServicer(Settings(), client=_FakeVertex())
    response = servicer.DraftDoorPitch(_draft_request(["ev-0001", "ev-0002"]), None)
    assert "ev-" not in response.script
    assert list(response.attribution.evidence_ids) == ["ev-0001"]
    assert response.attribution.model_id == "fake"
    assert response.attribution.prompt_version == "draft-door-pitch@v1"


def test_draft_rpc_drops_a_cited_id_outside_the_run_evidence_set():
    """An invented id is dropped, and an emptied script fails closed."""
    servicer = AgentServicer(
        Settings(),
        client=_FakeVertex(script="Whisler's ranks first of 25 (affinity 0.87; ev-9999)."),
    )
    response = servicer.DraftDoorPitch(_draft_request(["ev-0001", "ev-0002"]), None)
    assert response.script == ""
    assert response.error_code == common_pb2.ERROR_CODE_VALIDATION


def test_draft_rpc_with_an_empty_evidence_set_drops_every_number():
    """An empty allow-set is strict: nothing can be cited, so nothing stays."""
    servicer = AgentServicer(Settings(), client=_FakeVertex())
    response = servicer.DraftDoorPitch(_draft_request([]), None)
    assert response.script == ""
    assert response.error_code == common_pb2.ERROR_CODE_VALIDATION


def test_audit_rpc_keeps_a_competitor_the_qloo_payload_names():
    """A competitor is kept only when a source or the Qloo payload names it."""
    servicer = AgentServicer(Settings(), client=_FakeVertex(audit_json=AUDIT_JSON))
    response = servicer.AuditMenu(
        agent_pb2.AuditMenuRequest(
            mode=agent_pb2.CAMPAIGN_MODE_WHOLESALE,
            targets=[
                agent_pb2.MenuAuditTarget(
                    door=common_pb2.EntityRef(id="d-1", name="Whisler's"),
                    qloo_incumbent_names=["Aesop"],
                )
            ],
        ),
        None,
    )
    names = [item.name for item in response.results[0].incumbents]
    assert names == ["Aesop"]
    assert response.results[0].competitor_count == 1
    # The service resolves the provenance, not the model's self-report (K-08):
    # the audit JSON claims "gemini-3.8-flash", the client is the fake.
    assert response.attribution.model_id == "fake"
    assert list(response.attribution.evidence_ids) == []


def test_audit_rpc_drops_a_competitor_the_qloo_payload_does_not_name():
    """Without the Qloo name, an unsourced competitor is dropped."""
    servicer = AgentServicer(Settings(), client=_FakeVertex(audit_json=AUDIT_JSON))
    response = servicer.AuditMenu(
        agent_pb2.AuditMenuRequest(
            mode=agent_pb2.CAMPAIGN_MODE_WHOLESALE,
            targets=[
                agent_pb2.MenuAuditTarget(
                    door=common_pb2.EntityRef(id="d-1", name="Whisler's"),
                )
            ],
        ),
        None,
    )
    assert [item.name for item in response.results[0].incumbents] == []
    assert response.results[0].competitor_count == 0


def test_audit_rpc_keeps_a_competitor_a_grounding_source_carries():
    """A grounding source is enough on its own."""
    servicer = AgentServicer(Settings(), client=_FakeVertex(audit_json=SOURCED_AUDIT_JSON))
    response = servicer.AuditMenu(
        agent_pb2.AuditMenuRequest(
            mode=agent_pb2.CAMPAIGN_MODE_WHOLESALE,
            targets=[
                agent_pb2.MenuAuditTarget(
                    door=common_pb2.EntityRef(id="d-1", name="Whisler's"),
                )
            ],
        ),
        None,
    )
    assert [item.name for item in response.results[0].incumbents] == ["Aesop"]
    assert response.results[0].competitor_count == 1
