"""The AI gRPC adapter returns the baseline the fake model produced."""

from karum.agent.v1 import agent_pb2
from karum_service_ai.core.config import Settings
from karum_service_ai.grpcserv import AgentServicer
from karum_service_ai.vertex.models import ErrorCode, LlmOnlyBaselineResponse, ModelAttribution


class _FakeVertex:
    model_id = "fake"
    grounding_shortlist_cap = 5

    def generate(self, prompt, schema, prompt_version, model_id, grounded=False):
        return LlmOnlyBaselineResponse(
            text="ungrounded advice",
            venue_names=["La Condesa"],
            attribution=ModelAttribution(model_id="fake", prompt_version=prompt_version, ungrounded=True),
            error_code=ErrorCode.UNSPECIFIED,
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
