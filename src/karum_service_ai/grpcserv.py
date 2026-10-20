"""gRPC servicer for AgentService.

HTTP ``/health`` stays on the FastAPI port. The gateway dials this server for
the menu audit, the door pitch, and the ungrounded baseline.
"""

from __future__ import annotations

from concurrent import futures

import grpc

from karum.agent.v1 import agent_pb2, agent_pb2_grpc
from karum.common.v1 import common_pb2
from karum_service_ai.agent.audit import audit_menu
from karum_service_ai.agent.draft import draft_door_pitch, llm_only_baseline
from karum_service_ai.core.config import Settings
from karum_service_ai.vertex.client import VertexClient
from karum_service_ai.vertex.models import (
    BrandContext,
    DoorContext,
    EntityRef,
    ErrorCode,
    MenuAuditResult,
    Sku,
    SubScore,
    Tag,
)

_MODES = {
    agent_pb2.CAMPAIGN_MODE_FLAGSHIP: "flagship",
    agent_pb2.CAMPAIGN_MODE_COLLAB: "collab",
}


def _mode(value: int) -> str:
    return _MODES.get(value, "wholesale")


def _code(code: ErrorCode) -> int:
    return common_pb2.ErrorCode.Value(code.value)


def _entity(ref) -> common_pb2.EntityRef:
    if ref is None:
        return common_pb2.EntityRef()
    return common_pb2.EntityRef(
        id=ref.id, type=ref.type, name=ref.name, city=getattr(ref, "city", "") or ""
    )


def _entity_in(ref) -> EntityRef:
    return EntityRef(
        id=ref.id or "",
        type=ref.type or "urn:entity:place",
        name=ref.name or "",
        city=ref.city or "",
    )


def _sources(items) -> list:
    from karum_service_ai.vertex.models import GroundingSource

    return [GroundingSource(url=item.url, title=item.title) for item in items]


def _audit_in(result) -> MenuAuditResult:
    from karum_service_ai.vertex.models import Incumbent

    return MenuAuditResult(
        door=_entity_in(result.door)
        if result.HasField("door")
        else EntityRef(id="", type="", name="", city=""),
        succeeded=result.succeeded,
        shelf_unknown=result.shelf_unknown,
        incumbents=[
            Incumbent(name=item.name, sources=_sources(item.sources)) for item in result.incumbents
        ],
        competitor_count=result.competitor_count,
        shelf_opportunity=result.shelf_opportunity,
        menu_summary=result.menu_summary,
        sources=_sources(result.sources),
    )


def _audit_out(result: MenuAuditResult) -> agent_pb2.MenuAuditResult:
    from karum.agent.v1 import agent_pb2 as pb

    return pb.MenuAuditResult(
        door=_entity(result.door),
        succeeded=result.succeeded,
        shelf_unknown=result.shelf_unknown,
        incumbents=[
            pb.Incumbent(
                name=item.name,
                sources=[
                    pb.GroundingSource(url=source.url, title=source.title)
                    for source in item.sources
                ],
            )
            for item in result.incumbents
        ],
        competitor_count=result.competitor_count,
        shelf_opportunity=result.shelf_opportunity,
        menu_summary=result.menu_summary,
        sources=[
            pb.GroundingSource(url=source.url, title=source.title) for source in result.sources
        ],
    )


class AgentServicer(agent_pb2_grpc.AgentServiceServicer):
    """Adapts the in-process draft and audit functions onto the gRPC contract."""

    def __init__(self, settings: Settings, client: VertexClient | None = None) -> None:
        self.settings = settings
        self._client = client

    def client(self) -> VertexClient:
        if self._client is None:
            self._client = VertexClient(self.settings)
        return self._client

    def AuditMenu(self, request, context):  # noqa: N802
        targets = []
        for target in request.targets:
            door = target.door
            targets.append(
                {
                    "door": {
                        "id": door.id,
                        "type": door.type,
                        "name": door.name,
                        "city": door.city,
                    },
                    "website": target.website,
                    "taste_tags": [
                        {"urn": tag.urn, "label": tag.label} for tag in target.taste_tags
                    ],
                    "skip_candidate": target.skip_candidate,
                }
            )
        result = audit_menu(self.client(), targets)
        return agent_pb2.AuditMenuResponse(
            results=[_audit_out(item) for item in result.results],
            error_code=_code(result.error_code),
            error_message=result.error_message,
        )

    def DraftDoorPitch(self, request, context):  # noqa: N802
        brand = BrandContext(
            name=request.brand.name,
            skus=[Sku(id=sku.id, name=sku.name) for sku in request.brand.skus],
        )
        audit = _audit_in(request.door.audit)
        door = DoorContext(
            shared_tags=[Tag(urn=tag.urn, label=tag.label) for tag in request.door.shared_tags],
            affinity=SubScore(
                name=request.door.affinity.name or "cultural_affinity",
                value=request.door.affinity.value,
                raw_value=request.door.affinity.raw_value,
                inputs=list(request.door.affinity.inputs),
            ),
            compare_tags=[Tag(urn=tag.urn, label=tag.label) for tag in request.door.compare_tags],
            audit=audit,
        )
        result = draft_door_pitch(
            self.client(),
            brand,
            door,
            request.city,
            mode=_mode(request.mode),
            max_words=request.max_words or 220,
        )
        card = None
        if result.battlecard is not None:
            card = agent_pb2.Battlecard(
                headline=result.battlecard.headline,
                incumbent_name=result.battlecard.incumbent_name,
                talking_points=list(result.battlecard.talking_points),
                compare_summary=result.battlecard.compare_summary,
            )
        return agent_pb2.DraftDoorPitchResponse(
            script=result.script,
            battlecard=card,
            recommended_sku=agent_pb2.Sku(
                id=result.recommended_sku.id, name=result.recommended_sku.name
            ),
            mode_paragraph=result.mode_paragraph,
            error_code=_code(result.error_code),
            error_message=result.error_message,
        )

    def LlmOnlyBaseline(self, request, context):  # noqa: N802
        brand = BrandContext(
            name=request.brand.name,
            skus=[Sku(id=sku.id, name=sku.name) for sku in request.brand.skus],
        )
        result = llm_only_baseline(
            self.client(),
            brand,
            request.city,
            mode=_mode(request.mode),
            max_words=request.max_words or 160,
        )
        return agent_pb2.LlmOnlyBaselineResponse(
            text=result.text,
            venue_names=list(result.venue_names),
            error_code=_code(result.error_code),
            error_message=result.error_message,
        )


def serve(settings: Settings, port: int) -> grpc.Server:
    """Start the AI gRPC server on ``port`` and return it still running."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentServicer(settings), server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    return server
