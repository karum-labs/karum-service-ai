"""Model transports: the live Gemini call and the recorded replay (PLAN.md §18).

A transport turns one model request into one structured response. The live
transport is the Google GenAI client the :class:`~karum_service_ai.vertex.client.VertexClient`
builds; the recorded transport
(:class:`RecordedModelTransport`) answers from the fixture pack with no network
and no credential.

Transports stay thin: retries, schema validation, and error mapping belong to
the client, so a fake transport is trivial to write in tests.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from karum_service_ai.vertex.errors import FixtureError
from karum_service_ai.vertex.fixtures import FixturePack
from karum_service_ai.vertex.models import (
    AuditMenuResponse,
    DraftDoorPitchResponse,
    LlmOnlyBaselineResponse,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

#: The RPC each response schema belongs to. The pack is keyed by this method
#: name, so a replayed answer always lands on the RPC that asked for it.
SCHEMA_METHODS: dict[type, str] = {
    AuditMenuResponse: "audit_menu",
    DraftDoorPitchResponse: "draft_door_pitch",
    LlmOnlyBaselineResponse: "llm_only_baseline",
}


def summarize(method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The summary fields a recording's ``match`` spec is checked against.

    The summary names the request in the terms a recording can speak about: the
    city, the brand, and the door. The ids the run issues (door ids, evidence
    ids) are deliberately absent — they differ per run and would make every
    replay a miss.
    """
    if method == "audit_menu":
        targets = payload.get("targets") or []
        door = (targets[0].get("door") or {}) if targets else {}
        return {"city": door.get("city", ""), "door": door.get("name", "")}
    brand = payload.get("brand") or {}
    summary = {"city": payload.get("city", ""), "brand": brand.get("name", "")}
    if method == "draft_door_pitch":
        door = payload.get("door") or {}
        audit = door.get("audit") or {}
        summary["door"] = (audit.get("door") or {}).get("name", "")
    return summary


class RecordedModelTransport:
    """Serves the recorded pack with no network; a miss is an error, not a guess."""

    name = "fixture"

    def __init__(self, pack: FixturePack) -> None:
        self._pack = pack

    @property
    def pack(self) -> FixturePack:
        """The underlying pack, for health reporting and tests."""
        return self._pack

    def generate(self, schema: type[T], payload: Mapping[str, Any] | None = None) -> T:
        """Answer one request from the pack, validated against ``schema``."""
        method = SCHEMA_METHODS.get(schema)
        if method is None:
            raise FixtureError(
                f"no recorded method for response schema {getattr(schema, '__name__', schema)}"
            )
        body = dict(payload or {})
        entry = self._pack.match(method, body, summarize(method, body))
        try:
            return schema.model_validate(entry.response)
        except ValidationError as exc:
            raise FixtureError(
                f"recorded response {entry.id} does not fit "
                f"{getattr(schema, '__name__', schema)}: {exc}"
            ) from exc
