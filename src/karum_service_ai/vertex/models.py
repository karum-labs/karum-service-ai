"""Pydantic response schemas for the AgentService RPCs (PLAN.md §15).

One Pydantic model per RPC response so invalid model JSON fails closed. The
field names mirror ``karum/agent/v1/agent.proto``; JSON serialises as camelCase
(``to_camel``) matching the proto JSON mapping, which keeps these models
compatible with the contract test fixtures in
``karum-infra/contract/testdata/campaign``.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class ErrorCode(StrEnum):
    """Mirror of ``karum.common.v1.ErrorCode`` with the proto prefix off."""

    UNSPECIFIED = "ERROR_CODE_UNSPECIFIED"
    UNRESOLVED = "ERROR_CODE_UNRESOLVED"
    EMPTY = "ERROR_CODE_EMPTY"
    RATE_LIMITED = "ERROR_CODE_RATE_LIMITED"
    UPSTREAM = "ERROR_CODE_UPSTREAM"
    VALIDATION = "ERROR_CODE_VALIDATION"


class ModelAttribution(_Base):
    """Carried on every AgentService response (PLAN.md §3).

    ``ungrounded`` is true only for the LlmOnlyBaseline counterfactual, which by
    construction cites no Qloo evidence.
    """

    model_id: str
    prompt_version: str
    evidence_ids: list[str] = Field(default_factory=list)
    ungrounded: bool = False


def service_attribution(
    model_id: str,
    prompt_version: str,
    *,
    evidence_ids: Sequence[str] = (),
    ungrounded: bool = False,
) -> ModelAttribution:
    """The provenance the service knows, never the model's self-report.

    The model writes its own ``attribution`` block, and a model that invents a
    model id or a prompt version must never reach the gateway: the service
    resolves the model name from its own configuration and the prompt version
    from its own constants, and says so on every path (PLAN.md §3).
    """
    return ModelAttribution(
        model_id=model_id,
        prompt_version=prompt_version,
        evidence_ids=list(evidence_ids),
        ungrounded=ungrounded,
    )


class EntityRef(_Base):
    """One resolved entity: a venue, brand, or artist row."""

    id: str
    type: str
    name: str
    city: str


class GroundingSource(_Base):
    """One citation returned by Search grounding; a citation, never a fetch target (PLAN.md §42)."""

    url: str
    title: str


class Incumbent(_Base):
    """A competitor the audit found on the live menu or shelf."""

    name: str
    sources: list[GroundingSource] = Field(default_factory=list)


class MenuAuditResult(_Base):
    """One venue's shelf read."""

    door: EntityRef
    succeeded: bool
    shelf_unknown: bool
    incumbents: list[Incumbent] = Field(default_factory=list)
    competitor_count: int = 0
    shelf_opportunity: float = 0.0
    menu_summary: str = ""
    sources: list[GroundingSource] = Field(default_factory=list)


class AuditMenuResponse(_Base):
    results: list[MenuAuditResult]
    attribution: ModelAttribution
    error_code: ErrorCode = ErrorCode.UNSPECIFIED
    error_message: str = ""


class Tag(_Base):
    """One verified tag URN (the namespace allow-list limits what the pitch explains)."""

    urn: str
    label: str


class SubScore(_Base):
    """One 0-1 term of the door composite, inputs written beside it (PLAN.md §11)."""

    name: str
    value: float
    raw_value: float
    inputs: list[str] = Field(default_factory=list)


class Sku(_Base):
    """One entry of the brand's own product list."""

    id: str
    name: str


class Battlecard(_Base):
    """The displacement card, present only when the audit found an incumbent."""

    headline: str
    incumbent_name: str
    talking_points: list[str] = Field(default_factory=list)
    compare_summary: str = ""


class Citation(_Base):
    """One machine-readable citation: an evidence id plus citation text.

    Prose never carries ids inline; citations render as a separate source list
    at the bottom of the pitch (PLAN.md §64).
    """

    evidence_id: str
    text: str


class BrandContext(_Base):
    """The brand side of a draft request."""

    name: str
    skus: list[Sku] = Field(default_factory=list)


class DoorContext(_Base):
    """The Qloo-derived context a pitch draws on."""

    shared_tags: list[Tag] = Field(default_factory=list)
    affinity: SubScore
    compare_tags: list[Tag] = Field(default_factory=list)
    audit: MenuAuditResult


class DraftDoorPitchResponse(_Base):
    """One door's pitch: six citation-free sections plus a source list.

    ``script`` is the combined, citation-free renderable prose kept for
    backward compatibility; the six sections carry the human-readable text and
    ``sources`` carries the machine-readable citations (PLAN.md §64).
    """

    script: str = ""
    battlecard: Battlecard | None = None
    recommended_sku: Sku = Field(default_factory=lambda: Sku(id="", name=""))
    mode_paragraph: str = ""
    opening: str = ""
    why_this_door: str = ""
    pitch: str = ""
    product_to_lead: str = ""
    ask: str = ""
    close: str = ""
    sources: list[Citation] = Field(default_factory=list)
    attribution: ModelAttribution | None = None
    error_code: ErrorCode = ErrorCode.UNSPECIFIED
    error_message: str = ""


class LlmOnlyBaselineResponse(_Base):
    """The generic-LLM contrast stored beside the Qloo result, labeled ungrounded (PLAN.md §17)."""

    text: str
    venue_names: list[str] = Field(default_factory=list)
    attribution: ModelAttribution
    error_code: ErrorCode = ErrorCode.UNSPECIFIED
    error_message: str = ""
