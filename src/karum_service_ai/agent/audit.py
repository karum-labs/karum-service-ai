"""Menu audit: Google Search-grounded shelf read (PLAN.md §16).

Reads each target venue's website / menu / delivery listing with Search grounding
and reports whether the audit succeeded, which incumbents are stocked, a short
menu summary, the ``O_shelf`` inputs, and the source URLs.

Grounding URLs are citations returned by the model — they are never fetch
targets (PLAN.md §42). Grounding is capped to the working shortlist: the caller
passes only the doors that can appear on the board plus any skip candidates the
skip rule might cite, never the full ``take=25`` pool.

Online-only or unverifiable venues keep ``shelf_unknown=True`` and
``shelf_opportunity=0.0`` so the composite renormalises over affinity and
demographics (PLAN.md §16).
"""

from __future__ import annotations

import logging
from typing import Any

from karum_service_ai.vertex.client import VertexClient, VertexError
from karum_service_ai.vertex.models import (
    AuditMenuResponse,
    ErrorCode,
    ModelAttribution,
)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_VERSION = "audit-menu@v1"


def _build_audit_prompt(targets: list[dict[str, Any]]) -> str:
    lines = [
        "You are Karum's menu auditor. For each target venue below, use Google Search grounding "
        "to read its website, menu, or delivery listing and return a JSON AuditMenuResponse.",
        "Grounding rules:",
        "  - Name an incumbent competitor ONLY when a grounding source or the Qloo payload contains it.",
        "  - Grounding URLs are citations; never fetch them.",
        "  - If no readable public menu exists (online-only or unverifiable), set succeeded=true, "
        "    shelf_unknown=true, shelf_opportunity=0.0, and an empty incumbents list.",
        "  - shelf_opportunity is 0..1: 1.0 with no incumbent, each direct niche competitor takes a quarter off.",
        "  - Align results to the targets in order.",
        "",
        "Targets:",
    ]
    for index, target in enumerate(targets, start=1):
        door = target.get("door", {})
        lines.append(
            f"  {index}. {door.get('name', 'unknown')} ({door.get('city', 'unknown')})"
            f" | website={target.get('website', '')}"
            f" | tags={[t.get('urn', '') for t in target.get('taste_tags', [])]}"
        )
    lines.append(
        "\nReturn only the JSON AuditMenuResponse: results (one per target) each with door, "
        "succeeded, shelfUnknown, incumbents[{name, sources[{url, title}]}], competitorCount, "
        "shelfOpportunity, menuSummary, sources[{url, title}], plus attribution{modelId, "
        "promptVersion, evidenceIds, ungrounded:false}, errorCode, errorMessage."
    )
    return "\n".join(lines)


def audit_menu(
    client: VertexClient,
    targets: list[dict[str, Any]],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    model_id: str | None = None,
) -> AuditMenuResponse:
    """Audit the working shortlist. Returns a degraded response on Vertex failure."""
    resolved_model = model_id or client.model_id
    prompt = _build_audit_prompt(targets)
    try:
        response = client.generate(prompt, AuditMenuResponse, prompt_version, resolved_model)
    except VertexError as exc:
        logger.warning(
            "audit_menu.vertex_failed model=%s prompt_version=%s detail=%s",
            resolved_model,
            prompt_version,
            exc.detail,
        )
        return AuditMenuResponse(
            results=[],
            attribution=ModelAttribution(
                model_id=resolved_model,
                prompt_version=prompt_version,
                evidence_ids=[],
                ungrounded=False,
            ),
            error_code=ErrorCode.UPSTREAM,
            error_message=exc.detail,
        )
    return response
