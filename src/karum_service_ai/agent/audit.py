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
from collections.abc import Mapping, Sequence
from typing import Any

from karum_service_ai.agent.prompting import DATA_RULE, as_data
from karum_service_ai.vertex.client import VertexClient, VertexError
from karum_service_ai.vertex.models import (
    AuditMenuResponse,
    EntityRef,
    ErrorCode,
    MenuAuditResult,
    service_attribution,
)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_VERSION = "audit-menu@v1"


def _build_audit_prompt(targets: list[dict[str, Any]]) -> str:
    lines = [
        "You are Karum's menu auditor. For each target venue below, use Google Search grounding "
        "to read its website, menu, or delivery listing and return a JSON AuditMenuResponse.",
        "Grounding rules:",
        f"  - {DATA_RULE}",
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
        tags = ", ".join(str(t.get("urn", "")) for t in target.get("taste_tags", []))
        lines.append(
            f"  {index}. {as_data(door.get('name', 'unknown'))} "
            f"({as_data(door.get('city', 'unknown'))})"
            f" | website={as_data(target.get('website', ''))}"
            f" | tags={as_data(tags)}"
        )
    lines.append(
        "\nReturn only the JSON AuditMenuResponse: results (one per target) each with door, "
        "succeeded, shelfUnknown, incumbents[{name, sources[{url, title}]}], competitorCount, "
        "shelfOpportunity, menuSummary, sources[{url, title}], plus attribution{modelId, "
        "promptVersion, evidenceIds, ungrounded:false}, errorCode, errorMessage."
    )
    return "\n".join(lines)


def _unknown_result(target: dict[str, Any]) -> MenuAuditResult:
    """One door whose shelf could not be read: keep the Qloo row, mark the shelf unknown."""
    door = target.get("door", {}) or {}
    return MenuAuditResult(
        door=EntityRef(
            id=str(door.get("id", "")),
            type=str(door.get("type", "urn:entity:place")),
            name=str(door.get("name", "")),
            city=str(door.get("city", "")),
        ),
        succeeded=False,
        shelf_unknown=True,
        shelf_opportunity=0.0,
    )


def _mentions(name: str, text: str) -> bool:
    return bool(name) and name.casefold() in text.casefold()


def _incumbent_is_sourced(
    result: MenuAuditResult,
    incumbent_index: int,
    qloo_names: Sequence[str],
) -> bool:
    """A competitor is named only when a source or the Qloo payload carries it (§16)."""
    incumbent = result.incumbents[incumbent_index]
    if not incumbent.name.strip():
        return False
    sources = [*incumbent.sources, *result.sources]
    if any(_mentions(incumbent.name, f"{source.title} {source.url}") for source in sources):
        return True
    return any(
        _mentions(incumbent.name, known) or _mentions(known, incumbent.name)
        for known in qloo_names
        if known.strip()
    )


def _validate_incumbents(
    results: Sequence[MenuAuditResult],
    qloo_incumbent_names: Mapping[str, Sequence[str]],
    targets: Sequence[Mapping[str, Any]] = (),
) -> list[MenuAuditResult]:
    """Drop competitors the audit cannot source, and keep O_shelf consistent.

    The Qloo payload is keyed by the request's door id, so a model that mangles
    the door id it echoes back must not lose a sourced incumbent: the request's
    own target wins, and the result's door id is corrected to it (PLAN.md §16).
    """
    kept_results: list[MenuAuditResult] = []
    for position, result in enumerate(results):
        target = targets[position] if position < len(targets) else {}
        door = target.get("door", {}) or {}
        door_id = str(door.get("id", "")) or result.door.id
        if door_id != result.door.id:
            logger.warning(
                "audit_menu.door_id_mismatch model=%s request=%s",
                result.door.id,
                door_id,
            )
            result.door.id = door_id
        kept = [
            incumbent
            for index, incumbent in enumerate(result.incumbents)
            if _incumbent_is_sourced(result, index, qloo_incumbent_names.get(door_id, ()))
        ]
        if len(kept) != len(result.incumbents):
            logger.warning(
                "audit_menu.unsourced_incumbent door=%s dropped=%d kept=%d",
                result.door.id,
                len(result.incumbents) - len(kept),
                len(kept),
            )
            result.incumbents = kept
            result.competitor_count = len(kept)
            result.shelf_opportunity = max(0.0, 1.0 - 0.25 * len(kept))
        kept_results.append(result)
    return kept_results


def audit_menu(
    client: VertexClient,
    targets: list[dict[str, Any]],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    model_id: str | None = None,
    *,
    qloo_incumbent_names: Mapping[str, Sequence[str]] | None = None,
) -> AuditMenuResponse:
    """Audit the working shortlist with Search grounding.

    ``qloo_incumbent_names`` maps a door id to the competitor names the Qloo
    payload already carries, so a model-named competitor can be checked against
    either the grounding sources or that payload (PLAN.md §16).

    When Vertex fails, every target still comes back with its Qloo door row and
    ``shelf_unknown=True``, so the composite renormalises instead of losing the
    board (PLAN.md §16).
    """
    resolved_model = model_id or getattr(client, "grounded_model_id", client.model_id)
    attribution = service_attribution(
        model_id=resolved_model,
        prompt_version=prompt_version,
        evidence_ids=[],
        ungrounded=False,
    )

    cap = client.grounding_shortlist_cap
    if len(targets) > cap:
        logger.warning("audit_menu.too_many_targets targets=%d cap=%d", len(targets), cap)
        return AuditMenuResponse(
            results=[],
            attribution=attribution,
            error_code=ErrorCode.VALIDATION,
            error_message=(
                f"audit targets exceed the working shortlist cap ({len(targets)} > {cap}); "
                "pass only the doors that can appear on the board"
            ),
        )

    prompt = _build_audit_prompt(targets)
    try:
        response = client.generate(
            prompt,
            AuditMenuResponse,
            prompt_version,
            resolved_model,
            grounded=True,
            request_payload={"targets": targets},
        )
    except VertexError as exc:
        logger.warning(
            "audit_menu.vertex_failed model=%s prompt_version=%s detail=%s",
            resolved_model,
            prompt_version,
            exc.detail,
        )
        return AuditMenuResponse(
            results=[_unknown_result(target) for target in targets],
            attribution=attribution,
            error_code=ErrorCode.UPSTREAM,
            error_message=exc.detail,
        )

    response.results = _validate_incumbents(response.results, qloo_incumbent_names or {}, targets)
    # The provenance is the service's, not the model's: the model id, the
    # prompt version, and the grounded flag are resolved here (PLAN.md §3).
    # The audit's prose cites grounding URLs, not Qloo evidence ids, so the
    # cited ids stay empty.
    response.attribution = service_attribution(
        model_id=resolved_model,
        prompt_version=prompt_version,
        evidence_ids=[],
        ungrounded=False,
    )
    return response
