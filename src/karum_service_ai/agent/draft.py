"""Pitches, SKUs, and briefs (PLAN.md §17) plus draft resilience (§18).

DraftDoorPitch returns the shopkeeper script, the displacement battlecard when an
incumbent is present, the recommended SKU chosen from the brand's own list by the
door's shared taste tags, and the mode-specific paragraph:

- flagship  -> site-brief line for the top corridors
- collab    -> activation proposal
- wholesale -> empty

LlmOnlyBaseline answers the same question with no Qloo payload; it is labeled
ungrounded and its venue names source the free-entry skip candidates.

Resilience (§18): when Vertex fails, the caller still gets the door list and the
failed draft is the only error. Template sentences interpolate evidence fields
only, and a sentence whose cited evidence id is missing is dropped.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from karum_service_ai.agent.prompting import DATA_RULE, as_data
from karum_service_ai.vertex.client import VertexClient, VertexError
from karum_service_ai.vertex.models import (
    BrandContext,
    DoorContext,
    DraftDoorPitchResponse,
    ErrorCode,
    LlmOnlyBaselineResponse,
    Sku,
    service_attribution,
)

logger = logging.getLogger(__name__)

DRAFT_PROMPT_VERSION = "draft-door-pitch@v1"
BASELINE_PROMPT_VERSION = "llm-only-baseline@v1"

#: Evidence-citation placeholder, e.g. ``{ev-0001}``. A sentence whose cited id
#: is missing from the evidence map is dropped (PLAN.md §18).
_CITE_RE = re.compile(r"\{([A-Za-z0-9_-]+)\}")

#: An evidence id as the draft writes it: ``ev-0001``, with or without braces.
#: The gateway can issue more than four digits, so the shape is the prefix plus
#: any id characters, not a fixed width.
_EVIDENCE_ID_RE = re.compile(r"\bev-[A-Za-z0-9_-]+\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_DIGIT_RE = re.compile(r"\d")


def enforce_citations(
    text: str,
    evidence_ids: Sequence[str] | None = None,
) -> tuple[str, list[str]]:
    """Drop sentences that state a number without a verifiable evidence id.

    Returns the kept text and the dropped sentences. When ``evidence_ids`` is
    supplied, a cited id must be one of them; a sentence citing an invented id
    is dropped too (PLAN.md §17, §18). An empty allow-set is strict: every
    sentence that states a number is dropped, because the run has no evidence
    to back one up. ``None`` means no allow-set was supplied, which is the
    in-process path that has no run evidence to check against.
    """
    allowed = None if evidence_ids is None else set(evidence_ids)
    kept: list[str] = []
    dropped: list[str] = []
    for sentence in _SENTENCE_RE.split(text.strip()):
        if not sentence:
            continue
        if _DIGIT_RE.search(sentence):
            cited = _EVIDENCE_ID_RE.findall(sentence)
            if not cited or (allowed is not None and any(cid not in allowed for cid in cited)):
                dropped.append(sentence)
                continue
        kept.append(sentence)
    return " ".join(kept).strip(), dropped


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _sku_from_brand(sku: Sku, brand: BrandContext) -> Sku:
    """Keep the model's SKU only when it is one of the brand's own products."""
    if not brand.skus:
        return sku
    for candidate in brand.skus:
        if sku.id and candidate.id == sku.id:
            return candidate
    for candidate in brand.skus:
        if sku.name and _normalized(candidate.name) == _normalized(sku.name):
            return candidate
    logger.warning(
        "draft_door_pitch.sku_outside_brand_list model_sku=%r brand=%s fallback=%r",
        sku.id or sku.name,
        brand.name,
        brand.skus[0].id,
    )
    return brand.skus[0]


def _build_pitch_prompt(
    brand: BrandContext,
    door: DoorContext,
    city: str,
    mode: str,
    max_words: int,
) -> str:
    tags = ", ".join(t.label or t.urn for t in door.shared_tags) or "none recorded"
    compare = ", ".join(t.label or t.urn for t in door.compare_tags) or "none"
    audit = door.audit
    incumbent = audit.incumbents[0].name if audit.incumbents else "none found"
    skus = ", ".join(s.name for s in brand.skus) or "none listed"
    return (
        "You are Karum's door-pitch writer. Write a short shopkeeper script (under "
        f"{max_words} words) for a brand rep walking into {as_data(door.audit.door.name)} "
        f"in {as_data(city)}. The brand is {as_data(brand.name)}. "
        f"The door's shared taste tags: {as_data(tags)}. "
        f"The door's affinity percentile: {door.affinity.value:.2f} (raw {door.affinity.raw_value:.2f}). "
        f"Compare tags: {as_data(compare)}. Incumbent on the shelf: {as_data(incumbent)}. "
        # The audit's menu summary is live web content. It is deliberately not
        # relayed into this prompt: the draft already receives the structured
        # incumbent and competitor findings, and excluding the free-text
        # summary closes the prompt-injection relay (PLAN.md §17, §18).
        f"Brand SKUs: {as_data(skus)}. "
        f"Rules: {DATA_RULE} "
        "Every number in the script cites an evidence id, e.g. "
        "'ranks first of 25 (ev-0001)'. Do not invent evidence ids. "
        "Return a JSON DraftDoorPitchResponse: script, battlecard (only when an "
        "incumbent is present: headline, incumbentName, talkingPoints each citing "
        "an evidence id, compareSummary using the overlap tags when they exist and "
        "the affinity plus menu finding otherwise), recommendedSku {id, name} chosen "
        "from the brand's own list with the door's shared taste tags, modeParagraph "
        "(site-brief line in flagship mode, activation proposal in collab mode, empty "
        "in wholesale), attribution{modelId, promptVersion, evidenceIds, "
        "ungrounded:false}, errorCode, errorMessage."
    )


def draft_door_pitch(
    client: VertexClient,
    brand: BrandContext,
    door: DoorContext,
    city: str,
    mode: str = "wholesale",
    max_words: int = 220,
    prompt_version: str = DRAFT_PROMPT_VERSION,
    model_id: str | None = None,
    *,
    evidence_ids: Sequence[str] | None = None,
) -> DraftDoorPitchResponse:
    """Draft one door's pitch. Returns a degraded response on Vertex failure.

    The returned script is filtered through :func:`enforce_citations`, the SKU is
    checked against the brand's own list, the wholesale paragraph is cleared, and
    a battlecard is dropped when the audit found no incumbent (PLAN.md §17).

    ``evidence_ids`` is the run's evidence set, the allow-set a cited id must
    belong to. ``None`` means the caller has no run evidence to check against;
    an empty list is strict and drops every sentence that states a number.
    """
    resolved_model = model_id or client.model_id
    prompt = _build_pitch_prompt(brand, door, city, mode, max_words)
    try:
        response = client.generate(
            prompt,
            DraftDoorPitchResponse,
            prompt_version,
            resolved_model,
            # The request's own data identifies the recording in fixture mode;
            # the shape matches what a recording's ``match`` spec names.
            request_payload={
                "brand": {"name": brand.name},
                "city": city,
                "mode": mode,
                "door": {"audit": {"door": {"name": door.audit.door.name}}},
            },
        )
    except VertexError as exc:
        logger.warning(
            "draft_door_pitch.vertex_failed model=%s door=%s detail=%s",
            resolved_model,
            door.audit.door.name,
            exc.detail,
        )
        return DraftDoorPitchResponse(
            script="",
            battlecard=None,
            recommended_sku=Sku(id="", name=""),
            mode_paragraph="",
            attribution=service_attribution(
                model_id=resolved_model,
                prompt_version=prompt_version,
                evidence_ids=[],
                ungrounded=False,
            ),
            error_code=ErrorCode.UPSTREAM,
            error_message=exc.detail,
        )

    script, dropped = enforce_citations(response.script, evidence_ids)
    if dropped:
        logger.warning(
            "draft_door_pitch.uncited_sentences door=%s dropped=%d",
            door.audit.door.name,
            len(dropped),
        )
        response.script = script
        if not script:
            response.error_code = ErrorCode.VALIDATION
            response.error_message = (
                "every sentence was dropped: a number carried no verifiable evidence id"
            )

    if response.battlecard is not None:
        # The headline and the compare summary are numeric-bearing prose too,
        # so they go through the same allow-set as the script (PLAN.md §17).
        headline, _ = enforce_citations(response.battlecard.headline, evidence_ids)
        fallback_headline = f"Displace {response.battlecard.incumbent_name}".strip()
        response.battlecard.headline = headline or fallback_headline
        if response.battlecard.compare_summary:
            compare_summary, _ = enforce_citations(
                response.battlecard.compare_summary, evidence_ids
            )
            response.battlecard.compare_summary = compare_summary
        points = [
            kept
            for point in response.battlecard.talking_points
            if (kept := enforce_citations(point, evidence_ids)[0])
        ]
        response.battlecard.talking_points = points
        if not door.audit.incumbents:
            response.battlecard = None

    if mode == "wholesale":
        response.mode_paragraph = ""
    else:
        paragraph, _ = enforce_citations(response.mode_paragraph, evidence_ids)
        response.mode_paragraph = paragraph

    response.recommended_sku = _sku_from_brand(response.recommended_sku, brand)
    # Provenance is the service's, not the model's. Overwrite whatever the
    # model wrote, including a forged model id or prompt version (PLAN.md §3).
    response.attribution = service_attribution(
        model_id=resolved_model,
        prompt_version=prompt_version,
        evidence_ids=_cited_ids(
            response.script,
            *(response.battlecard.talking_points if response.battlecard is not None else ()),
        ),
        ungrounded=False,
    )
    return response


def _cited_ids(*texts: str) -> list[str]:
    """The evidence ids the kept prose actually cites, in a stable order."""
    found: set[str] = set()
    for text in texts:
        found.update(_EVIDENCE_ID_RE.findall(text))
    return sorted(found)


def _build_baseline_prompt(
    brand: BrandContext,
    city: str,
    mode: str,
    max_words: int,
) -> str:
    skus = ", ".join(s.name for s in brand.skus) or "none listed"
    return (
        "You are a generic LLM with no Qloo payload. Answer the same question "
        f"without any taste-graph data: brand {as_data(brand.name)}, "
        f"city {as_data(city)}, mode {as_data(mode)}. "
        f"Brand SKUs: {as_data(skus)}. "
        f"Rules: {DATA_RULE} "
        "Give wide-band, unverified advice and suggest 2-4 venue names the rep could "
        "approach. Return a JSON LlmOnlyBaselineResponse: text, venueNames, "
        "attribution{modelId, promptVersion, evidenceIds:[], ungrounded:true}, "
        "errorCode, errorMessage. Label the panel ungrounded."
    )


def llm_only_baseline(
    client: VertexClient,
    brand: BrandContext,
    city: str,
    mode: str = "wholesale",
    max_words: int = 160,
    prompt_version: str = BASELINE_PROMPT_VERSION,
    model_id: str | None = None,
) -> LlmOnlyBaselineResponse:
    """Return the generic-LLM counterfactual with no Qloo payload, labeled ungrounded."""
    resolved_model = model_id or client.model_id
    prompt = _build_baseline_prompt(brand, city, mode, max_words)
    try:
        response = client.generate(
            prompt,
            LlmOnlyBaselineResponse,
            prompt_version,
            resolved_model,
            request_payload={"brand": {"name": brand.name}, "city": city, "mode": mode},
        )
    except VertexError as exc:
        logger.warning(
            "llm_only_baseline.vertex_failed model=%s detail=%s",
            resolved_model,
            exc.detail,
        )
        return LlmOnlyBaselineResponse(
            text="",
            venue_names=[],
            attribution=service_attribution(
                model_id=resolved_model,
                prompt_version=prompt_version,
                evidence_ids=[],
                ungrounded=True,
            ),
            error_code=ErrorCode.UPSTREAM,
            error_message=exc.detail,
        )
    # The counterfactual has no Qloo evidence to cite, so a numeric claim in it
    # cannot pass the filter: keep the unquantified advice and drop the rest
    # (PLAN.md §17). The panel stays labeled ungrounded either way.
    text, _ = enforce_citations(response.text, ())
    response.text = text
    # The provenance is the service's, never the model's self-report (PLAN.md §3).
    response.attribution = service_attribution(
        model_id=resolved_model,
        prompt_version=prompt_version,
        evidence_ids=[],
        ungrounded=True,
    )
    return response


def render_sentence(template: str, evidence: dict[str, str]) -> str | None:
    """Interpolate evidence fields only; drop the sentence if any cited id is missing.

    Used by the degraded path so a draft never cites an evidence id it cannot
    back up (PLAN.md §18). Returns None when the sentence must be dropped.
    """
    missing = [cid for cid in _CITE_RE.findall(template) if cid not in evidence]
    if missing:
        return None
    return _CITE_RE.sub(lambda m: evidence[m.group(1)], template)
