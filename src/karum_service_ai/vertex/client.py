"""Vertex AI client for service-ai.

This is a skeleton. The real client calls Gemini 2.5 Flash through
``google-genai`` with **one Pydantic response model per RPC** so invalid JSON
fails closed. It will:

- set a timeout and a max-output-token cap,
- retry once when a response fails schema validation,
- log model, region, and prompt version — never credentials,
- return a typed error the gateway can map onto ``degraded``.

Replay tests inject a fake model that returns the fixture schemas. See PLAN.md
epic 4 for the full contract.
"""

from __future__ import annotations

from karum_service_ai.core.config import Settings


class VertexClient:
    """Typed client for Gemini via Vertex AI.

    A placeholder that holds configuration and performs no network calls. Each
    method will be added alongside its Pydantic response model.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def model_id(self) -> str:
        """Gemini model id, resolved from settings."""
        return self._settings.gemini_model

    @property
    def location(self) -> str:
        """GCP region, resolved from settings."""
        return self._settings.gcp_location

    # TODO(vertex): initialise the google-genai client from ADC.
    #
    # TODO(rpc): check_retail_presence(brand, evidence) -> PresenceResult
    # TODO(rpc): probe_shortlist(shortlist, budget) -> ProbeResult
    # TODO(rpc): draft_memo / draft_outreach / draft_listing -> typed drafts
    # TODO(rpc): llm_only_baseline(inputs) -> ungrounded baseline text
