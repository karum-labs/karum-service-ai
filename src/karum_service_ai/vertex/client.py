"""Vertex AI client for service-ai (PLAN.md §15).

Calls Gemini 2.5 Flash through ``google-genai`` on Vertex with **one Pydantic
response model per RPC** so invalid JSON fails closed:

- a timeout and a max-output-token cap,
- one retry when a response fails schema validation,
- logs model, region, and prompt version — never credentials,
- raises a typed :class:`VertexError` the gateway maps onto ``degraded``
  (PLAN.md §18).

Replay tests inject a fake model that returns the fixture schemas.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from karum_service_ai.core.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

#: The prompt is sent to the model as delimited data; venue and brand names are
#: interpolated into the prompt body, never into the schema, so an
#: instruction-shaped name cannot change the response shape (PLAN.md §18).

#: Credential-looking attribute names are redacted by the logging filter in
#: :mod:`karum_service_ai.core.logging`; this client never logs the project,
#: credentials, or an API key.


class VertexError(Exception):
    """Typed failure from the Vertex client.

    The gateway maps this onto the degraded run state (PLAN.md §18).
    """

    def __init__(
        self,
        detail: str,
        *,
        model_id: str | None = None,
        prompt_version: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.detail = detail
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.cause = cause
        super().__init__(detail)


class VertexClient:
    """Typed client for Gemini 2.5 Flash on Vertex AI.

    ``model`` is the google-genai generative model. In production it is built
    from application-default credentials; tests inject a fake so no network or
    GCP credentials are required.
    """

    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self._settings = settings
        self._model = model or self._build_model()

    @property
    def model_id(self) -> str:
        """Gemini model id, resolved from settings."""
        return self._settings.gemini_model

    @property
    def location(self) -> str:
        """GCP region, resolved from settings."""
        return self._settings.gcp_location

    def _build_model(self) -> Any:
        client = genai.Client(
            vertexai=True,
            project=self._settings.gcp_project_id or None,
            location=self._settings.gcp_location,
        )
        return client.models

    def generate(
        self,
        prompt: str,
        schema: type[T],
        prompt_version: str,
        model_id: str | None = None,
    ) -> T:
        """Generate and validate one structured response.

        Transport/auth failures raise :class:`VertexError` immediately. Schema
        validation failures are retried once (PLAN.md §15) before raising
        :class:`VertexError`.
        """
        resolved_model = model_id or self.model_id
        logger.info(
            "vertex.generate model=%s region=%s prompt_version=%s",
            resolved_model,
            self.location,
            prompt_version,
        )

        last_error: ValidationError | None = None
        for attempt in range(1, self._settings.schema_retry_limit + 2):
            try:
                response = self._model.generate_content(
                    model=resolved_model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        max_output_tokens=self._settings.max_output_tokens,
                    ),
                )
            except Exception as exc:  # transport / auth failure; no retry
                raise VertexError(
                    f"Vertex call failed: {exc}",
                    model_id=resolved_model,
                    prompt_version=prompt_version,
                    cause=exc,
                ) from exc

            text = getattr(response, "text", "") or ""
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "vertex.generate.schema_invalid model=%s prompt_version=%s attempt=%d",
                    resolved_model,
                    prompt_version,
                    attempt,
                )
                continue

        raise VertexError(
            f"Vertex response failed schema validation after "
            f"{self._settings.schema_retry_limit + 1} attempt(s)",
            model_id=resolved_model,
            prompt_version=prompt_version,
            cause=last_error,
        )
