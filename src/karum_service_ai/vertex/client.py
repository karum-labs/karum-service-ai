"""Vertex AI client for service-ai (PLAN.md §15).

Calls the Gemini Flash model through ``google-genai`` with **one Pydantic
response model per RPC** so invalid JSON fails closed:

- a timeout and a max-output-token cap,
- one retry when a response fails schema validation,
- logs model, region, and prompt version — never credentials,
- raises a typed :class:`VertexError` the gateway maps onto ``degraded``
  (PLAN.md §18).

In ``fixture`` mode the client replays the recorded pack instead of calling
the model: no network, no credential, and the same typed failures when a
recording is missing (PLAN.md §4).

Replay tests inject a fake model that returns the fixture schemas.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, TypeVar

from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel, ValidationError

from karum_service_ai.core.config import Mode, Settings
from karum_service_ai.vertex.errors import VertexError
from karum_service_ai.vertex.fixtures import FixturePack, resolve_fixtures_dir
from karum_service_ai.vertex.transport import RecordedModelTransport

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = ["VertexClient", "VertexError"]

#: The prompt is sent to the model as delimited data; venue and brand names are
#: interpolated into the prompt body, never into the schema, so an
#: instruction-shaped name cannot change the response shape (PLAN.md §18).

#: Credential-looking attribute names are redacted by the logging filter in
#: :mod:`karum_service_ai.core.logging`; this client never logs the project,
#: credentials, or an API key.


def _failure_detail(exc: Exception) -> str:
    """A client-safe summary of a failed model call (S-11, PLAN.md §18).

    A ``google-genai`` API error renders as the provider's raw response body
    — quota internals, request echoes — and that text used to ride the typed
    error into ``error_message`` and on to the client UI. Only the error class
    and HTTP status are needed there; the raw detail is logged server-side and
    stays on the ``cause`` chain.
    """
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        label = type(exc).__name__
        if isinstance(code, int):
            return f"Vertex call failed ({label}, HTTP {code})"
        return f"Vertex call failed ({label})"
    return f"Vertex call failed: {exc}"


class VertexClient:
    """Typed client for the Gemini Flash model.

    The transport is Google AI Studio when ``GEMINI_API_KEY`` is set and
    Vertex AI on the configured project otherwise. In production the model is
    built from application-default credentials; tests inject a fake so no
    network or GCP credentials are required.

    In ``fixture`` mode the client replays the recorded pack instead of
    building a model, so no credential is required and no call leaves the
    process (PLAN.md §4).
    """

    def __init__(self, settings: Settings, model: Any | None = None, backend: Any | None = None) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._model: Any | None = model
        self._transport: RecordedModelTransport | None = None
        self._backend: Any | None = backend
        # A failed build is kept, not raised: the model client and the fixture
        # pack are resolved here, but a missing credential or pack must not
        # escape the servicer as a gRPC UNKNOWN. generate() re-raises the typed
        # VertexError, which the agent functions catch and map onto
        # ERROR_CODE_UPSTREAM (PLAN.md §15, §18, §4).
        self._build_error: VertexError | None = None
        if self._backend is None:
            try:
                from karum_service_ai.vertex.backend import build_backend

                self._backend = build_backend(settings, model=model)
            except Exception as exc:  # noqa: BLE001 - typed below
                logger.warning("vertex.transport_unavailable detail=%s", exc)
                self._build_error = VertexError(str(exc))

    @property
    def transport_name(self) -> str:
        """The transport this client speaks through: ``fixture`` or ``live``."""
        if self._settings.mode == Mode.FIXTURE:
            return "fixture"
        return "live"

    @property
    def model_id(self) -> str:
        """Resolved model id (provider default or explicit)."""
        backend = self._backend
        if backend is not None and hasattr(backend, "model_id"):
            return backend.model_id
        return self._settings.gemini_model or self._settings.model

    @property
    def location(self) -> str:
        """Provider/region label for logging."""
        return self._settings.gcp_location or self._settings.provider

    @property
    def timeout_ms(self) -> int:
        """Per-request timeout in milliseconds (the google-genai unit)."""
        return int(self._settings.model_timeout_seconds * 1000)

    @property
    def grounding_shortlist_cap(self) -> int:
        """Maximum targets the menu audit may send to Search grounding."""
        return self._settings.grounding_shortlist_cap

    @property
    def last_usage(self) -> dict[str, int] | None:
        backend = self._backend
        if backend is not None and hasattr(backend, "last_usage"):
            return backend.last_usage
        return None

    def _build_model(self) -> Any:
        # A Studio key is not a Cloud project credential. When it is set, the
        # call does not use Vertex or AI_GCP_PROJECT_ID.
        if self._settings.gemini_api_key:
            client = genai.Client(api_key=self._settings.gemini_api_key)
        else:
            if not self._settings.gcp_project_id:
                raise VertexError(
                    "no model credential: set AI_GEMINI_API_KEY for Gemini AI Studio "
                    "or AI_GCP_PROJECT_ID for Vertex AI"
                )
            client = genai.Client(
                vertexai=True,
                project=self._settings.gcp_project_id,
                location=self._settings.gcp_location,
            )
        # Keep the client alive for the lifetime of this object. genai.Client
        # closes its HTTP client in __del__, so a client that is only reachable
        # through `client.models` is collected while the service is still
        # running and every later call fails with "client has been closed".
        self._client = client
        return client.models

    def generate(
        self,
        prompt: str,
        schema: type[T],
        prompt_version: str,
        model_id: str | None = None,
        *,
        grounded: bool = False,
        request_payload: Mapping[str, Any] | None = None,
    ) -> T:
        """Generate and validate one structured response.

        Transport/auth failures raise :class:`VertexError` immediately. Schema
        validation failures are retried once (PLAN.md §15) before raising
        :class:`VertexError`.

        ``grounded`` attaches the Google Search tool. It is set only for the
        menu audit: the audit is the one RPC that reads a live menu, and the
        plan caps it to the working shortlist (PLAN.md §16).

        ``request_payload`` is the request's own data, not the prompt. In
        fixture mode it is what identifies the recording to replay; the live
        path ignores it.
        """
        resolved_model = model_id or self.model_id
        logger.info(
            "vertex.generate model=%s region=%s prompt_version=%s grounded=%s timeout_ms=%d",
            resolved_model,
            self.location,
            prompt_version,
            grounded,
            self.timeout_ms,
        )

        if self._build_error is not None:
            raise self._build_error

        backend = self._backend
        last_error: ValidationError | None = None
        for attempt in range(1, self._settings.schema_retry_limit + 2):
            try:
                text = backend.generate(
                    prompt,
                    schema,
                    grounded=grounded,
                    timeout_ms=self.timeout_ms,
                    max_tokens=self._settings.max_output_tokens,
                    request_payload=request_payload,
                )
            except Exception as exc:  # transport / auth failure; no retry
                # Fixture misses are typed errors the agent layer expects to
                # receive untranslated (PLAN.md §4, §18).
                from karum_service_ai.vertex.errors import FixtureError

                if isinstance(exc, FixtureError):
                    raise
                logger.warning(
                    "vertex.generate.transport_failed model=%s prompt_version=%s error=%s",
                    resolved_model,
                    prompt_version,
                    exc,
                )
                raise VertexError(
                    _failure_detail(exc),
                    model_id=resolved_model,
                    prompt_version=prompt_version,
                    cause=exc,
                ) from exc

            try:
                result = schema.model_validate_json(text)
                usage = self.last_usage
                if usage:
                    logger.info('vertex.generate.usage model=%s usage=%s', resolved_model, usage)
                return result
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
