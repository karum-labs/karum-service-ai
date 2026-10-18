"""Environment-driven configuration for service-ai.

Every variable read here is documented in the infra repo's ``.env.example``,
along with the process that reads it. Defaults are chosen so the service boots
in fixture mode with no ``.env`` file present.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Mode(StrEnum):
    """How the service is currently operating.

    Mirrors the gateway's mode so ``/health`` payloads are consistent across the
    stack.
    """

    LIVE = "live"
    FIXTURE = "fixture"
    DEGRADED = "degraded"


class Settings(BaseSettings):
    """Settings for service-ai, loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service identity -----------------------------------------------------
    #: gRPC port the AI service listens on.
    port: int = 8082
    #: Operational mode reported on /health.
    mode: Mode = Mode.FIXTURE
    #: Human-readable service name used in logs and health payloads.
    service_name: str = "service-ai"

    # Gemini ----------------------------------------------------------------
    #: Gemini API key from Google AI Studio. When set, model calls use this
    #: key and do not bill the Cloud project. Never logged.
    gemini_api_key: str = ""
    #: GCP project id. Used only when ``gemini_api_key`` is empty, so calls
    #: go through Vertex on that project.
    gcp_project_id: str = ""
    #: GCP region hosting the Vertex endpoint.
    gcp_location: str = "us-central1"
    #: Gemini model id used for grounded checks and drafting.
    gemini_model: str = "gemini-2.5-flash"
    #: Per-request timeout for a single model call, in seconds.
    model_timeout_seconds: float = 30.0
    #: Maximum output tokens allowed for one draft.
    max_output_tokens: int = 2048
    #: Number of retries when a response fails schema validation.
    schema_retry_limit: int = 1

    # Grounding ------------------------------------------------------------
    #: Maximum number of shortlist brands to run search grounding against.
    grounding_shortlist_cap: int = 5
    #: Maximum bounded follow-up taste probes the agent may request per run.
    probe_budget: int = 3

    # Logging -------------------------------------------------------------
    #: Log level name.
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, cached after the first call."""
    return Settings()
