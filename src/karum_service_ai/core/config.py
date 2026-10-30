"""Environment-driven configuration for service-ai.

Every variable read here is documented in the infra repo's ``.env.example``,
along with the process that reads it. Defaults are chosen so the service boots
in fixture mode with no ``.env`` file present.

The variables are namespaced with ``AI_``. The bare spellings (``PORT``,
``MODE``, ``LOG_LEVEL``) are not read: on a cloud host they name something
else, and a stray ``MODE`` in the environment must not change how this service
behaves.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
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
        env_prefix="AI_",
        extra="ignore",
        populate_by_name=True,
    )

    # Service identity -----------------------------------------------------
    #: HTTP port. Compose and Cloud Run probe /health here.
    port: int = 8082
    #: Interface the HTTP server binds. The default serves the container; a
    #: laptop run can narrow it to the loopback.
    host: str = "0.0.0.0"
    #: gRPC port the gateway dials. Health stays on ``port``.
    grpc_port: int = 9092
    #: Operational mode reported on /health. ``fixture`` replays the recorded
    #: model outputs under ``AI_FIXTURES_DIR`` — no network, no credential;
    #: ``live`` calls the model; ``degraded`` calls the model and labels the
    #: service as degraded.
    mode: Mode = Mode.FIXTURE
    #: Human-readable service name used in logs and health payloads.
    service_name: str = "service-ai"

    # Recorded pack ---------------------------------------------------------
    #: Directory holding the recorded model outputs replayed in fixture mode.
    #: Relative paths resolve against this repository's root; the pack ships
    #: in karum-infra under ``fixtures/ai`` (mirrors ``TASTE_FIXTURES_DIR``).
    ai_fixtures_dir: Path = Path("../karum-infra/fixtures/ai")

    # Provider selection ----------------------------------------------------
    #: Which LLM provider to use: ``gemini`` (default), ``openai``,
    #: ``anthropic``, or ``ollama``. Switching providers is a one-env-var
    #: change; no code edit required (PLAN.md §71).
    provider: str = "gemini"
    #: Model id. Empty means "use the provider's default" (see
    #: ``vertex.backend.LangChainBackend._PROVIDER_DEFAULTS``).
    model: str = ""
    #: Comma-separated fallback providers tried in order when the primary
    #: fails (LangChain ``with_fallbacks``). Empty disables fallbacks.
    fallback_providers: str = ""
    #: Opt-in LangSmith tracing. Off by default so prompts never leave the box.
    langchain_tracing: bool = False

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        allowed = {"gemini", "google", "openai", "openai-responses", "anthropic", "ollama"}
        if value.lower() not in allowed:
            raise ValueError(f"AI_PROVIDER must be one of {sorted(allowed)}, got {value!r}")
        return value.lower()

    # Provider credentials --------------------------------------------------
    #: OpenAI API key. Used when ``provider == "openai"``.
    openai_api_key: str = ""
    #: Anthropic API key. Used when ``provider == "anthropic"``.
    anthropic_api_key: str = ""
    #: Base URL for an Ollama server. Used when ``provider == "ollama"``.
    ollama_base_url: str = ""
    #: Base URL for an OpenAI-compatible endpoint (e.g. Nebius). Used when
    #: ``provider == "openai"``. Leave empty for the official OpenAI API.
    openai_base_url: str = ""

    # Gemini ----------------------------------------------------------------
    #: Gemini API key from Google AI Studio. When set, model calls use this
    #: key and do not bill the Cloud project. Never logged. ``GEMINI_API_KEY``
    #: is still accepted as the legacy spelling.
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("AI_GEMINI_API_KEY", "GEMINI_API_KEY"),
    )
    #: GCP project id. Used only when ``gemini_api_key`` is empty, so calls
    #: go through Vertex on that project.
    gcp_project_id: str = ""
    #: GCP region hosting the Vertex endpoint.
    gcp_location: str = "us-central1"
    #: Gemini model id used for grounded checks and drafting. The default
    #: follows the provider's current Flash model; ``gemini-2.5-flash`` is
    #: retired for new API Studio keys and answers 404.
    gemini_model: str = "gemini-3.8-flash"
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
