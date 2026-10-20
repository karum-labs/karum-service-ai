"""The environment contract of service-ai (PLAN.md §5).

The variables are namespaced with ``AI_``. A bare ``PORT`` or ``MODE`` in the
environment must not change how the service behaves, and the legacy
``GEMINI_API_KEY`` spelling must keep working.
"""

from __future__ import annotations

import pytest

from karum_service_ai.core.config import Settings


def test_ai_prefixed_variables_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PORT", "9000")
    monkeypatch.setenv("AI_GRPC_PORT", "9999")
    monkeypatch.setenv("AI_MODE", "live")
    monkeypatch.setenv("AI_SERVICE_NAME", "ai-under-test")
    monkeypatch.setenv("AI_GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("AI_MODEL_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("AI_SCHEMA_RETRY_LIMIT", "3")
    monkeypatch.setenv("AI_GROUNDING_SHORTLIST_CAP", "7")
    monkeypatch.setenv("AI_PROBE_BUDGET", "2")
    monkeypatch.setenv("AI_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.port == 9000
    assert settings.grpc_port == 9999
    assert settings.mode.value == "live"
    assert settings.service_name == "ai-under-test"
    assert settings.gemini_model == "gemini-test"
    assert settings.model_timeout_seconds == 12.5
    assert settings.max_output_tokens == 1024
    assert settings.schema_retry_limit == 3
    assert settings.grounding_shortlist_cap == 7
    assert settings.probe_budget == 2
    assert settings.log_level == "DEBUG"


def test_bare_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stray PORT, MODE, or LOG_LEVEL must not change the service."""
    monkeypatch.setenv("PORT", "1234")
    monkeypatch.setenv("MODE", "degraded")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("GRPC_PORT", "4321")
    monkeypatch.setenv("SERVICE_NAME", "someone-else")

    settings = Settings()

    assert settings.port == 8082
    assert settings.grpc_port == 9092
    assert settings.mode.value == "fixture"
    assert settings.log_level == "INFO"
    assert settings.service_name == "service-ai"


def test_the_legacy_gemini_key_spelling_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-key")
    assert Settings().gemini_api_key == "legacy-key"

    monkeypatch.setenv("AI_GEMINI_API_KEY", "current-key")
    assert Settings().gemini_api_key == "current-key"


def test_the_defaults_need_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(dict(__import__("os").environ)):
        if name.startswith("AI_") or name in {"PORT", "MODE", "LOG_LEVEL", "GEMINI_API_KEY"}:
            monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.port == 8082
    assert settings.grpc_port == 9092
    assert settings.mode.value == "fixture"
    assert settings.gemini_model == "gemini-3.8-flash"
    assert settings.gemini_api_key == ""
    assert settings.gcp_project_id == ""
