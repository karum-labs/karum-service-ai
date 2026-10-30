"""Provider backend factory tests (PLAN.md §71-72)."""
from __future__ import annotations

from karum_service_ai.core.config import Mode, Settings
from karum_service_ai.vertex.backend import (
    FixtureBackend,
    GeminiBackend,
    LangChainBackend,
    build_backend,
)


def test_build_backend_fixture_mode_returns_fixture_backend() -> None:
    settings = Settings(mode=Mode.FIXTURE)
    backend = build_backend(settings)
    assert isinstance(backend, FixtureBackend)


def test_build_backend_gemini_live_returns_gemini_backend() -> None:
    settings = Settings(mode=Mode.LIVE, gemini_api_key="k", gcp_project_id="p")
    backend = build_backend(settings)
    assert isinstance(backend, GeminiBackend)
    assert backend.model_id == "gemini-3.8-flash"


def test_build_backend_openai_uses_default_model() -> None:
    settings = Settings(mode=Mode.LIVE, provider="openai", openai_api_key="k")
    backend = build_backend(settings)
    assert isinstance(backend, LangChainBackend)
    assert backend.model_id == "gpt-4o-mini"


def test_build_backend_explicit_model_overrides_default() -> None:
    settings = Settings(mode=Mode.LIVE, provider="openai", model="gpt-4o", openai_api_key="k")
    backend = build_backend(settings)
    assert isinstance(backend, LangChainBackend)
    assert backend.model_id == "gpt-4o"


def test_build_backend_rejects_unknown_provider() -> None:
    import pytest

    with pytest.raises(ValueError):
        Settings(mode=Mode.LIVE, provider="nope", gemini_api_key="k", gcp_project_id="p")