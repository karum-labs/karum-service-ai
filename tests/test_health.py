"""Tests for the /health endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from karum_service_ai.api.routes import health
from karum_service_ai.core.config import Mode, Settings
from karum_service_ai.main import app


def _health_body(settings: Settings) -> dict[str, Any]:
    """One GET /health against an app holding these settings (S-06)."""
    probe = FastAPI()
    probe.state.settings = settings
    probe.include_router(health.router)
    with TestClient(probe) as client:
        response = client.get("/health")
    assert response.status_code == 200
    return response.json()


def test_health_returns_ok() -> None:
    """GET /health should report the service name, status, and mode."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "service-ai"
    assert body["status"] == "ok"
    # Default mode with no env overrides is fixture, and the recorded pack
    # ships beside the service, so the default app reports ok.
    assert body["mode"] == "fixture"


def test_health_includes_uptime_and_timestamp() -> None:
    """The health payload should carry uptime and a UTC timestamp."""
    with TestClient(app) as client:
        body = client.get("/health").json()

    assert isinstance(body["uptime_seconds"], int)
    assert body["uptime_seconds"] >= 0
    # ISO-8601 timestamp, terminated with +00:00 for UTC.
    assert body["time"].endswith("+00:00")


def test_health_reports_degraded_without_the_fixture_pack(tmp_path: Path) -> None:
    """Fixture mode with no pack reports degraded, not a false ok (S-06)."""
    body = _health_body(Settings(mode=Mode.FIXTURE, ai_fixtures_dir=tmp_path / "no-pack"))

    assert body["status"] == "degraded"
    assert body["mode"] == "fixture"


def test_health_reports_degraded_in_live_mode_without_a_credential() -> None:
    """Live mode with neither a Studio key nor a project reports degraded (S-06)."""
    settings = Settings(mode=Mode.LIVE, gemini_api_key="", gcp_project_id="")
    body = _health_body(settings)

    assert body["status"] == "degraded"
    assert body["mode"] == "live"


def test_health_reports_ok_in_live_mode_with_a_credential() -> None:
    """Live mode with a Studio key is serviceable, so the status stays ok (S-06)."""
    body = _health_body(Settings(mode=Mode.LIVE, gemini_api_key="studio-key"))

    assert body["status"] == "ok"
    assert body["mode"] == "live"
