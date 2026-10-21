"""Health endpoint for service-ai.

``GET /health`` is the single probe that Compose, Cloud Run, and the uptime
check point at. It reports the service's operational mode so a fixture or
degraded run is visible to the client banner, and the status is an honest read
of whether the service can serve that mode: a missing fixture pack or model
credential reports ``degraded`` instead of a hardcoded ``ok`` (S-06). The HTTP
status stays 200 — the process is up, and only the service knows whether its
dependencies can answer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from time import monotonic
from typing import Any

from fastapi import APIRouter, Request

from karum_service_ai.core.config import Mode, Settings
from karum_service_ai.vertex.fixtures import MANIFEST_NAME, resolve_fixtures_dir

router = APIRouter()

# Process start time, used to compute uptime. Set at import so every health
# response reports a stable figure.
_STARTED = monotonic()


def _status(settings: Settings) -> str:
    """``degraded`` when the configured mode cannot be served, else ``ok``.

    Fixture mode needs the recorded pack's manifest; the live path needs a
    credential (a Studio key or a Cloud project). Without one, every request
    fails as a typed upstream error, so the health status says so up front.
    """
    if settings.mode == Mode.FIXTURE:
        pack = resolve_fixtures_dir(settings.ai_fixtures_dir)
        return "ok" if (pack / MANIFEST_NAME).is_file() else "degraded"
    if settings.gemini_api_key or settings.gcp_project_id:
        return "ok"
    return "degraded"


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Return service identity, status, and operational mode."""
    settings: Settings = request.app.state.settings
    return {
        "service": settings.service_name,
        "status": _status(settings),
        "mode": settings.mode.value,
        "uptime_seconds": int(monotonic() - _STARTED),
        "time": datetime.now(UTC).isoformat(),
    }
