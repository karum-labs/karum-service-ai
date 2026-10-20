"""FastAPI application factory for service-ai.

Creates the ASGI app, wires routes, and configures logging. The gRPC server the
gateway dials starts beside this app in the lifespan, so the two share one
process and one settings object; the HTTP surface exposes ``/health`` so the
service is probeable by Compose and Cloud Run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from karum_service_ai import __version__
from karum_service_ai.api.routes import health
from karum_service_ai.core.config import get_settings
from karum_service_ai.core.logging import configure as configure_logging
from karum_service_ai.grpcserv import serve as serve_grpc


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the gRPC server beside HTTP and stop it on shutdown."""
    grpc_server = serve_grpc(app.state.settings, app.state.settings.grpc_port)
    yield
    grpc_server.stop(3)


def create_app() -> FastAPI:
    """Build the FastAPI application.

    Settings are read once at import so the health endpoint and any future
    handlers share a single cached configuration.
    """
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="Karum service-ai",
        description="Gemini, search grounding, and pitch drafting for the Karum brand agent.",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
        lifespan=_lifespan,
    )

    # Expose the resolved settings on the app state so handlers and tests can
    # reach them without re-reading the environment.
    app.state.settings = settings

    app.include_router(health.router)

    return app


# Module-level ASGI entrypoint. ``uvicorn karum_service_ai.main:app`` works out
# of the box, and the gRPC server can import :func:`create_app` directly.
app = create_app()
