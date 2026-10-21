"""Typed failures from the model transport (PLAN.md §18).

The errors live in their own module so the recorded-replay path
(:mod:`karum_service_ai.vertex.fixtures`) can raise them without importing the
client that would import it back. Every failure is a :class:`VertexError`, so
the agent functions degrade one way whether the model call failed or the
recording was missing.
"""

from __future__ import annotations


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


class FixtureError(VertexError):
    """The recorded pack cannot be read: no directory, no manifest, bad file."""


class FixtureMissError(FixtureError):
    """No recorded response answers this request.

    Fixture mode replays; it never invents. A miss is a bug in the pack or in
    the request, and it is reported as one (PLAN.md §4).
    """
