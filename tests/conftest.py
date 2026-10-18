"""Shared fixtures for service-ai tests.

The Vertex client is exercised with a fake model so no network or GCP
credentials are ever required (PLAN.md §15, §38).
"""

from __future__ import annotations

from karum_service_ai.core.config import Settings
from karum_service_ai.vertex.client import VertexClient


class FakeResponse:
    """Stand-in for a google.genai GenerateContentResponse."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    """Stand-in for the google-genai generative model.

    Records each ``generate_content`` call. ``text`` may be a single string
    returned every time, or a list of strings returned in order (the last one
    repeats once the list is exhausted), which is useful for retry tests.
    """

    def __init__(self, text: str | list[str] = "{}") -> None:
        self.calls: list[tuple] = []
        self._queue: list[str] = [text] if isinstance(text, str) else list(text)
        self._default = self._queue[-1] if self._queue else "{}"

    def generate_content(self, model: str, contents: str, config: object) -> FakeResponse:
        self.calls.append((model, contents, config))
        text = self._queue.pop(0) if self._queue else self._default
        return FakeResponse(text)


def make_client(
    text: str = "{}",
    *,
    model_timeout_seconds: float = 30.0,
    max_output_tokens: int = 2048,
    schema_retry_limit: int = 1,
) -> tuple[VertexClient, FakeModel, Settings]:
    """Build a VertexClient wired to a fake model, plus the fake and settings."""
    settings = Settings(
        model_timeout_seconds=model_timeout_seconds,
        max_output_tokens=max_output_tokens,
        schema_retry_limit=schema_retry_limit,
    )
    fake = FakeModel(text=text)
    client = VertexClient(settings, model=fake)
    return client, fake, settings
