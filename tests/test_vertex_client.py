"""Vertex client tests (PLAN.md §15): Pydantic schema per RPC, timeout/token cap,
one retry on invalid JSON, typed error, and credentials kept out of logs."""

from __future__ import annotations

from pydantic import BaseModel
from tests.conftest import make_client

from karum_service_ai.core.config import Settings
from karum_service_ai.vertex.client import VertexClient, VertexError
from karum_service_ai.vertex.models import ErrorCode, ModelAttribution


class _Thing(BaseModel):
    id: str
    value: int


def test_generate_returns_validated_model() -> None:
    """A well-formed JSON response is parsed into the requested schema."""
    client, fake, _ = make_client('{"id": "t-1", "value": 7}')
    out = client.generate("prompt", _Thing, "audit-menu@v1")

    assert isinstance(out, _Thing)
    assert out.id == "t-1"
    assert out.value == 7
    assert len(fake.calls) == 1


def test_generate_retries_once_on_invalid_json() -> None:
    """Schema validation fails on the first attempt and succeeds on the retry."""
    client, fake, _ = make_client(
        ["{not valid json", '{"id": "t-2", "value": 3}'],
    )
    out = client.generate("prompt", _Thing, "audit-menu@v1")

    assert isinstance(out, _Thing)
    assert out.id == "t-2"
    assert len(fake.calls) == 2  # original + one retry


def test_generate_raises_vertex_error_when_schema_always_invalid() -> None:
    """Invalid JSON on every attempt raises a typed VertexError, not a raw exception."""
    client, fake, _ = make_client(["{bad", "{also bad"])
    try:
        client.generate("prompt", _Thing, "audit-menu@v1")
    except VertexError as exc:
        assert "schema validation" in exc.detail
        assert exc.model_id == "gemini-2.5-flash"
        assert exc.prompt_version == "audit-menu@v1"
        assert exc.cause is not None
    else:  # pragma: no cover - the try must raise
        raise AssertionError("expected VertexError")


def test_generate_raises_vertex_error_on_transport_failure_without_retry() -> None:
    """A model that throws transports the failure as VertexError with no retry."""

    calls = 0

    def boom(model: str, contents: str, config: object) -> object:
        nonlocal calls
        calls += 1
        raise ConnectionError("network down")

    client, fake, _ = make_client("{}")
    fake.generate_content = boom  # type: ignore[method-assign]
    try:
        client.generate("prompt", _Thing, "audit-menu@v1")
    except VertexError as exc:
        assert "network down" in exc.detail
        assert calls == 1  # no retry on transport failure
    else:  # pragma: no cover
        raise AssertionError("expected VertexError")


def test_generate_logs_model_region_and_prompt_version(caplog) -> None:
    """Logs include model, region, and prompt version (never credentials)."""
    client, fake, settings = make_client('{"id": "x", "value": 1}')
    with caplog.at_level("INFO"):
        client.generate("prompt", _Thing, "draft-door-pitch@v1")

    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "gemini-2.5-flash" in messages
    assert settings.gcp_location in messages
    assert "draft-door-pitch@v1" in messages
    # No credential material is logged.
    for rec in caplog.records:
        for key in ("api_key", "credentials", "service_account", "private_key", "token"):
            assert key not in rec.getMessage().lower()


def test_generate_sets_timeout_and_max_output_tokens() -> None:
    """The request carries the configured timeout (ms) and output cap."""
    client, fake, _ = make_client('{"id": "x", "value": 1}', model_timeout_seconds=7.5)
    client.generate("prompt", _Thing, "audit-menu@v1")

    config = fake.calls[0][2]
    assert config.http_options.timeout == 7500
    assert config.max_output_tokens == 2048


def test_generate_grounded_attaches_the_search_tool() -> None:
    """Grounding is opt-in and sends the Google Search tool."""
    client, fake, _ = make_client('{"id": "x", "value": 1}')
    client.generate("prompt", _Thing, "audit-menu@v1", grounded=True)

    config = fake.calls[0][2]
    assert config.tools is not None
    assert config.tools[0].google_search is not None


def test_generate_ungrounded_has_no_tools() -> None:
    """A draft call does not pay for Search grounding."""
    client, fake, _ = make_client('{"id": "x", "value": 1}')
    client.generate("prompt", _Thing, "audit-menu@v1")

    assert fake.calls[0][2].tools is None


def test_gemini_api_key_builds_a_studio_client(monkeypatch) -> None:
    """A Studio key is passed as api_key and does not open a Vertex client."""
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.models = object()

    monkeypatch.setattr("karum_service_ai.vertex.client.genai.Client", _Client)
    settings = Settings(gemini_api_key="studio-key", gcp_project_id="karum-512016")
    VertexClient(settings)

    assert captured == {"api_key": "studio-key"}


def test_empty_gemini_key_keeps_the_vertex_client(monkeypatch) -> None:
    """With no Studio key, calls still go through Vertex on the Cloud project."""
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.models = object()

    monkeypatch.setattr("karum_service_ai.vertex.client.genai.Client", _Client)
    settings = Settings(
        gemini_api_key="", gcp_project_id="karum-512016", gcp_location="us-central1"
    )
    VertexClient(settings)

    assert captured["vertexai"] is True
    assert captured["project"] == "karum-512016"
    assert captured["location"] == "us-central1"


def test_error_code_enum_matches_proto_spelling() -> None:
    """The Pydantic ErrorCode mirrors karum.common.v1.ErrorCode wire spelling."""
    assert ErrorCode.UNSPECIFIED == "ERROR_CODE_UNSPECIFIED"
    assert ErrorCode.UPSTREAM == "ERROR_CODE_UPSTREAM"
    assert ModelAttribution(
        model_id="gemini-2.5-flash",
        prompt_version="v1",
        evidence_ids=["ev-1"],
        ungrounded=False,
    ).model_dump(by_alias=True)["evidenceIds"] == ["ev-1"]
