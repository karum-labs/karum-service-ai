"""Model backends for service-ai (PLAN.md §71).

A backend is the one thing that actually talks to an LLM. ``VertexClient``
owns retries, timeouts, fixture replay, and schema validation; it delegates
the raw call to a backend. Swapping providers is therefore a matter of
building a different backend from ``Settings`` — the agents and the gRPC
servicer never change.

Backends return the model's raw JSON text; parsing and Pydantic validation
stay in ``VertexClient`` so every backend is held to the same contract.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from karum_service_ai.core.config import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelBackend(Protocol):
    """The minimal surface a model provider must implement."""

    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        grounded: bool = False,
        timeout_ms: int = 30000,
        max_tokens: int = 2048,
    ) -> str: ...


class GeminiBackend:
    """The original google-genai path, including Google Search grounding."""

    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self._settings = settings
        self._model = model
        self._client: Any | None = None
        if model is None:
            self._build()

    def _build(self) -> None:
        from google import genai

        if self._settings.gemini_api_key:
            client = genai.Client(api_key=self._settings.gemini_api_key)
        else:
            if not self._settings.gcp_project_id:
                raise RuntimeError("no model credential")
            client = genai.Client(
                vertexai=True,
                project=self._settings.gcp_project_id,
                location=self._settings.gcp_location,
            )
        self._client = client
        self._model = client.models

    @property
    def model_id(self) -> str:
        return self._settings.gemini_model

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        grounded: bool = False,
        timeout_ms: int = 30000,
        max_tokens: int = 2048,
        request_payload: Any = None,
    ) -> str:
        from google import genai
        from google.genai import types as genai_types

        tools = None
        if grounded:
            tools = [genai_types.Tool(google_search=genai_types.GoogleSearch())]
        response = self._model.generate_content(
            model=self.model_id,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=max_tokens,
                http_options=genai_types.HttpOptions(timeout=timeout_ms),
                tools=tools,
            ),
        )
        return getattr(response, "text", "") or ""


class LangChainBackend:
    """A provider-agnostic backend built on LangChain chat models."""

    _PROVIDER_DEFAULTS: dict[str, str] = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-latest",
        "google": "gemini-2.0-flash",
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._provider = (settings.provider or "gemini").lower()
        if self._provider == "gemini":
            self._provider = "google"
        self._model_name = settings.model or self._PROVIDER_DEFAULTS.get(self._provider, "")
        if not self._model_name:
            raise RuntimeError(f"no default model for provider {self._provider!r}; set AI_MODEL")
        self._chat = self._build_chat()

    def _build_chat(self) -> Any:
        if self._provider in ("google", "gemini"):
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self._model_name,
                google_api_key=self._settings.gemini_api_key or None,
                temperature=0,
            )
        if self._provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=self._model_name, api_key=self._settings.openai_api_key, temperature=0)
        if self._provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=self._model_name, api_key=self._settings.anthropic_api_key, temperature=0)
        if self._provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=self._model_name, base_url=self._settings.ollama_base_url or None)
        raise RuntimeError(f"unknown provider {self._provider!r}")

    @property
    def model_id(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        grounded: bool = False,
        timeout_ms: int = 30000,
        max_tokens: int = 2048,
        request_payload: Any = None,
    ) -> str:
        if grounded:
            logger.warning(
                "langchain.generate.grounding_unsupported provider=%s; running ungrounded",
                self._provider,
            )
        chain = self._chat.with_structured_output(schema)
        result = chain.invoke(prompt)
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        return str(result)


class FixtureBackend:
    """Replays the recorded pack; implements the same protocol."""

    def __init__(self, transport: Any) -> None:
        self._transport = transport
        self._model_id = "fixture"

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        grounded: bool = False,
        timeout_ms: int = 30000,
        max_tokens: int = 2048,
        request_payload: Any = None,
    ) -> str:
        obj = self._transport.generate(schema, request_payload)
        if isinstance(obj, BaseModel):
            return obj.model_dump_json()
        return str(obj)


def build_backend(settings: Settings, model: Any | None = None) -> ModelBackend:
    """Construct the right backend for ``settings``.

    ``model`` is an injection seam for tests (a fake google-genai model).
    """
    # Test injection: a fake google-genai model forces the live Gemini path
    # regardless of mode, preserving the historical make_client contract.
    if model is not None:
        return GeminiBackend(settings, model=model)

    if settings.mode.value == "fixture":
        # Imported lazily to avoid a circular import.
        from karum_service_ai.vertex.fixtures import FixturePack, resolve_fixtures_dir
        from karum_service_ai.vertex.transport import RecordedModelTransport

        pack = FixturePack(resolve_fixtures_dir(settings.ai_fixtures_dir))
        return FixtureBackend(RecordedModelTransport(pack))

    provider = (settings.provider or "gemini").lower()
    if provider == "gemini":
        return GeminiBackend(settings)
    return LangChainBackend(settings)
