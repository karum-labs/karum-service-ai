"""Model backends for service-ai (PLAN.md §71)."""
from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel

from karum_service_ai.core.config import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelBackend(Protocol):
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
        request_payload: Any = None,
    ) -> str: ...


class GeminiBackend:
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

    @property
    def last_usage(self) -> None:
        return None

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
        if settings.langchain_tracing:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        self._chat = self._build_chat()
        self._last_usage = None

    def _build_chat(self) -> Any:
        primary = self._build_one(self._provider, self._model_name)
        fallbacks = []
        for name in (self._settings.fallback_providers or "").split(","):
            name = name.strip().lower()
            if not name or name == self._provider:
                continue
            model = self._PROVIDER_DEFAULTS.get(name)
            if not model:
                continue
            try:
                fallbacks.append(self._build_one(name, model))
            except Exception:  # noqa: BLE001
                logger.warning("langchain.fallback_unavailable provider=%s", name)
        if fallbacks:
            return primary.with_fallbacks(fallbacks)
        return primary

    def _build_one(self, provider: str, model: str) -> Any:
        if provider in ("google", "gemini"):
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=model, google_api_key=self._settings.gemini_api_key or None, temperature=0)
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=self._settings.openai_api_key,
                base_url=self._settings.openai_base_url or None,
                temperature=0,
            )
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model, api_key=self._settings.anthropic_api_key, temperature=0)
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=model, base_url=self._settings.ollama_base_url or None)
        raise RuntimeError(f"unknown provider {provider!r}")

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def last_usage(self) -> dict[str, int] | None:
        return getattr(self, "_last_usage", None)

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
        grounding_context = ""
        if grounded:
            urls = []
            try:
                for t in (request_payload or {}).get("targets", []) or []:
                    w = t.get("website") or (t.get("door") or {}).get("website") or ""
                    if w:
                        urls.append(w)
            except Exception:  # noqa: BLE001
                urls = []
            fetched = []
            for url in urls[:3]:
                try:
                    r = httpx.get(url, timeout=10.0, follow_redirects=True)
                    if r.status_code == 200:
                        fetched.append("--- %s ---\n%s" % (url, r.text[:4000]))
                except Exception:  # noqa: BLE001
                    logger.warning("langchain.grounding.fetch_failed url=%s", url)
            if fetched:
                grounding_context = "\n\nGrounding (fetched):\n" + "\n".join(fetched)
            else:
                logger.warning(
                    "langchain.generate.grounding_unsupported provider=%s; running ungrounded",
                    self._provider,
                )
        if grounding_context:
            prompt = prompt + grounding_context
        try:
            from langchain_core.callbacks import UsageMetadataCallbackHandler

            handler = UsageMetadataCallbackHandler()
            callbacks = [handler]
        except Exception:  # noqa: BLE001
            handler = None
            callbacks = None
        chain = self._chat.with_structured_output(schema)
        config = {"callbacks": callbacks} if callbacks else None
        result = chain.invoke(prompt, config=config) if config else chain.invoke(prompt)
        if handler is not None:
            try:
                usage = getattr(handler, "usage_metadata", None)
                if usage:
                    self._last_usage = {}
                    for k, v in usage.items():
                        if isinstance(v, dict):
                            self._last_usage[k] = v.get("total_tokens", 0)
            except Exception:  # noqa: BLE001
                self._last_usage = None
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        return str(result)


class OpenAIResponsesBackend:
    """Direct OpenAI SDK backend using ``client.responses.create()``.

    Used for OpenAI-compatible endpoints (Groq, Nebius, etc.) without LangChain.
    Configured via ``AI_PROVIDER=openai-responses``, ``AI_OPENAI_API_KEY``,
    ``AI_OPENAI_BASE_URL``, and ``AI_MODEL``.
    """

    _MODEL_DEFAULTS: dict[str, str] = {
        "openai-responses": "gpt-4o-mini",
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_name = settings.model or self._MODEL_DEFAULTS.get("openai-responses", "")
        if not self._model_name:
            raise RuntimeError("no default model for openai-responses; set AI_MODEL")
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or None,
            )
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("openai package not installed; run `uv add openai`") from exc
        self._last_usage: dict[str, int] | None = None

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def last_usage(self) -> dict[str, int] | None:
        return self._last_usage

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
        """Generate one structured response via the OpenAI Responses API.

        Returns the raw output text; the caller (VertexClient) validates it
        against ``schema`` with ``model_validate_json``.
        """
        if grounded:
            logger.warning(
                "openai.responses.grounding_unsupported provider=openai-responses; running ungrounded"
            )

        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "input": prompt,
        }
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens
        if timeout_ms:
            kwargs["timeout"] = timeout_ms / 1000.0

        response = self._client.responses.create(**kwargs)
        text = getattr(response, "output_text", "") or ""
        # Strip markdown code fences the model often wraps JSON in
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # Drop the opening fence (```json or ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Drop the closing fence if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        # Sanitize fields the service overwrites anyway (provenance is the
        # service's, not the model's). errorCode must be a valid ErrorCode
        # enum value; errorMessage defaults to empty.
        try:
            import json as _json

            payload = _json.loads(text)
            if isinstance(payload, dict):
                ec = payload.get("errorCode", "")
                valid_codes = {
                    "ERROR_CODE_UNSPECIFIED",
                    "ERROR_CODE_UNRESOLVED",
                    "ERROR_CODE_EMPTY",
                    "ERROR_CODE_RATE_LIMITED",
                    "ERROR_CODE_UPSTREAM",
                    "ERROR_CODE_VALIDATION",
                }
                if ec not in valid_codes:
                    payload["errorCode"] = "ERROR_CODE_UNSPECIFIED"
                payload.setdefault("errorMessage", "")
                text = _json.dumps(payload)
        except Exception:  # noqa: BLE001 - pass raw text if not JSON
            pass

        usage = getattr(response, "usage", None)
        if usage is not None:
            self._last_usage = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if hasattr(usage, key):
                    self._last_usage[key] = int(getattr(usage, key) or 0)

        return text


class FixtureBackend:
    def __init__(self, transport: Any) -> None:
        self._transport = transport
        self._model_id = "fixture"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def last_usage(self) -> None:
        return None

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


def build_backend(settings: Settings, model: Any | None = None) -> "ModelBackend":
    if model is not None:
        return GeminiBackend(settings, model=model)
    if settings.mode.value == "fixture":
        from karum_service_ai.vertex.fixtures import FixturePack, resolve_fixtures_dir
        from karum_service_ai.vertex.transport import RecordedModelTransport

        pack = FixturePack(resolve_fixtures_dir(settings.ai_fixtures_dir))
        return FixtureBackend(RecordedModelTransport(pack))
    provider = (settings.provider or "gemini").lower()
    if provider == "gemini":
        return GeminiBackend(settings)
    if provider == "openai-responses":
        return OpenAIResponsesBackend(settings)
    return LangChainBackend(settings)