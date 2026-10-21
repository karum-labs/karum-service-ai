"""The recorded model pack: the offline path for the AI service (PLAN.md §4, §18).

The pack lives in the karum-infra repo under ``fixtures/ai/``: a manifest plus
one JSON file per recorded model answer. Nothing in it is secret — the recorded
answers are the model's own output, the same text the live path would have
returned.

Fixture mode replays *exactly* those recordings. The pack indexes them by a
deterministic request key — the RPC method plus a stable digest of the
request payload (:func:`request_key`) — and a request that matches nothing
raises :class:`FixtureMissError`. Replay never invents an answer.

One narrow exception exists, and it is narrow on purpose. A recorded answer
belongs to one request, but a run's request differs in the details the
recording cannot know (the door ids the taste service issued, the evidence set
of the run). A recording therefore also declares a ``match`` — a small set of
summary fields (city, brand, door) — and a request whose summary satisfies it
is answered by that recording, but only when exactly one recording of that
method matches. Two candidates mean the request cannot be answered by replay,
and a guess is worse than a miss.

Manifest schema (``fixture_version`` 1)::

    {
      "fixture_version": 1,
      "recorded": "2026-03-04T11:20:00Z",
      "entries": [
        {
          "id": "audit-menu-austin-whislers",
          "method": "audit_menu",
          "file": "audit-menu-austin-whislers.json",
          "key": "audit_menu:0123456789abcdef",
          "request": {...},        # the payload the key is computed from
          "match": {"city": "Austin", "door": "Whisler's"}
        }
      ]
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from karum_service_ai.vertex.errors import FixtureError, FixtureMissError

logger = logging.getLogger(__name__)

#: Schema version carried by the manifest and by every recorded response.
FIXTURE_VERSION = 1

#: The repo root for this checkout (…/karum-service-ai).
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The manifest file inside the pack directory.
MANIFEST_NAME = "manifest.json"


def resolve_fixtures_dir(configured: Path) -> Path:
    """Absolute path of the pack; relative settings resolve against the repo root."""
    return configured if configured.is_absolute() else (REPO_ROOT / configured).resolve()


def canonical_payload(payload: Mapping[str, Any]) -> str:
    """The stable JSON rendering of a request payload, used for the request key."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def request_key(method: str, payload: Mapping[str, Any]) -> str:
    """The deterministic key of one model call: the method and a payload digest.

    The same request always produces the same key, so a recording can be found
    again after a restart (PLAN.md §4).
    """
    digest = hashlib.sha256(canonical_payload(payload).encode("utf-8")).hexdigest()
    return f"{method}:{digest[:16]}"


def _summary_matches(spec: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """Whether a request summary satisfies a recording's ``match`` spec."""
    return all(key in summary and summary[key] == value for key, value in spec.items())


@dataclass(frozen=True)
class FixtureEntry:
    """One indexed recording."""

    id: str
    method: str
    path: Path
    key: str
    request: dict[str, Any]
    match: dict[str, Any]
    response: dict[str, Any]


class FixturePack:
    """The indexed fixture pack: request key -> recorded answer."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._entries: dict[str, FixtureEntry] = {}
        self._by_method: dict[str, list[FixtureEntry]] = {}
        self._load()

    @property
    def root(self) -> Path:
        """The pack directory (the one holding the manifest)."""
        return self._root

    @property
    def entries(self) -> dict[str, FixtureEntry]:
        """The indexed recordings, keyed by request key."""
        return self._entries

    def lookup(self, method: str, payload: Mapping[str, Any]) -> FixtureEntry | None:
        """The recording for this exact request, if any."""
        return self._entries.get(request_key(method, payload))

    def match(
        self,
        method: str,
        payload: Mapping[str, Any],
        summary: Mapping[str, Any] | None = None,
    ) -> FixtureEntry:
        """The recording that answers this request.

        The exact recording wins. When the request has none, a recording of the
        same method whose ``match`` spec the request satisfies is served — but
        only when exactly one such recording exists. Two candidates mean the
        request cannot be answered by replay, and a guess is worse than a miss.
        """
        entry = self.lookup(method, payload)
        if entry is not None:
            return entry
        wanted = dict(summary or {})
        candidates = [
            candidate
            for candidate in self._by_method.get(method, [])
            if _summary_matches(candidate.match, wanted)
        ]
        if len(candidates) != 1:
            raise FixtureMissError(
                f"fixture mode has no recording for {method} request "
                f"{request_key(method, payload)}; the pack at {self._root} holds "
                f"{len(self._entries)} recordings"
            )
        entry = candidates[0]
        logger.warning(
            "fixtures.substituted method=%s entry=%s request_key=%s",
            method,
            entry.id,
            request_key(method, payload),
        )
        return entry

    def _load(self) -> None:
        manifest_path = self._root / MANIFEST_NAME
        if not self._root.is_dir():
            raise FixtureError(
                f"fixture pack not found at {self._root}; set AI_FIXTURES_DIR or "
                "check out the recorded pack next to the service"
            )
        if not manifest_path.is_file():
            raise FixtureError(f"fixture pack at {self._root} has no {MANIFEST_NAME}")
        manifest = self._read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise FixtureError(f"fixture manifest {manifest_path} is not an object")
        version = manifest.get("fixture_version")
        if version != FIXTURE_VERSION:
            raise FixtureError(
                f"fixture manifest {manifest_path} has version {version!r}, "
                f"expected {FIXTURE_VERSION}"
            )
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise FixtureError(f"fixture manifest {manifest_path} has no entries list")
        for item in entries:
            entry = self._read_entry(self._root, item)
            if entry.key in self._entries:
                raise FixtureError(
                    f"fixture pack holds two recordings for {entry.key}: "
                    f"{self._entries[entry.key].path} and {entry.path}"
                )
            self._entries[entry.key] = entry
            self._by_method.setdefault(entry.method, []).append(entry)

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FixtureError(f"fixture {path} is unreadable: {exc}") from exc

    def _read_entry(self, root: Path, item: Any) -> FixtureEntry:
        if not isinstance(item, dict):
            raise FixtureError(f"fixture manifest holds a non-object entry: {item!r}")
        entry_id = item.get("id")
        method = item.get("method")
        file = item.get("file")
        request = item.get("request")
        if not isinstance(entry_id, str) or not entry_id:
            raise FixtureError(f"fixture entry has no id: {item!r}")
        if not isinstance(method, str) or not method:
            raise FixtureError(f"fixture entry {entry_id} has no method")
        if not isinstance(file, str) or not file:
            raise FixtureError(f"fixture entry {entry_id} has no file")
        if not isinstance(request, dict):
            raise FixtureError(f"fixture entry {entry_id} has no request payload")
        path = root / file
        document = self._read_json(path)
        if not isinstance(document, dict):
            raise FixtureError(f"fixture {path} is not an object")
        if document.get("fixture_version") != FIXTURE_VERSION:
            raise FixtureError(
                f"fixture {path} has version {document.get('fixture_version')!r}, "
                f"expected {FIXTURE_VERSION}"
            )
        response = document.get("response")
        if not isinstance(response, dict):
            raise FixtureError(f"fixture {path} has no response object")
        key = request_key(method, request)
        declared = item.get("key")
        if isinstance(declared, str) and declared and declared != key:
            raise FixtureError(
                f"fixture entry {entry_id} declares key {declared!r} but its request "
                f"payload hashes to {key!r}"
            )
        match = item.get("match")
        return FixtureEntry(
            id=entry_id,
            method=method,
            path=path,
            key=key,
            request=request,
            match=dict(match) if isinstance(match, dict) else {},
            response=response,
        )
