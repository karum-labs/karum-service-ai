"""Fixture replay tests (PLAN.md §4, §18).

The recorded pack is the offline demo path. Replay must answer with the
recorded content, verify every declared key, refuse an ambiguous match, and
fail as a typed miss — never invent an answer. The shipped pack (derived from
the contract campaign fixtures in ``karum-infra``) is replayed end to end when
its checkout is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from karum_service_ai.agent.audit import audit_menu
from karum_service_ai.agent.draft import draft_door_pitch, llm_only_baseline
from karum_service_ai.core.config import Mode, Settings
from karum_service_ai.vertex.client import VertexClient
from karum_service_ai.vertex.errors import FixtureError, FixtureMissError, VertexError
from karum_service_ai.vertex.fixtures import (
    FixturePack,
    request_key,
    resolve_fixtures_dir,
)
from karum_service_ai.vertex.models import (
    BrandContext,
    DoorContext,
    DraftDoorPitchResponse,
    EntityRef,
    ErrorCode,
    Incumbent,
    MenuAuditResult,
    Sku,
    SubScore,
    Tag,
)

#: The pack a default-configured service replays. The infrastructure repo is a
#: sibling checkout; when it is absent (a CI run of this repo alone), the
#: shipped-pack tests skip rather than guess.
SHIPPED_PACK = resolve_fixtures_dir(Settings(mode=Mode.FIXTURE).ai_fixtures_dir)

shipped_pack_present = pytest.mark.skipif(
    not (SHIPPED_PACK / "manifest.json").is_file(),
    reason="karum-infra fixtures are not checked out beside this repo",
)


def _write_pack(root: Path, entries: list[dict], files: dict[str, dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, document in files.items():
        (root / name).write_text(json.dumps(document), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"fixture_version": 1, "entries": entries}), encoding="utf-8"
    )
    return root


def _envelope(response: dict) -> dict:
    return {"fixture_version": 1, "response": response}


def _draft_document(door: str) -> dict:
    return {
        "script": "text",
        "recommendedSku": {"id": "sk-1", "name": "SKU"},
        "attribution": {"modelId": "x", "promptVersion": "draft-door-pitch@v1"},
    }


def _draft_entry(door: str, match: dict) -> tuple[dict, dict]:
    request = {
        "brand": {"name": "Ilegal Mezcal"},
        "city": "Austin",
        "door": {"audit": {"door": {"name": door}}},
    }
    name = f"draft-{door.lower().replace(' ', '-')}.json"
    entry = {
        "id": f"draft-{door}",
        "method": "draft_door_pitch",
        "file": name,
        "key": request_key("draft_door_pitch", request),
        "request": request,
        "match": match,
    }
    return entry, {name: _envelope(_draft_document(door))}


def _fixture_client(pack_root: Path) -> VertexClient:
    return VertexClient(Settings(mode=Mode.FIXTURE, ai_fixtures_dir=pack_root))


def _draft_payload(door: str) -> dict:
    return {
        "brand": {"name": "Ilegal Mezcal"},
        "city": "Austin",
        "mode": "wholesale",
        "door": {"audit": {"door": {"name": door}}},
    }


def test_request_key_is_stable_and_payload_sensitive() -> None:
    payload = {"targets": [{"door": {"name": "Whisler's", "city": "Austin"}}]}

    assert request_key("audit_menu", payload) == request_key("audit_menu", payload)
    assert request_key("audit_menu", payload) != request_key("draft_door_pitch", payload)
    assert request_key("audit_menu", payload) != request_key("audit_menu", {"targets": []})


def test_a_manifest_key_that_drifted_is_rejected(tmp_path: Path) -> None:
    root = _write_pack(
        tmp_path / "pack",
        entries=[
            {
                "id": "audit-1",
                "method": "audit_menu",
                "file": "audit-1.json",
                "key": "audit_menu:0000000000000000",
                "request": {"targets": []},
                "match": {},
            }
        ],
        files={"audit-1.json": _envelope({"results": []})},
    )

    with pytest.raises(FixtureError, match="declares key"):
        FixturePack(root)


def test_a_pack_without_a_manifest_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    with pytest.raises(FixtureError, match="no manifest.json"):
        FixturePack(root)


def test_an_ambiguous_match_is_a_miss(tmp_path: Path) -> None:
    """Two recordings that both describe the request: a guess is worse than a miss."""
    entries = []
    files: dict[str, dict] = {}
    for door in ("Whisler's", "Here Nor There"):
        entry, file = _draft_entry(door, match={"city": "Austin", "brand": "Ilegal Mezcal"})
        entries.append(entry)
        files.update(file)
    client = _fixture_client(_write_pack(tmp_path / "pack", entries, files))

    with pytest.raises(FixtureMissError):
        client.generate(
            "prompt",
            DraftDoorPitchResponse,
            "draft-door-pitch@v1",
            request_payload=_draft_payload("Midnight Cowboy"),
        )


def test_a_missing_recording_is_a_typed_miss(tmp_path: Path) -> None:
    """No recording for this door: a FixtureMissError, and the agent degrades typed."""
    entry, files = _draft_entry(
        "Whisler's", match={"city": "Austin", "brand": "Ilegal Mezcal", "door": "Whisler's"}
    )
    client = _fixture_client(_write_pack(tmp_path / "pack", [entry], files))

    with pytest.raises(VertexError, match="no recording"):
        client.generate(
            "prompt",
            DraftDoorPitchResponse,
            "draft-door-pitch@v1",
            request_payload=_draft_payload("La Condesa"),
        )

    brand = BrandContext(name="Ilegal Mezcal")
    door = DoorContext(
        shared_tags=[],
        affinity=SubScore(name="cultural_affinity", value=1.0, raw_value=0.9),
        audit=MenuAuditResult(
            door=EntityRef(id="d-9", type="urn:entity:place", name="La Condesa", city="Austin"),
            succeeded=True,
            shelf_unknown=True,
        ),
    )
    response = draft_door_pitch(client, brand, door, "Austin", evidence_ids=[])
    assert response.error_code == ErrorCode.UPSTREAM
    assert "no recording" in response.error_message


AUSTIN_TARGETS = [
    {
        "door": {
            "id": "d-1",
            "type": "urn:entity:place",
            "name": "Whisler's",
            "city": "Austin",
        },
        "website": "https://www.whislersaustin.com/menu",
        "taste_tags": [{"urn": "urn:tag:genre:place:restaurant:bar", "label": "Bars"}],
        "skip_candidate": False,
    },
    {
        "door": {
            "id": "d-2",
            "type": "urn:entity:place",
            "name": "Here Nor There",
            "city": "Austin",
        },
        "website": "",
        "taste_tags": [],
        "skip_candidate": False,
    },
    {
        "door": {
            "id": "d-3",
            "type": "urn:entity:place",
            "name": "Midnight Cowboy",
            "city": "Austin",
        },
        "website": "",
        "taste_tags": [],
        "skip_candidate": False,
    },
]


@shipped_pack_present
def test_the_shipped_pack_replays_the_austin_audit() -> None:
    client = VertexClient(Settings(mode=Mode.FIXTURE))
    response = audit_menu(
        client,
        AUSTIN_TARGETS,
        qloo_incumbent_names={"d-1": ["Ilegal Mezcal", "Aesop", "Khruangbin"]},
    )

    assert client.transport_name == "fixture"
    assert [result.door.name for result in response.results] == ["Whisler's", "Here Nor There"]
    first, second = response.results
    # The request's door id wins over the id the recording echoes (K-08).
    assert first.door.id == "d-1"
    assert first.incumbents[0].name == "Del Maguey Single Village Mezcal"
    assert first.competitor_count == 1
    assert first.shelf_opportunity == 0.75
    assert first.sources[0].url == "https://www.whislersaustin.com/menu"
    assert second.door.id == "d-2"
    assert second.shelf_unknown is True
    assert second.incumbents == []
    assert response.attribution.model_id == client.model_id
    assert response.error_code == ErrorCode.UNSPECIFIED


@shipped_pack_present
def test_the_shipped_pack_replays_the_austin_pitch() -> None:
    client = VertexClient(Settings(mode=Mode.FIXTURE))
    brand = BrandContext(
        name="Ilegal Mezcal",
        skus=[Sku(id="sk-ilegal-espadin-750", name="Ilegal Mezcal Espadín Joven 750ml")],
    )
    door = DoorContext(
        shared_tags=[Tag(urn="urn:tag:genre:place:restaurant:bar", label="Bars")],
        affinity=SubScore(name="cultural_affinity", value=1.0, raw_value=0.8655),
        audit=MenuAuditResult(
            door=EntityRef(id="d-1", type="urn:entity:place", name="Whisler's", city="Austin"),
            succeeded=True,
            shelf_unknown=False,
            incumbents=[Incumbent(name="Del Maguey Single Village Mezcal")],
            competitor_count=1,
            shelf_opportunity=0.75,
        ),
    )
    response = draft_door_pitch(
        client,
        brand,
        door,
        "Austin",
        mode="wholesale",
        evidence_ids=["ev-0001", "ev-0003", "ev-0004"],
    )

    assert "ranks first of 25" in response.script
    assert "ev-" not in response.script
    assert response.battlecard is not None
    assert response.battlecard.incumbent_name == "Del Maguey Single Village Mezcal"
    # The cited ev-0006 is outside the run's evidence set, so it is dropped
    # (§17) — the filter working on recorded content.
    assert response.battlecard.talking_points == [
        "Ilegal's taste overlap with this room beats the incumbent's (ev-0004)"
    ]
    assert response.recommended_sku.id == "sk-ilegal-espadin-750"
    assert response.mode_paragraph == ""
    assert response.error_code == ErrorCode.UNSPECIFIED
    assert response.attribution.model_id == client.model_id


@shipped_pack_present
def test_the_shipped_pack_replays_the_ungrounded_baseline() -> None:
    client = VertexClient(Settings(mode=Mode.FIXTURE))
    response = llm_only_baseline(client, BrandContext(name="Ilegal Mezcal"), "Austin")

    assert response.text.startswith("For a mezcal brand entering Austin")
    assert response.venue_names == ["La Condesa", "Suerte", "Tommy's Mexican Restaurant"]
    assert response.attribution.ungrounded is True
    assert response.attribution.evidence_ids == []
    assert response.error_code == ErrorCode.UNSPECIFIED
