# karum-service-ai

The **agent & synthesis service** for [Karum](https://github.com/kaarum/karum-infra#readme). This microservice performs live web verification with Google Search grounding and generates the judge-facing sales deliverables. The Go gateway calls it over gRPC.

Part of the five-repo Karum layout:

| Repo | Role |
| --- | --- |
| [`karum-infra`](https://github.com/kaarum/karum-infra) | Shared protos, Docker Compose, Cloud Run manifests |
| [`karum-client`](https://github.com/kaarum/karum-client) | Flutter UI (web / desktop) |
| [`karum-server`](https://github.com/kaarum/karum-server) | Go + Gin API gateway |
| [`karum-service-taste`](https://github.com/kaarum/karum-service-taste) | Qloo integration, caching, taste scoring |
| **`karum-service-ai`** (this repo) | Vertex AI, search grounding, pitch drafting |

## What it does

- **Live menu verification** — calls Gemini 2.5 Flash with Google Search Grounding to check whether target venues already stock a product or its competitors.
- **Pitch & battlecard generation** — ingests venue context alongside Qloo's cultural tags to draft customized sales scripts.
- **Buyer persona mapping** — infers the beverage director / GM profile for a style of venue.
- **Grounded retail check** — demotes online-only brands and attaches the grounding source as evidence.

The gRPC RPCs (`ProbeShortlist`, `LlmOnlyBaseline`, `DraftMemo`, `DraftOutreach`, `DraftListing`, `CheckRetailPresence`) are defined in the shared protos in [`karum-infra`](https://github.com/kaarum/karum-infra). Every response carries `model_id`, `prompt_version`, and cited evidence ids so the gateway can persist provenance without parsing prose.

## Run path

Requires **Python 3.13+** and [`uv`](https://docs.astral.sh/uv/) (see the version pin in [`karum-infra`](https://github.com/kaarum/karum-infra)).

```bash
# Install dependencies and run the service.
uv run uvicorn karum_service_ai.main:app --reload --port 8082
```

The service listens on **`:8082`** by default. Confirm it is healthy:

```bash
curl http://localhost:8082/health
```

```json
{
  "service": "service-ai",
  "status": "ok",
  "mode": "fixture",
  "uptime_seconds": 0,
  "time": "2026-10-12T00:00:00+00:00"
}
```

Interactive API docs are served at `http://localhost:8082/docs`.

### Configuration

Every setting is read from the environment (optionally via a `.env` file). All values have working defaults, so the service boots in fixture mode with no configuration. The full list is in [`.env.example`](.env.example).

Key variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_PORT` | `8082` | HTTP/gRPC listen port |
| `AI_MODE` | `fixture` | `live`, `fixture`, or `degraded`; reported on `/health` |
| `AI_GCP_PROJECT_ID` | _(empty)_ | GCP project id — **never commit** |
| `AI_GCP_LOCATION` | `us-central1` | GCP region |
| `AI_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model id |
| `AI_MAX_OUTPUT_TOKENS` | `2048` | Output cap per draft |
| `AI_PROBE_BUDGET` | `3` | Bounded follow-up taste probes per run |

### Health

`GET /health` reports the service's operational mode:

- `fixture` — replaying recorded model payloads, no outbound Vertex calls
- `live` — GCP credentials present
- `degraded` — Vertex unavailable; the gateway falls back to its deterministic autopilot

### Development

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
uv run pytest          # tests
```

### Docker

```bash
docker build -t karum-service-ai .
docker run -p 8082:8082 -e AI_GCP_PROJECT_ID=... karum-service-ai
```

## Repository layout

```
src/karum_service_ai/
  main.py               FastAPI app factory
  api/routes/health.py  GET /health
  core/config.py        environment-driven settings
  core/logging.py       structured logging with secret redaction
  vertex/client.py      Gemini / Vertex client (stub)
tests/                  pytest suite
```

## Licence

MIT. See [`LICENSE`](LICENSE).
