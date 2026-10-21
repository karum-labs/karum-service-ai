# karum-service-ai

The **agent & synthesis service** for [Karum](https://github.com/karum-labs/karum-infra#readme). This microservice performs live web verification with Google Search grounding and generates the judge-facing sales deliverables. The Go gateway calls it over gRPC.

Part of the five-repo Karum layout:

| Repo | Role |
| --- | --- |
| [`karum-infra`](https://github.com/karum-labs/karum-infra) | Shared protos, Docker Compose, Cloud Run manifests |
| [`karum-client`](https://github.com/karum-labs/karum-client) | Flutter UI (web / desktop) |
| [`karum-server`](https://github.com/karum-labs/karum-server) | Go + Gin API gateway |
| [`karum-service-taste`](https://github.com/karum-labs/karum-service-taste) | Qloo integration, caching, taste scoring |
| **`karum-service-ai`** (this repo) | Gemini (AI Studio or Vertex AI), search grounding, pitch drafting |

## What it does

- **Live menu verification** — calls the configured Gemini Flash model with Google Search Grounding to check whether target venues already stock a product or its competitors.
- **Pitch & battlecard generation** — ingests venue context alongside Qloo's cultural tags to draft customized sales scripts.
- **Buyer persona mapping** — infers the beverage director / GM profile for a style of venue.
- **Grounded retail check** — demotes online-only brands and attaches the grounding source as evidence.

The gRPC RPCs (`AuditMenu`, `DraftDoorPitch`, `LlmOnlyBaseline`) are defined in the shared protos in [`karum-infra`](https://github.com/karum-labs/karum-infra). Every response carries `model_id`, `prompt_version`, and cited evidence ids so the gateway can persist provenance without parsing prose.

## Run path

Requires **Python 3.13+** and [`uv`](https://docs.astral.sh/uv/) (see the version pin in [`karum-infra`](https://github.com/karum-labs/karum-infra)).

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
| `AI_PORT` | `8082` | HTTP listen port; `/health` is served here |
| `AI_GRPC_PORT` | `9092` | gRPC listen port the gateway dials |
| `AI_MODE` | `fixture` | `live`, `fixture`, or `degraded`; reported on `/health` |
| `AI_FIXTURES_DIR` | `../karum-infra/fixtures/ai` | Recorded pack replayed in fixture mode; relative paths resolve against this repo's root |
| `AI_GEMINI_API_KEY` | _(empty)_ | Google AI Studio key for Gemini calls — **never commit** (`GEMINI_API_KEY` still works) |
| `AI_GCP_PROJECT_ID` | _(empty)_ | Vertex project, used only when `AI_GEMINI_API_KEY` is empty |
| `AI_GCP_LOCATION` | `us-central1` | GCP region |
| `AI_GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model id |
| `AI_MAX_OUTPUT_TOKENS` | `2048` | Output cap per draft |
| `AI_PROBE_BUDGET` | `3` | Bounded follow-up taste probes per run |

Every variable is namespaced with `AI_`. The bare spellings (`PORT`, `MODE`,
`LOG_LEVEL`) are not read, so a stray variable in the environment cannot change
how the service behaves.

### Health

`GET /health` reports the service's operational mode:

- `fixture` — model calls replay the recorded pack under `AI_FIXTURES_DIR` (the pack ships in `karum-infra`); no network call and no credential is used, and a request with no recording fails as a typed upstream error rather than a guess
- `live` — a Gemini API key (AI Studio) or Vertex credentials are present and calls go to the model
- `degraded` — calls go to the model, and the service labels itself degraded for the client banner

The `status` field is an honest read of the configured mode, not a constant: it
is `degraded` when fixture mode has no recorded pack or when a live service has
neither a Studio key nor a project credential, and `ok` otherwise. The HTTP
status stays 200 in both cases, so a liveness probe does not fail over a
configuration gap that a per-request error would explain.

### Development

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
uv run pytest          # tests
```

### Docker

```bash
docker build -t karum-service-ai .
docker run -p 8082:8082 -e AI_GEMINI_API_KEY=... karum-service-ai
```

## Repository layout

```
src/karum_service_ai/
  main.py               FastAPI app factory
  grpcserv.py           AgentService gRPC servicer
  api/routes/health.py  GET /health
  agent/audit.py        menu audit and shelf scoring
  agent/draft.py        pitch, battlecard, and LLM-only baseline
  agent/prompting.py    prompt assembly and inert-name handling
  core/config.py        environment-driven settings
  core/logging.py       structured logging with secret redaction
  vertex/client.py      Gemini client (AI Studio or Vertex AI)
  vertex/models.py      Pydantic response schemas
tests/                  pytest suite
```

## Ownership

Karum is a two-person build (record: [`DECISIONS.md`](https://github.com/karum-labs/karum-infra/blob/main/DECISIONS.md)). **Side A** owns taste, score, agent, and eval; **side B** owns UI, map, drafts, deploy, and Devpost. This repo is **side A** — the agent loop, probes, autopilot, and draft RPCs. The draft copy judges read is side B's to review, so prompt wording and length caps get a pass from over there.

## Licence

MIT. See [`LICENSE`](LICENSE).
