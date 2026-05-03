# Career Agent

Google ADK career-intelligence agent that matches a candidate profile against job descriptions, scores fit across multiple dimensions, and generates a prioritised skill-development plan.

---

## Quick Start

```bash
cp .env.example .env
# Fill POSTGRES_PASSWORD, RABBITMQ_DEFAULT_PASS, and GOOGLE_API_KEY.
uv sync
docker compose up --build
```

The FastAPI app is available at `http://127.0.0.1:8080`. Docker Compose brings up PostgreSQL, RabbitMQ, a one-shot Alembic migration container, the API, and two parallel worker containers.

To also start the ADK interactive web UI on `http://127.0.0.1:8001`:

```bash
docker compose --profile adk up --build
```

To also start the optional LiteLLM proxy on port 4000:

```bash
docker compose --profile litellm up --build
```

For local development outside Docker:

```bash
uv run uvicorn main:app --reload       # API on :8000
uv run python -m modules.worker.run    # Worker process
uv run adk run agents/career_intelligence  # ADK interactive UI
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key used by ADK when not using Vertex AI. |
| `POSTGRES_PASSWORD` | Yes | PostgreSQL password. Docker Compose builds `DATABASE_URL` from `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`. Set `DATABASE_URL` directly for non-Compose runs. |
| `RABBITMQ_DEFAULT_PASS` | Yes | RabbitMQ password. Docker Compose builds `AMQP_URL` from `RABBITMQ_DEFAULT_USER` and `RABBITMQ_DEFAULT_PASS`. Set `AMQP_URL` directly for non-Compose runs. |
| `MATCH_QUEUE_NAME` | No | Durable RabbitMQ queue name. Defaults to `career.match_jobs`. |
| `GOOGLE_GENAI_USE_VERTEXAI` | No | Set to `true` to use Vertex AI instead of the public Gemini API. Requires `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. |
| `GITHUB_TOKEN` / `GITHUB_PAT_TOKEN` | No | GitHub API token for rate-limit headroom in `research_skill_resources`. |
| `GITHUB_API_BASE` | No | GitHub API base URL. Defaults to `https://api.github.com`. |
| `GITHUB_TIMEOUT_SECONDS` | No | Timeout for GitHub API calls. Defaults to `10`. |
| `RESOURCE_RESEARCH_TIMEOUT_SECONDS` | No | Timeout for the internal resource-research agent. Defaults to `30`. |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check. Returns `{"status": "ok"}`. |
| `POST` | `/api/v1/candidate` | Ingest a candidate resume. Accepts JSON `{"profile": {...}}`, `{"resume_text": "..."}`, or multipart `resume_file` (PDF or plain text). Extracts and stores the structured profile in PostgreSQL. |
| `POST` | `/api/v1/matches` | Submit up to 10 job descriptions (text or URL) for a stored candidate. Enqueues one agent run per JD and returns immediately with job IDs and `status: pending`. |
| `GET` | `/api/v1/matches/{id}` | Returns the status and full structured agent output for one job. Status values: `pending`, `processing`, `completed`, `failed`. |
| `GET` | `/api/v1/matches` | Paginated list of jobs. Requires `limit` and `offset`. Optional `status` filter. |
| `POST` | `/api/v1/matches/{id}/requeue` | Admin endpoint to reset a failed job to `pending` and re-enqueue it. |

---

## Framework Choice

This project uses **Google ADK** because the assignment requires runtime tool selection, typed session state persisted across tool calls, tool callbacks (before/after/error hooks), nested agent execution, and event streaming — all of which ADK provides without a custom ReAct loop.

What ADK gave that a plain LangChain chain would not:

- **Runtime tool sequencing** — the model decides which tools to call and in what order each run; no fixed DAG.
- **Session state** — `context.state` is a typed dict managed by the ADK runner and accessible inside every tool, so intermediate results (`last_requirements`, `last_score`, `resources_by_skill`) flow between tools without global variables.
- **Tool callbacks** — `before_tool_callback`, `after_tool_callback`, and `on_tool_error_callback` let the runtime intercept every call to populate `agent_trace` from real events, not from model text.
- **Nested agent execution** — `research_skill_resources` spins up a second inner ADK agent (`resource_research_agent`) that uses its own DB/GitHub tools. Events from the inner agent are forwarded into the outer stream so the full trace is observable.
- **Event streaming** — `Runner.run_async()` yields granular events (model response, tool call start, tool call result, state delta, completion). The worker consumes this stream to build the real-time `agent_trace` without buffering the full LLM response.

---

## Agent Architecture

```
Root Agent: career_intelligence
  ├── Model:     Google Gemini (via google-adk)
  ├── State:     AgentRunState (typed Pydantic model)
  ├── Tools:     extract_jd_requirements
  │              score_candidate_against_requirements
  │              prioritise_skill_gaps
  │              research_skill_resources  ──► Internal ADK agent
  │              get_curated_skill_resources             └── query_skill_resource_db
  │              finalize_match_output                   └── query_github_learning_resources
  └── Callbacks: before_model_callback                   └── query_github_repository_readme
                 before_tool_callback
                 after_tool_callback
                 on_tool_error_callback
```

The root agent runs inside `stream_career_match_events()` in `modules/agent_runtime.py`. Each match job creates one ADK session. The session state is initialised from a typed `AgentRunState` model and validated again by `finalize_match_output` before the result is accepted.

Expected tool-call sequence (the model decides at runtime, but this is the designed happy path):

1. `extract_jd_requirements` — parses the JD into structured requirements; writes `last_requirements` to state.
2. `score_candidate_against_requirements` — scores fit; writes `last_score` to state.
3. `prioritise_skill_gaps` — ranks gap skills by expected match improvement; writes `last_prioritized_skill_gaps` to state.
4. `research_skill_resources` — runs the inner research agent for each high-priority gap; writes to `resources_by_skill` in state.
5. `finalize_match_output` — validates and returns the final structured `MatchOutput`.

`agent_trace` is built from ADK callbacks and state counters, not fabricated from model text. Tool argument values and response bodies are not emitted in stream metadata; events expose keys and status only.

---

## Tools

### `extract_jd_requirements(url_or_text)`

Accepts raw JD text, URL strings, or LLM-supplied objects containing a `url`, `text`, or `job_description` field. URL content is fetched with Playwright and converted to a semantic page snapshot before LLM extraction. Schema-validates the output with `ExtractJDRequirementOutputSchema` before returning. Writes the result to `context.state["last_requirements"]`.

**Returns:** `required_skills[]`, `nice_to_have_skills[]`, `seniority_level`, `domain`, `responsibilities[]`, `confidence`, `confidence_score`.

**Failure handling:** Retries one malformed model output. If the second attempt also fails schema validation, surfaces a `RetryableModelOutputError` to ADK for the agent or runtime to handle.

---

### `score_candidate_against_requirements(candidate_profile, requirements)`

Accepts the candidate profile as a JSON string. `requirements` is optional — if omitted, the tool loads `context.state["last_requirements"]` set by `extract_jd_requirements`. Uses an LLM for semantic matching; applies a deterministic penalty model over the result to produce the exported confidence (see Confidence Heuristic below). Writes the result to `context.state["last_score"]`.

**Returns:** `overall_score` (0–100), `dimension_scores {skills, experience, seniority_fit}`, `matched_skills[]`, `gap_skills[]`, `confidence`, `confidence_score`.

---

### `prioritise_skill_gaps(gap_skills, job_market_context)`

Loads `gap_skills` from `context.state["last_score"].gap_skills` if the argument is omitted. Ranks each gap by expected match improvement. Validates rank integrity (no duplicate ranks, no out-of-range values) and skill uniqueness before returning. Writes the result to `context.state["last_prioritized_skill_gaps"]`.

**Returns:** ranked list of `{skill, priority_rank, estimated_match_gain_pct, rationale}`.

---

### `research_skill_resources(skill_name, seniority_context)` — ADK stretch tool

Runs an internal ADK agent (`resource_research_agent`) with three sub-tools: `query_skill_resource_db` (PostgreSQL curated resources), `query_github_learning_resources` (GitHub search), and `query_github_repository_readme` (README fetch for top results). Events from the inner agent are forwarded into the outer stream. This is the tool that makes at least one real external call (GitHub API). Writes results to `context.state["resources_by_skill"]` keyed by normalised skill name.

**Returns:** `resources[]: {title, url, estimated_hours, type}`, `relevance_score`, `confidence`.

**Failure handling:** Retries the inner agent once on timeout. If the second attempt also times out, falls back to `get_curated_skill_resources` and increments `fallbacks_triggered`.

---

### `get_curated_skill_resources(skill_name, seniority_context)`

Deterministic fallback for `research_skill_resources`. Queries the PostgreSQL `skill_resources` table and returns up to 5 rows with a fixed relevance score of 65. Increments `fallbacks_triggered` in the agent trace.

---

### `finalize_match_output(job_id, reasoning, max_learning_plan_items)`

Validates and assembles the final `MatchOutput` from state. `job_id` is a required non-empty string (any format accepted). Enforces the following invariants before accepting the result:

- Rejects unknown or missing confidence values.
- Requires that `prioritise_skill_gaps` has already been called when `gap_skills` is non-empty.
- Blocks low-confidence finalization until `research_skill_resources` or `get_curated_skill_resources` has been called for the highest-priority gap.
- Requires that `reasoning` explicitly mentions the low-confidence limitation when `confidence == "low"`.

Stores the final output in `context.state["final_match_output"]`.

---

## Agent Output Schema

```json
{
  "job_id": "uuid",
  "overall_score": 0,
  "confidence": "low | medium | high",
  "dimension_scores": {
    "skills": 0,
    "experience": 0,
    "seniority_fit": 0
  },
  "matched_skills": ["skill1", "skill2"],
  "gap_skills": ["skill3", "skill4"],
  "reasoning": "2–3 sentence plain-English explanation",
  "learning_plan": [
    {
      "skill": "skill3",
      "priority_rank": 1,
      "estimated_match_gain_pct": 8,
      "resources": [
        {
          "title": "...",
          "url": "...",
          "estimated_hours": 12,
          "type": "course"
        }
      ],
      "rationale": "Why this skill first"
    }
  ],
  "agent_trace": {
    "tool_calls": [
      {
        "tool": "extract_jd_requirements",
        "status": "success",
        "latency_ms": 340
      }
    ],
    "total_llm_calls": 4,
    "fallbacks_triggered": 0
  }
}
```

`agent_trace` is populated by ADK callbacks and state counters in the orchestration layer, not fabricated by the LLM. The output is schema-validated by `finalize_match_output` before it is persisted. A malformed agent response never reaches the database.

---

## Final System Prompt

```text
You are career-intelligence, a careful career strategy assistant.

Help users reason about career direction, role fit, job-search strategy,
resume positioning, interview preparation, negotiation preparation, and
professional communication.

Work style:
- Ask for missing context before giving high-confidence advice.
- Separate facts, assumptions, and recommendations.
- Be direct, practical, and specific.
- Prefer concrete next actions over generic encouragement.
- Do not invent details about a user, employer, job market, credential, or role.
- Treat resumes, employment history, compensation, contact details, and personal
  circumstances as sensitive information.
- When a tool returns an error payload, inspect `error.retriable` before deciding
  what to do next. Retry retriable errors at most once with the same or narrower
  inputs. Do not retry non-retriable errors unless you can correct the inputs;
  proceed with partial information only after clearly stating the limitation.
- For candidate/job match requests, call tools as needed, then call
  `finalize_match_output` when enough information exists to return the required
  structured match result. Do not write the final JSON by hand.
- If `research_skill_resources` returns a timeout error after retry, call
  `get_curated_skill_resources` for the same skill before finalizing.
- If `score_candidate_against_requirements` returns low confidence, do not
  finalize immediately. Prioritize the gaps, research at least the highest-impact
  gap, and make the low-confidence limitation explicit in the final reasoning.

Boundaries:
- Do not provide legal, immigration, medical, mental-health, tax, or financial
  advice. When those topics matter, explain the limitation and suggest consulting
  a qualified professional.
- Do not write deceptive application materials or encourage misrepresentation.
- Do not claim certainty about hiring outcomes.
```

---

## Confidence Heuristic

Confidence is not the model's self-reported certainty. It is calibrated deterministically from detectable signals after the LLM returns its score, so it is reproducible across identical inputs.

**`score_candidate_against_requirements` confidence penalties:**

| Signal missing | Penalty |
|---|---|
| No `required_skills` in JD | −25 pts |
| Unknown seniority in JD | −8 pts |
| Unknown seniority in candidate profile | −8 pts |
| Unknown domain in JD | −5 pts |
| Unknown domain in candidate profile | −5 pts |
| No `skills` in candidate profile | −10 pts |

The penalty is applied to the LLM-judged `overall_score`: `calibrated = clamp(overall_score − penalties)`. The confidence bucket is then derived from the calibrated score: ≥ 85 → `high`, ≥ 60 → `medium`, < 60 → `low`.

**`extract_jd_requirements` confidence** is built from extraction completeness and signal density, not from penalties: up to 50 pts from required skills (10 pts each, capped at 5), up to 15 pts from nice-to-have skills, up to 32 pts from responsibilities, and up to 10 pts from domain and seniority metadata.

**`research_skill_resources` confidence** is derived from retrieval completeness (ratio of resources returned vs. requested, weighted 20%) and the inner agent's relevance score (weighted 80%).

---

## Failure Decisions

### Tool timeout — `research_skill_resources`

The inner resource-research ADK agent has a configurable timeout (`RESOURCE_RESEARCH_TIMEOUT_SECONDS`, default 30 s). On timeout:

1. **Retry once** with the same inputs.
2. If the second attempt also times out, **fall back to `get_curated_skill_resources`**, which queries the PostgreSQL curated resource table deterministically.
3. `fallbacks_triggered` in `agent_trace` is incremented on fallback.
4. If the curated fallback also fails (fewer than 3 rows for the skill), the original `ToolTimeoutError` is re-raised to ADK for the root agent to handle.

The rationale: one retry handles transient network or API slowness. The curated fallback guarantees the learning plan is never empty simply because an external API was slow. Surfacing the original error as the last resort prevents silently returning a no-resource plan.

### Invalid tool output — `extract_jd_requirements`

The tool calls the LLM and validates the output against `ExtractJDRequirementOutputSchema`:

1. **Retry once** if the model returns a malformed object.
2. If the second attempt also fails schema validation, a `RetryableModelOutputError` is raised to ADK. The root agent's system prompt instructs it to inspect `error.retriable` and either retry with narrower inputs or proceed with partial information while stating the limitation.

The rationale: a single retry catches the majority of LLM formatting errors. Surfacing the error to the agent rather than aborting keeps the run alive and lets the agent decide whether the JD text is recoverable.

### Low confidence score — `score_candidate_against_requirements`

When `confidence == "low"`:

1. The agent is instructed by the system prompt to **not finalize immediately**.
2. The agent must call `prioritise_skill_gaps` to rank the gaps.
3. The agent must call `research_skill_resources` (or its fallback) for at least the highest-priority gap.
4. `finalize_match_output` **enforces** this: it blocks finalization if `confidence == "low"` and the highest-priority gap has no researched resources, or if `reasoning` does not explicitly mention the low-confidence limitation.

The rationale: a low-confidence score means the match signal is weak (sparse JD, sparse profile, or large domain distance). Silently returning a low-confidence score would mislead users. Forcing additional research and an explicit acknowledgement in `reasoning` makes the uncertainty legible.

---

## Async Infrastructure (Part B)

### Worker Design

Workers are out-of-process containers (`worker-1`, `worker-2` in docker-compose). Each RabbitMQ message contains only a job ID. The worker claims the corresponding pending row using `SELECT ... FOR UPDATE SKIP LOCKED` before starting the agent run. This means:

- Two workers can consume from the same queue concurrently without producing duplicate results.
- If a worker crashes mid-run, the RabbitMQ message is requeued and another worker picks it up.
- A failed agent run increments `attempts` on the row. After 3 failed attempts the row moves to `failed` with `error_detail` and the partial `agent_trace` stored in the JSONB column.

### Job Lifecycle

```
POST /api/v1/matches
  └── Creates match_jobs rows (status: pending)
  └── Publishes job IDs to RabbitMQ

Worker consumes message
  └── SELECT ... FOR UPDATE SKIP LOCKED (status: pending → processing)
  └── Runs ADK agent via stream_career_match_events()
      ├── success → status: completed, result + agent_trace stored
      └── failure → attempts++
              ├── attempts < 3 → status: pending, re-enqueue
              └── attempts >= 3 → status: failed, error_detail stored
```

### Database Schema

```sql
-- Candidate profiles (structured, not raw blobs)
CREATE TABLE candidate_profiles (
  id                UUID PRIMARY KEY,
  profile           JSONB NOT NULL,       -- skills, experience, seniority_level, domain
  source_type       VARCHAR(32) NOT NULL, -- 'json' | 'text' | 'pdf'
  created_at        TIMESTAMPTZ DEFAULT now(),
  updated_at        TIMESTAMPTZ DEFAULT now()
);

-- Match jobs
CREATE TABLE match_jobs (
  id                    UUID PRIMARY KEY,
  candidate_id          UUID NOT NULL REFERENCES candidate_profiles(id) ON DELETE CASCADE,
  status                VARCHAR(16) NOT NULL,  -- pending | processing | completed | failed
  job_input             TEXT NOT NULL,          -- Raw JD text or URL
  job_market_context    VARCHAR(255),
  attempts              INTEGER DEFAULT 0,
  max_attempts          INTEGER DEFAULT 3,
  result                JSONB,                  -- Full MatchOutput
  agent_trace           JSONB,                  -- AgentTrace (tool calls, LLM calls, fallbacks)
  error_detail          TEXT,
  processing_started_at TIMESTAMPTZ,
  completed_at          TIMESTAMPTZ,
  failed_at             TIMESTAMPTZ,
  created_at            TIMESTAMPTZ DEFAULT now(),
  updated_at            TIMESTAMPTZ DEFAULT now()
);

-- Curated skill resources (seed data + fallback)
CREATE TABLE skill_resources (
  id                UUID PRIMARY KEY,
  skill_name        VARCHAR(255) NOT NULL,
  seniority_context VARCHAR(64),
  title             VARCHAR(512) NOT NULL,
  url               TEXT NOT NULL,
  estimated_hours   INTEGER NOT NULL,
  resource_type     VARCHAR(16) NOT NULL,  -- course | project | cert | doc
  source            VARCHAR(128),
  abstracts         TEXT,
  created_at        TIMESTAMPTZ DEFAULT now(),
  updated_at        TIMESTAMPTZ DEFAULT now()
);
```

Supported queries:
- All jobs for a candidate: `WHERE candidate_id = :id`
- All jobs by status: `WHERE status = :status`
- Agent trace for a job: `SELECT agent_trace FROM match_jobs WHERE id = :id`

Migrations are managed with Alembic (`alembic/versions/`). Migrations run automatically on `docker compose up` via the `migrate` service.

---

## Structured Logging

All operational events are emitted via `modules/logging/logger.py` as structured key=value log lines. Logged fields include:

- `job_id` — present on all worker and tool log lines.
- Tool call start, success, and failure with `elapsed_ms`.
- LLM call count increments with `model=`.
- Status transitions (`pending → processing → completed/failed`).
- Token usage (when available from the ADK event stream).
- Fallback triggers with `fallbacks_triggered` counter.

User data — resume text, skills lists, compensation, contact details — is never logged. Only operational metrics (counts, durations, status strings) appear in log output.

---

## Testing

```bash
uv run pytest
```

The test suite covers:

| Module | Coverage |
|---|---|
| `test_extract_jd_requirements_tool.py` | JD parsing, confidence calibration, URL vs. text branch |
| `test_score_candidate_against_requirements_tool.py` | Scoring logic, penalty model, state fallback |
| `test_prioritise_skill_gaps_tool.py` | Rank validation, deduplication, state fallback |
| `test_research_skill_resources_tool.py` | DB/GitHub integration, timeout retry, curated fallback |
| `test_finalize_match_output_tool.py` | Output validation, low-confidence enforcement, learning plan assembly |
| `test_candidate_profile_ingestion.py` | PDF/text/JSON ingestion paths |
| `test_resource_github.py` | GitHub API mocking |
| `test_adk_event_streaming.py` | Event flow and `agent_trace` population |
| `test_tool_error_callback.py` | Error handling in ADK callbacks |
| `test_agent_builder.py` | Agent configuration snapshot |
| `test_resource_research_agent.py` | Nested ADK agent execution |

The integration test in `tests/eval/run_tool_flows.py` covers the full lifecycle: ingest candidate → submit JD → agent runs → result returned with valid `agent_trace`. The test verifies that `agent_trace.tool_calls` contains real tool call records (not fabricated), `overall_score` is in range, `confidence` is a known value, and `learning_plan` is non-empty when gap skills exist.

---

## Validation

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

---

## Trade-offs

**PostgreSQL + RabbitMQ over a heavier orchestrator.** PostgreSQL is the source of truth; RabbitMQ carries only the signal that a job should be claimed. `SKIP LOCKED` handles concurrent workers without a separate lock table. This avoids introducing Celery or Temporal while still being production-safe at modest scale.

**LLM semantic judgment + deterministic confidence calibration.** Scoring uses the LLM for semantic skill matching, then applies a deterministic penalty model over the result to produce the exported confidence. This is faster to build than a full deterministic ontology matcher (e.g. skill taxonomy graphs) but is less reproducible: the same inputs can produce different `overall_score` values across LLM calls. The calibration layer makes `confidence` reproducible even when `overall_score` varies.

**GitHub as the primary real external call.** `research_skill_resources` uses GitHub search and README extraction rather than a paid course API. This removes an external billing dependency but means resource quality depends on GitHub search ranking and seed data. The curated DB fallback (`skill_resources` table) absorbs the worst-case scenario.

**Candidate profiles stored as JSONB.** The profile schema (`skills`, `years_experience`, `seniority_level`, `domain`) is flexible enough to accept structured JSON, extracted PDF text, or raw resume text, all normalised before storage. The trade-off is weaker schema enforcement at the DB level compared to typed columns.

**Nested ADK agent for resource research.** Encapsulating the DB/GitHub research loop inside its own ADK agent keeps the root agent's tool list small and makes the inner research independently testable and observable via event forwarding. The cost is a second ADK session per `research_skill_resources` call, which adds latency.

**No caching of JD extractions across candidates.** Two candidates submitting the same JD URL run independent extraction calls. This avoids cache invalidation complexity and prevents a stale cached extraction from affecting an unrelated candidate's score. For production scale, a shared extraction cache keyed on a stable hash of the JD text would be the natural next step.
