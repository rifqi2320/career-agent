## Career Agent

Google ADK career-intelligence agent for matching candidate profiles against job
requirements and recommending focused skill-development resources.

## Framework Choice

This project uses Google ADK because the assignment needs runtime tool
selection, session state, tool callbacks, nested agent execution, and event
streaming without building a custom ReAct loop.

## Local Setup

```bash
cp .env.example .env
# Fill POSTGRES_PASSWORD, RABBITMQ_DEFAULT_PASS, and GOOGLE_API_KEY.
# POSTGRES_USER, POSTGRES_DB, RABBITMQ_DEFAULT_USER, and MATCH_QUEUE_NAME have
# non-secret local defaults.
uv sync
docker compose up --build
```

The FastAPI app is served at `http://127.0.0.1:8080`. Docker Compose starts
PostgreSQL, RabbitMQ, a one-shot Alembic migration container, the API container,
and two worker containers. Workers consume durable RabbitMQ job IDs and claim
pending rows in PostgreSQL before running the ADK agent.

ADK web remains available behind a Compose profile on `http://127.0.0.1:8001`:

```bash
docker compose --profile adk up --build
```

The optional LiteLLM proxy is behind a Compose profile:

```bash
docker compose --profile litellm up --build
```

For local CLI runs outside Docker:

```bash
uv run adk run agents/career_intelligence
uv run uvicorn main:app --reload
uv run python -m modules.worker.run
```

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string for curated skill resources.
  Local Docker Compose builds it from `POSTGRES_USER`, `POSTGRES_PASSWORD`, and
  `POSTGRES_DB`; set `DATABASE_URL` directly for non-Compose runs.
- `AMQP_URL`: RabbitMQ connection string for non-Compose API/worker runs. Local
  Docker Compose builds it from `RABBITMQ_DEFAULT_USER` and
  `RABBITMQ_DEFAULT_PASS`.
- `MATCH_QUEUE_NAME`: durable RabbitMQ queue name. Defaults to
  `career.match_jobs`.
- `GOOGLE_API_KEY`: Google Gemini API key used by ADK when not using Vertex AI.
- `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`,
  `GOOGLE_CLOUD_LOCATION`: optional Vertex AI settings for Google ADK.
- `GITHUB_TOKEN` or `GITHUB_PAT_TOKEN`: optional token used by
  `research_skill_resources` for GitHub API rate limits.
- `GITHUB_API_BASE`: optional GitHub API base URL. Defaults to
  `https://api.github.com`.
- `GITHUB_TIMEOUT_SECONDS`: optional timeout for GitHub API calls. Defaults to
  `10`.
- `RESOURCE_RESEARCH_TIMEOUT_SECONDS`: optional timeout for the internal
  resource research agent. Defaults to `30`.

## Validation

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

## Part A Workflow

The root career-intelligence agent is executed through `google.adk.runners.Runner`
in `modules.agent_runtime`. `stream_career_match_events(...)` creates one ADK
session per match job, streams sanitized ADK events for model responses, tool
calls, tool responses, state deltas, completion, and failures, and returns the
validated output produced by `finalize_match_output`.

Expected tool flow:

1. `extract_jd_requirements`
2. `score_candidate_against_requirements`
3. `prioritise_skill_gaps`
4. `research_skill_resources` for the highest-impact gaps
5. `finalize_match_output`

`agent_trace` is built from ADK callbacks and state counters, not from model
text. Tool argument values and response bodies are not emitted in stream
metadata; events expose keys and status only. The ADK session state is
initialized from the typed `AgentRunState` schema and validated again before the
final output is accepted.

## Part B Async API

- `POST /api/v1/candidate`: accepts JSON `{ "profile": ... }`,
  `{ "resume_text": "..." }`, or multipart `resume_file` PDF/text fields. It
  stores only the structured candidate profile in PostgreSQL.
- `POST /api/v1/matches`: accepts a stored `candidate_id` and up to 10
  `job_descriptions`, creates one pending row per JD, publishes durable RabbitMQ
  job messages, and returns immediately with job IDs.
- `GET /api/v1/matches/{id}`: returns `pending`, `processing`, `completed`, or
  `failed` plus the validated output when complete.
- `GET /api/v1/matches?limit=20&offset=0&status=pending`: lists jobs with
  pagination and optional status filtering.
- `POST /api/v1/matches/{id}/requeue`: admin retry for failed jobs.

Workers are out-of-process containers. Each RabbitMQ message contains only a job
ID; the worker uses `SELECT ... FOR UPDATE SKIP LOCKED` on the pending
`match_jobs` row before processing, so two workers can consume concurrently
without producing duplicate results. A failed run is retried up to three total
attempts; after that the row moves to `failed` with error detail and the partial
agent trace available so far.

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

## Tools

- `extract_jd_requirements(job_url_or_text)`: accepts raw JD text, URL strings,
  or LLM-supplied objects containing a URL/text field. URL content is fetched and
  converted into a semantic page snapshot before schema-validated extraction.
- `score_candidate_against_requirements(candidate_profile, requirements)`: scores
  candidate fit and writes `last_score` into ADK session state.
- `prioritise_skill_gaps(gap_skills, job_market_context)`: ranks gaps by likely
  match improvement and validates rank integrity.
- `research_skill_resources(skill_name, seniority_context)`: runs an internal ADK
  research agent over curated DB resources and GitHub search/README tools.
- `get_curated_skill_resources(skill_name, seniority_context)`: deterministic
  fallback that returns DB-curated resources when resource research times out.
- `finalize_match_output(...)`: validates and returns the final `MatchOutput`
  with orchestrator-populated `agent_trace`. It requires a UUID `job_id`, rejects
  unknown confidence values, requires gap prioritization before a learning plan
  is finalized, and blocks low-confidence finalization until the highest-priority
  gap has researched resources.

## Confidence Heuristic

The score tool asks the LLM to judge semantic fit, but the exported confidence is
calibrated deterministically from detectable signals:

- missing required skills lowers confidence;
- unknown job seniority or candidate seniority lowers confidence;
- unknown job domain or candidate domain lowers confidence;
- missing candidate skills lowers confidence;
- the final confidence bucket is derived from the calibrated 0-100 score.

JD extraction confidence is based on extracted required skills, nice-to-have
skills, responsibilities, domain, and seniority signal density. Resource
research confidence is based on relevance score plus retrieval completeness.

## Failure Decisions

- Tool timeout: `research_skill_resources` retries the internal resource
  research agent once. If the second attempt times out, it falls back to
  `get_curated_skill_resources` and increments `fallbacks_triggered`.
- Invalid JD extraction output: `extract_jd_requirements` retries one malformed
  model output. If the retry also fails schema validation, the retryable error is
  surfaced to ADK for the agent/runtime to handle.
- Low confidence score: `finalize_match_output` enforces prioritized gaps and
  researched resources for the highest-priority gap before accepting a
  low-confidence final payload, then ensures the reasoning explicitly says the
  result is low confidence.

## ADK Runtime Streaming

`stream_career_match_events(...)` emits sanitized events for the root ADK run.
The `research_skill_resources` tool also forwards events from its internal ADK
resource-research agent, so DB/GitHub retrieval steps can appear in the same
stream as the root match workflow.

## Trade-Offs

- The API/worker slice is intentionally thin: PostgreSQL is the source of truth,
  and RabbitMQ is only the durable signal that a job should be claimed. This
  avoids a larger orchestration framework while still supporting multiple worker
  processes safely.
- Candidate PDF/text ingestion extracts the minimal structured profile needed by
  the scoring tool. Passing structured JSON remains the highest-quality path.
- Scoring still uses LLM semantic judgment for matched/gap skills, then applies a
  deterministic confidence calibration. This is faster to build than a full
  deterministic ontology matcher but less reproducible.
- Resource research uses GitHub as the real external signal and curated DB rows
  as a fallback. This avoids paid course APIs but means resource quality depends
  on GitHub search and seed data.
