# Agent Engineering Guide

This project builds a Google ADK agent for career-related workflows. Keep the implementation small, explicit, testable, and easy to operate.

## Goals

- Build one clear agent experience before adding orchestration complexity.
- Keep business logic outside prompt strings.
- Treat every tool call as an external boundary with validation, errors, and tests.
- Prefer boring, typed Python over clever abstractions.

## Recommended Structure

Use a package layout instead of growing logic in `main.py`. `agents/` is the
only runtime entrypoint and is served via ADK web:

```text
agents/
  career_intelligence/
    __init__.py
    agent.py        # ADK agent construction only
    config.py       # environment and runtime configuration
    prompts.py      # system instructions and prompt fragments
modules/
  __init__.py
  <domain_a>/
    __init__.py
    ...             # business logic + data access for this domain
  <domain_b>/
    __init__.py
    ...             # business logic + data access for this domain
tests/
  test_agent.py
  test_tools.py
```

`agent.py` should compose the agent. It should not contain domain algorithms, API parsing, persistence logic, or large prompt literals.
All business logic, builders, and data access must live under `modules/`.
Module folders are separated by domain scope, not by technical layers, and a
single domain folder may include multiple layers when it improves cohesion.

## ADK Agent Rules

- Define the root agent in one place and export it with a stable name, for example `root_agent`.
- Keep the agent instruction specific, concise, and behavior-oriented.
- Give every tool a narrow responsibility and a clear docstring because tool descriptions affect model behavior.
- Prefer deterministic tool code for scoring, filtering, formatting, and validation. Do not ask the model to do work that simple Python can do reliably.
- Return structured data from tools where possible, not prose blobs.
- Make handoffs or sub-agents only when there is a real separation of responsibility. Start with a single agent unless the workflow clearly needs routing.

## Prompt Standards

- Put reusable prompt text in `prompts.py`.
- Separate stable policy from dynamic context.
- Avoid embedding secrets, environment names, file paths, or one-off examples in the core instruction.
- Specify refusal and escalation behavior for sensitive career advice, legal concerns, immigration concerns, compensation claims, and medical or mental-health topics.
- Make uncertainty explicit. The agent should say when it lacks enough information instead of inventing details about a user, company, job, or market.

## Tool Standards

Every tool should:

- Use typed inputs and outputs.
- Validate required fields at the boundary.
- Fail with actionable error messages.
- Avoid hidden network calls unless the tool name and docstring make that behavior obvious.
- Be idempotent when practical.
- Log operational details without logging private user content.

Tools must not:

- Read arbitrary local files without an explicit allowlist.
- Expose secrets or raw environment variables.
- Make irreversible changes without a confirmation step.
- Mix retrieval, ranking, and final response generation in one function.

## Configuration

- Load configuration from environment variables through `config.py`.
- Keep defaults safe for local development.
- Validate configuration during startup.
- Never hard-code API keys, credentials, user identifiers, or deployment-specific URLs.
- Document required environment variables in `README.md` when they are introduced.

## Data And Privacy

Career workflows may process resumes, employment history, compensation data, and personal contact details. Treat all of it as sensitive.

- Collect the minimum data needed for the task.
- Do not persist user data unless the product requirement explicitly needs it.
- Redact sensitive values in logs and test snapshots.
- Do not send user data to third-party services through tools without making that boundary explicit in code and documentation.
- Prefer derived summaries over storing raw resumes or transcripts.

## Testing Expectations

Add tests when adding behavior. At minimum:

- Unit test each non-trivial tool.
- Test validation and error cases, not only successful paths.
- Snapshot or assert the root agent configuration enough to catch accidental tool removal or instruction regressions.
- Mock network and filesystem boundaries.
- Keep tests deterministic; do not require live LLM calls in normal CI.

Useful local commands:

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run adk run agents/career_intelligence
uv run pytest
```

Add `pytest` to project dependencies before relying on the test command.

## Code Quality

- Use Python 3.12 features where they simplify the code.
- Prefer `dataclass`, `TypedDict`, or Pydantic models for structured data.
- Keep functions short and named after domain actions.
- Avoid global mutable state.
- Keep imports acyclic: tools may import schemas and config, but schemas and config must not import tools or agents.
- Format with `ruff`, lint with `ruff check`, and type-check with `ty check`.
- Keep `ruff` on `select = ["ALL"]` and `ty` on `all = "error"` unless there is a documented reason to narrow a specific rule.

## Coding Style Preference

- Prefer straightforward control flow over clever constructs.
- Use plain `if` statements for simple branching; avoid pattern matching unless it clearly improves readability.
- Keep builder modules small and practical: one entry function and focused provider-specific helpers.
- Keep schema models as data containers; place business logic in module functions.
- Prefer explicit fields over generic `kwargs`; if extensibility is needed, use a named `extra_kwargs` field.
- Optimize for readability first: short functions, obvious variable names, and minimal indirection.
- For any function that can fail, prefer `returns.result.safe` wrappers for explicit success/failure flow.
- Use project errors from `modules/error` (for example `UnknownOptionsError`, `IncorrectCombinationError`) instead of generic exceptions when meaningful.
- Keep enum types strict and include `UNKNOWN` fallback behavior via `_missing_` where external input may be untrusted.
- For LiteLLM config, keep connection fields explicit (`client_type`, `api_base`, `api_key`, `api_version`, `timeout`, `max_retries`, `stream`) and keep provider-specific extras in `extra_kwargs`.
- Keep provider-to-model compatibility explicit via `MODEL_PROVIDER_MAP`.

## Review Checklist

Before considering the agent ready:

- The root agent starts locally.
- The instruction is readable and does not contain secrets.
- Each tool has typed boundaries and focused tests.
- Errors are handled without stack traces leaking to users.
- Logs avoid private career or resume content.
- README documents setup, environment variables, and local run commands.
- There is a small example conversation or smoke test for the main career workflow.
