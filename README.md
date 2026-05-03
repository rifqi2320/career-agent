## Career Agent

Google ADK career-intelligence agent for matching candidate profiles against job
requirements and recommending focused skill-development resources.

## Local Setup

```bash
uv sync
docker compose up -d
uv run adk run agents/career_intelligence
```

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string for curated skill resources.
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
