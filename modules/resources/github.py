"""GitHub API helpers for learning-resource discovery."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from safe_result import safe

from modules.config.resources import (
    ResourceResearchConfig,
    load_resource_research_config,
)
from modules.error.common import ToolExecutionError, ToolInputError
from modules.resources.schemas import (
    CandidateResourceSchema,
    GitHubReadmeSchema,
    ResourceType,
)

DEFAULT_GITHUB_LIMIT = 10
MAX_GITHUB_LIMIT = 20
README_TEXT_LIMIT = 4_000


@safe
def search_github_learning_resources(
    *,
    skill_name: str,
    seniority_context: str,
    limit: int = DEFAULT_GITHUB_LIMIT,
    config: ResourceResearchConfig | None = None,
) -> list[CandidateResourceSchema]:
    """Search GitHub repositories for learning resources related to a skill."""
    normalized_skill = skill_name.strip()
    if not normalized_skill:
        raise ToolInputError("skill_name must not be empty.")
    if limit <= 0:
        raise ToolInputError("limit must be greater than 0.")

    research_config = config or load_resource_research_config()
    per_page = min(limit, MAX_GITHUB_LIMIT)
    payload = _request_github_repository_search(
        skill_name=normalized_skill,
        seniority_context=seniority_context,
        per_page=per_page,
        config=research_config,
    )
    return _map_github_repositories(
        payload=payload,
        skill_name=normalized_skill,
        seniority_context=seniority_context,
        limit=per_page,
    )


@safe
def fetch_github_repository_readme(
    *,
    repository: str,
    config: ResourceResearchConfig | None = None,
) -> GitHubReadmeSchema:
    """Fetch README text for a GitHub repository in `owner/name` format."""
    normalized_repository = repository.strip()
    if not normalized_repository:
        raise ToolInputError("repository must not be empty.")
    if "/" not in normalized_repository:
        raise ToolInputError("repository must use owner/name format.")

    owner, repo = normalized_repository.split("/", maxsplit=1)
    if not owner.strip() or not repo.strip():
        raise ToolInputError("repository must use owner/name format.")

    research_config = config or load_resource_research_config()
    payload = _request_github_readme(
        owner=owner.strip(),
        repo=repo.strip(),
        config=research_config,
    )
    return _map_github_readme(
        payload=payload,
        repository=normalized_repository,
    )


def _request_github_repository_search(
    *,
    skill_name: str,
    seniority_context: str,
    per_page: int,
    config: ResourceResearchConfig,
) -> Mapping[str, Any]:
    """Call GitHub's repository search API."""
    query = _build_github_query(
        skill_name=skill_name,
        seniority_context=seniority_context,
    )
    query_string = urlencode(
        {
            "q": query,
            "per_page": str(per_page),
        }
    )
    base_url = config.github_api_base.rstrip("/")
    request = Request(
        url=f"{base_url}/search/repositories?{query_string}",
        headers=_build_github_headers(config),
        method="GET",
    )

    try:
        with urlopen(request, timeout=config.github_timeout_seconds) as response:
            raw_payload = response.read().decode("utf-8")
    except HTTPError as error:
        raise ToolExecutionError(
            f"GitHub repository search failed with HTTP {error.code}.",
            original_error=error,
            retriable=error.code in {403, 429, 500, 502, 503, 504},
        ) from error
    except URLError as error:
        raise ToolExecutionError(
            "GitHub repository search failed due to a network error.",
            original_error=error,
            retriable=True,
        ) from error

    parsed_payload = json.loads(raw_payload)
    if not isinstance(parsed_payload, Mapping):
        raise ToolExecutionError("GitHub repository search returned invalid JSON.")
    return parsed_payload


def _request_github_readme(
    *,
    owner: str,
    repo: str,
    config: ResourceResearchConfig,
) -> Mapping[str, Any]:
    """Call GitHub's repository README API."""
    base_url = config.github_api_base.rstrip("/")
    request = Request(
        url=f"{base_url}/repos/{owner}/{repo}/readme",
        headers=_build_github_headers(config),
        method="GET",
    )

    try:
        with urlopen(request, timeout=config.github_timeout_seconds) as response:
            raw_payload = response.read().decode("utf-8")
    except HTTPError as error:
        raise ToolExecutionError(
            f"GitHub README fetch failed with HTTP {error.code}.",
            original_error=error,
            retriable=error.code in {403, 429, 500, 502, 503, 504},
        ) from error
    except URLError as error:
        raise ToolExecutionError(
            "GitHub README fetch failed due to a network error.",
            original_error=error,
            retriable=True,
        ) from error

    parsed_payload = json.loads(raw_payload)
    if not isinstance(parsed_payload, Mapping):
        raise ToolExecutionError("GitHub README fetch returned invalid JSON.")
    return parsed_payload


def _build_github_headers(config: ResourceResearchConfig) -> dict[str, str]:
    """Build GitHub API headers without exposing secrets."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "career-agent-resource-research",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if config.github_token is not None:
        headers["Authorization"] = f"Bearer {config.github_token}"
    return headers


def _build_github_query(*, skill_name: str, seniority_context: str) -> str:
    """Build a repository-search query for a target skill."""
    terms = [
        skill_name,
        "tutorial OR project OR roadmap OR examples",
        "in:name,description,readme",
    ]
    normalized_seniority = seniority_context.strip().casefold()
    if normalized_seniority and normalized_seniority != "unknown":
        terms.append(normalized_seniority)
    return " ".join(terms)


def _map_github_repositories(
    *,
    payload: Mapping[str, Any],
    skill_name: str,
    seniority_context: str,
    limit: int,
) -> list[CandidateResourceSchema]:
    """Map GitHub repository payloads into candidate resource schemas."""
    raw_items = payload.get("items", [])
    if not isinstance(raw_items, list):
        raise ToolExecutionError("GitHub repository search returned invalid items.")

    candidates: list[CandidateResourceSchema] = []
    for raw_item in raw_items[:limit]:
        if not isinstance(raw_item, Mapping):
            continue
        candidate = _map_github_repository(
            raw_item,
            skill_name=skill_name,
            seniority_context=seniority_context,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _map_github_repository(
    raw_item: Mapping[str, Any],
    *,
    skill_name: str,
    seniority_context: str,
) -> CandidateResourceSchema | None:
    """Map one GitHub repository object into a candidate resource."""
    raw_title = raw_item.get("full_name") or raw_item.get("name")
    raw_url = raw_item.get("html_url")
    if not isinstance(raw_title, str) or not isinstance(raw_url, str):
        return None

    description = raw_item.get("description")
    stars = raw_item.get("stargazers_count")
    language = raw_item.get("language")
    updated_at = raw_item.get("updated_at")
    metadata_parts: list[str] = []
    if isinstance(description, str) and description.strip():
        metadata_parts.append(description.strip())
    if isinstance(stars, int):
        metadata_parts.append(f"GitHub stars: {stars}")
    if isinstance(language, str) and language.strip():
        metadata_parts.append(f"Primary language: {language.strip()}")
    if isinstance(updated_at, str) and updated_at.strip():
        metadata_parts.append(f"Last updated: {updated_at.strip()}")

    return CandidateResourceSchema(
        title=raw_title,
        abstracts="; ".join(metadata_parts) or None,
        url=raw_url,
        estimated_hours=0,
        type=ResourceType.PROJECT,
        skill_name=skill_name,
        seniority_context=seniority_context or "unknown",
        source="github_api",
    )


def _map_github_readme(
    *,
    payload: Mapping[str, Any],
    repository: str,
) -> GitHubReadmeSchema:
    """Map GitHub README payload into bounded text for the internal agent."""
    raw_url = payload.get("html_url")
    raw_content = payload.get("content")
    raw_encoding = payload.get("encoding")
    if not isinstance(raw_url, str) or not isinstance(raw_content, str):
        raise ToolExecutionError("GitHub README payload is missing required fields.")
    if raw_encoding != "base64":
        raise ToolExecutionError("GitHub README payload uses unsupported encoding.")

    try:
        readme_bytes = base64.b64decode(raw_content, validate=False)
        readme_text = readme_bytes.decode("utf-8", errors="replace")
    except ValueError as error:
        raise ToolExecutionError(
            "GitHub README payload could not be decoded.",
            original_error=error,
        ) from error

    return GitHubReadmeSchema(
        repository=repository,
        url=raw_url,
        readme_text=readme_text[:README_TEXT_LIMIT],
    )
