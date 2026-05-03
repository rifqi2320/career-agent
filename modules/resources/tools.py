"""Internal tools for the resource research agent."""

from __future__ import annotations

from modules.error.common import ToolExecutionError, ToolInputError
from modules.resources.github import (
    fetch_github_repository_readme,
    search_github_learning_resources,
)
from modules.resources.repository import list_skill_resources
from modules.resources.service import rows_to_candidate_resources

DB_RESOURCE_LIMIT = 15
GITHUB_RESOURCE_LIMIT = 15


def query_skill_resource_db(
    skill_name: str,
    seniority_context: str = "unknown",
    limit: int = DB_RESOURCE_LIMIT,
) -> list[dict[str, object]]:
    """Search curated skill resources in the project database."""
    result = list_skill_resources(
        skill_name=skill_name,
        seniority_context=seniority_context,
        limit=limit,
    )
    if result.is_err():
        raise ToolExecutionError(
            f"Failed to fetch skill resources from database: {result.error}",
            original_error=result.error,
        )
    rows = result.value or []
    candidates = rows_to_candidate_resources(rows)
    return [candidate.model_dump(mode="json") for candidate in candidates]


def query_github_learning_resources(
    skill_name: str,
    seniority_context: str = "unknown",
    limit: int = GITHUB_RESOURCE_LIMIT,
) -> list[dict[str, object]]:
    """Search GitHub repositories for relevant learning projects and examples."""
    result = search_github_learning_resources(
        skill_name=skill_name,
        seniority_context=seniority_context,
        limit=limit,
    )
    if result.is_err():
        error = result.error
        if isinstance(error, ToolInputError):
            raise error
        raise ToolExecutionError(
            f"Failed to fetch learning resources from GitHub: {error}",
            original_error=error,
            retriable=getattr(error, "retriable", False),
        )
    candidates = result.value or []
    return [candidate.model_dump(mode="json") for candidate in candidates]


def query_github_repository_readme(repository: str) -> dict[str, object]:
    """Fetch README text for one GitHub repository returned by GitHub search."""
    result = fetch_github_repository_readme(repository=repository)
    if result.is_err():
        error = result.error
        if isinstance(error, ToolInputError):
            raise error
        raise ToolExecutionError(
            f"Failed to fetch GitHub README: {error}",
            original_error=error,
            retriable=getattr(error, "retriable", False),
        )
    readme = result.value
    if readme is None:
        raise ToolExecutionError("GitHub README fetch returned no result.")
    return readme.model_dump(mode="json")
