"""Typed accessors for ADK match-run state."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

from modules.error.common import ToolInputError
from modules.utils.trace import get_state

LAST_REQUIREMENTS_KEY = "last_requirements"
LAST_SCORE_KEY = "last_score"
LAST_PRIORITIZED_SKILL_GAPS_KEY = "last_prioritized_skill_gaps"
LAST_RESOURCES_RESEARCH_KEY = "last_resources_research"
RESOURCES_BY_SKILL_KEY = "resources_by_skill"
FINAL_MATCH_OUTPUT_KEY = "final_match_output"
JOB_ID_KEY = "job_id"


def require_match_state(context: object) -> MutableMapping[str, object]:
    """Return mutable ADK state or raise a tool-facing input error."""
    state = get_state(context)
    if state is None:
        raise ToolInputError("Tool context state is required.")
    return state


def state_snapshot(state: MutableMapping[str, object]) -> dict[str, object]:
    """Return a plain dict for normal mappings and ADK State objects."""
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        snapshot = to_dict()
        if isinstance(snapshot, dict):
            return cast("dict[str, object]", snapshot)
    return dict(state)


def require_mapping(
    state: MutableMapping[str, object],
    field_name: str,
) -> dict[str, object]:
    """Read a required mapping payload from match state."""
    value = state.get(field_name)
    if not isinstance(value, dict):
        raise ToolInputError(f"`context.state['{field_name}']` is required.")
    return cast("dict[str, object]", value)


def string_list(value: object) -> list[str]:
    """Return normalized non-empty strings from a loose list payload."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def int_field(payload: dict[str, object], field_name: str) -> int:
    """Read a required integer field from a state payload."""
    value = payload.get(field_name)
    if not isinstance(value, int):
        raise ToolInputError(f"`{field_name}` must be an integer.")
    return value
