"""Runtime trace helpers for ADK agent callbacks and tools."""

from __future__ import annotations

from collections.abc import MutableMapping
from time import perf_counter
from typing import Any, cast

from google.adk.agents.context import Context
from google.adk.tools.base_tool import BaseTool

from models.match import AgentTrace

TOOL_CALLS_STATE_KEY = "tool_calls"
TOTAL_LLM_CALLS_STATE_KEY = "total_llm_calls"
FALLBACKS_TRIGGERED_STATE_KEY = "fallbacks_triggered"
_TOOL_STARTS_STATE_KEY = "_tool_starts"


def get_state(context: Context | object) -> MutableMapping[str, object] | None:
    """Return mutable ADK state when the context exposes it."""
    state = getattr(context, "state", None)
    if isinstance(state, MutableMapping):
        return state
    if _is_state_like(state):
        return cast("MutableMapping[str, object]", state)
    return None


def _is_state_like(value: object) -> bool:
    """Return true for ADK State, which is mutable but not a MutableMapping."""
    return callable(getattr(value, "get", None)) and callable(
        getattr(value, "__setitem__", None)
    )


def tool_name(tool: BaseTool | object) -> str:
    """Return the runtime tool name used in traces."""
    return str(
        getattr(tool, "name", None)
        or getattr(tool, "__name__", None)
        or type(tool).__name__
    )


def begin_tool_trace(tool: BaseTool | object, context: Context | object) -> None:
    """Record a tool start timestamp in mutable state."""
    state = get_state(context)
    if state is None:
        return

    raw_starts = state.get(_TOOL_STARTS_STATE_KEY)
    if isinstance(raw_starts, dict):
        starts = cast("dict[str, float]", raw_starts)
    else:
        starts: dict[str, float] = {}
        state[_TOOL_STARTS_STATE_KEY] = starts
    starts[tool_name(tool)] = perf_counter()


def append_tool_trace(
    tool: BaseTool | object,
    context: Context | object,
    *,
    status: str,
    error_type: str | None = None,
    message: str | None = None,
) -> None:
    """Append one tool call trace entry to mutable state."""
    state = get_state(context)
    if state is None:
        return

    name = tool_name(tool)
    started_at = None
    raw_starts = state.get(_TOOL_STARTS_STATE_KEY)
    if isinstance(raw_starts, dict):
        starts = cast("dict[str, object]", raw_starts)
        raw_started_at = starts.pop(name, None)
        if isinstance(raw_started_at, int | float):
            started_at = float(raw_started_at)
    latency_ms = int((perf_counter() - started_at) * 1000) if started_at else 0

    entry: dict[str, object] = {
        "tool": name,
        "status": status,
        "latency_ms": max(latency_ms, 0),
    }
    if error_type is not None:
        entry["error_type"] = error_type
    if message is not None:
        entry["message"] = message

    raw_tool_calls = state.get(TOOL_CALLS_STATE_KEY)
    if isinstance(raw_tool_calls, list):
        tool_calls = cast("list[dict[str, object]]", raw_tool_calls)
        tool_calls.append(entry)
    else:
        state[TOOL_CALLS_STATE_KEY] = [entry]


def increment_llm_calls(context: Context | object, *, count: int = 1) -> None:
    """Increment the orchestrator-visible LLM call counter."""
    state = get_state(context)
    if state is None:
        return

    raw_count = state.get(TOTAL_LLM_CALLS_STATE_KEY)
    current_count = raw_count if isinstance(raw_count, int) else 0
    state[TOTAL_LLM_CALLS_STATE_KEY] = max(0, current_count + count)


def increment_fallbacks(context: Context | object, *, count: int = 1) -> None:
    """Increment the fallback counter."""
    state = get_state(context)
    if state is None:
        return

    raw_count = state.get(FALLBACKS_TRIGGERED_STATE_KEY)
    current_count = raw_count if isinstance(raw_count, int) else 0
    state[FALLBACKS_TRIGGERED_STATE_KEY] = max(0, current_count + count)


def build_agent_trace(context: Context | object) -> AgentTrace:
    """Build a validated trace from mutable runtime state."""
    state = get_state(context)
    if state is None:
        return AgentTrace()

    raw_tool_calls = state.get(TOOL_CALLS_STATE_KEY)
    tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
    raw_llm_calls = state.get(TOTAL_LLM_CALLS_STATE_KEY)
    raw_fallbacks = state.get(FALLBACKS_TRIGGERED_STATE_KEY)
    return AgentTrace.model_validate(
        {
            "tool_calls": tool_calls,
            "total_llm_calls": raw_llm_calls if isinstance(raw_llm_calls, int) else 0,
            "fallbacks_triggered": raw_fallbacks
            if isinstance(raw_fallbacks, int)
            else 0,
        }
    )


def store_tool_result_by_key(
    context: Context | object,
    *,
    state_key: str,
    item_key: str,
    value: Any,
) -> None:
    """Store a keyed tool result without losing previous same-tool outputs."""
    state = get_state(context)
    if state is None:
        return

    raw_bucket = state.get(state_key)
    if isinstance(raw_bucket, dict):
        bucket = cast("dict[str, Any]", raw_bucket)
    else:
        bucket: dict[str, Any] = {}
        state[state_key] = bucket
    bucket[item_key] = value
