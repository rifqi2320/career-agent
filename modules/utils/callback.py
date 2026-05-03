"""ADK callback helpers."""

from __future__ import annotations

from typing import Any
from typing import cast

from google.adk.agents.context import Context
from google.adk.tools.base_tool import BaseTool

from modules.error.common import (
    ConfigurationError,
    DependencyError,
    ToolError,
    ToolExecutionError,
    ToolInputError,
    ToolTimeoutError,
    UnexpectedToolError,
    ValidationError,
)
from modules.logging import logging
from modules.utils.trace import (
    append_tool_trace,
    begin_tool_trace,
    get_state,
    increment_llm_calls,
)


def handle_before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: Context,
) -> None:
    """Record the start of a tool call for runtime trace construction."""
    del args
    begin_tool_trace(tool, tool_context)


def handle_after_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: Context,
    tool_response: dict[str, Any],
) -> None:
    """Record a successful tool call for runtime trace construction."""
    del args
    if isinstance(tool_response, dict) and "error" in tool_response:
        return
    append_tool_trace(tool, tool_context, status="success")


def handle_before_model_callback(callback_context: Context, llm_request: Any) -> None:
    """Count model calls made through ADK orchestration."""
    del llm_request
    increment_llm_calls(callback_context)


def handle_tool_error_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: Context,
    error: Exception,
) -> dict[str, Any]:
    """Convert tool exceptions into structured payloads for the agent."""
    del args

    error_payload = _build_tool_error_payload(tool, error)
    _store_tool_error(tool_context, error_payload)
    append_tool_trace(
        tool,
        tool_context,
        status="error",
        error_type=str(error_payload["error_type"]),
        message=str(error_payload["message"]),
    )
    logging.warning(
        "tool error handled | tool=%s error_type=%s retriable=%s message=%s",
        error_payload["tool"],
        error_payload["error_type"],
        error_payload["retriable"],
        error_payload["message"],
    )
    return {"error": error_payload}


def _build_tool_error_payload(
    tool: BaseTool,
    error: Exception,
) -> dict[str, object]:
    tool_name = getattr(tool, "name", None) or type(tool).__name__
    tool_error = normalize_tool_error(error)

    payload: dict[str, object] = {
        "tool": str(tool_name),
        "status": "error",
        "error_type": type(tool_error).__name__,
        "message": str(tool_error),
        "retriable": tool_error.retriable,
        "retry_guidance": (
            "Retry once with the same or narrower inputs, then use partial data if it fails again."
            if tool_error.retriable
            else "Do not retry unless the inputs can be corrected or additional context is available."
        ),
    }
    if tool_error.original_error_type is not None:
        payload["original_error_type"] = tool_error.original_error_type
    return payload


def normalize_tool_error(error: Exception) -> ToolError:
    """Translate arbitrary exceptions into project tool errors with retry metadata."""
    if isinstance(error, ToolError):
        return error
    if isinstance(error, TimeoutError):
        return ToolTimeoutError(str(error), original_error=error)
    if isinstance(error, ValidationError):
        return ToolInputError(str(error), original_error=error)
    if isinstance(error, ConfigurationError | DependencyError):
        return ToolExecutionError(str(error), original_error=error)
    if isinstance(error, ValueError):
        return ToolInputError(str(error), original_error=error)
    return UnexpectedToolError(str(error), original_error=error)


def _store_tool_error(context: Context, error_payload: dict[str, object]) -> None:
    state = get_state(context)
    if state is None:
        return

    state["last_tool_error"] = error_payload
    raw_errors = state.get("tool_errors")
    if isinstance(raw_errors, list):
        cast("list[dict[str, object]]", raw_errors).append(error_payload)
    else:
        state["tool_errors"] = [error_payload]
