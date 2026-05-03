from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

from google.adk.agents.context import Context
from google.adk.tools.base_tool import BaseTool

from modules.error.common import RetryableModelOutputError
from modules.utils.callback import (
    handle_after_tool_callback,
    handle_before_model_callback,
    handle_before_tool_callback,
    handle_tool_error_callback,
)


@dataclass
class DummyContext:
    state: dict[str, object] = field(default_factory=dict)


def test_tool_error_callback_returns_structured_retriable_payload() -> None:
    context = cast("Context", DummyContext())
    tool = cast("BaseTool", SimpleNamespace(name="extract_jd_requirements"))

    payload = handle_tool_error_callback(
        tool,
        {},
        context,
        RetryableModelOutputError("schema mismatch"),
    )

    assert payload == {
        "error": {
            "tool": "extract_jd_requirements",
            "status": "error",
            "error_type": "RetryableModelOutputError",
            "message": "schema mismatch",
            "retriable": True,
            "retry_guidance": (
                "Retry once with the same or narrower inputs, "
                "then use partial data if it fails again."
            ),
        }
    }
    assert context.state["last_tool_error"] == payload["error"]
    assert context.state["tool_errors"] == [payload["error"]]
    assert context.state["tool_calls"] == [
        {
            "tool": "extract_jd_requirements",
            "status": "error",
            "latency_ms": 0,
            "error_type": "RetryableModelOutputError",
            "message": "schema mismatch",
        }
    ]


def test_tool_error_callback_marks_validation_error_non_retriable() -> None:
    context = cast("Context", DummyContext())
    tool = cast("BaseTool", SimpleNamespace(name="research_skill_resources"))

    payload = handle_tool_error_callback(tool, {}, context, ValueError("missing skill"))

    assert payload["error"]["tool"] == "research_skill_resources"
    assert payload["error"]["error_type"] == "ToolInputError"
    assert payload["error"]["original_error_type"] == "ValueError"
    assert payload["error"]["retriable"] is False


def test_tool_error_callback_translates_stray_error_to_project_error() -> None:
    context = cast("Context", DummyContext())
    tool = cast("BaseTool", SimpleNamespace(name="research_skill_resources"))

    payload = handle_tool_error_callback(tool, {}, context, RuntimeError("db failed"))

    assert payload["error"]["tool"] == "research_skill_resources"
    assert payload["error"]["error_type"] == "UnexpectedToolError"
    assert payload["error"]["original_error_type"] == "RuntimeError"
    assert payload["error"]["retriable"] is False


def test_tool_callbacks_record_success_trace() -> None:
    context = cast("Context", DummyContext())
    tool = cast("BaseTool", SimpleNamespace(name="extract_jd_requirements"))

    handle_before_tool_callback(tool, {}, context)
    handle_after_tool_callback(tool, {}, context, {})

    assert context.state["tool_calls"][0]["tool"] == "extract_jd_requirements"
    assert context.state["tool_calls"][0]["status"] == "success"
    assert context.state["tool_calls"][0]["latency_ms"] >= 0


def test_after_tool_callback_does_not_mark_handled_error_as_success() -> None:
    context = cast("Context", DummyContext())
    tool = cast("BaseTool", SimpleNamespace(name="extract_jd_requirements"))

    handle_before_tool_callback(tool, {}, context)
    handle_tool_error_callback(tool, {}, context, RuntimeError("read failed"))
    handle_after_tool_callback(
        tool,
        {},
        context,
        {"error": {"message": "read failed"}},
    )

    assert context.state["tool_calls"] == [
        {
            "tool": "extract_jd_requirements",
            "status": "error",
            "latency_ms": 0,
            "error_type": "UnexpectedToolError",
            "message": "read failed",
        }
    ]


def test_before_model_callback_counts_llm_calls() -> None:
    context = cast("Context", DummyContext())

    handle_before_model_callback(context, object())
    handle_before_model_callback(context, object())

    assert context.state["total_llm_calls"] == 2
