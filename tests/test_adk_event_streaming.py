from __future__ import annotations

from google.adk.events import Event
from google.adk.tools import ToolContext
from google.genai import types

from models.match import AgentStreamEvent
from modules.utils.adk_events import (
    append_runtime_event,
    register_runtime_event_sink,
    stream_events_from_adk_event,
    unregister_runtime_event_sink,
)

JOB_ID = "user_1777818627811_l4nrvwv7p"


def test_stream_event_for_tool_call_redacts_argument_values() -> None:
    event = Event(
        author="career_intelligence",
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="extract_jd_requirements",
                    args={"url_or_text": "private job text"},
                )
            ],
        ),
    )

    stream_events = stream_events_from_adk_event(event, job_id=JOB_ID)

    assert len(stream_events) == 1
    assert stream_events[0].event_type == "tool_call"
    assert stream_events[0].tool == "extract_jd_requirements"
    assert stream_events[0].status == "started"
    assert stream_events[0].payload["arg_keys"] == ["url_or_text"]
    assert "private job text" not in str(stream_events[0].payload)


def test_stream_event_for_tool_response_records_status_and_keys_only() -> None:
    event = Event(
        author="career_intelligence",
        content=types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name="score_candidate_against_requirements",
                    response={"overall_score": 82, "gap_skills": ["rag"]},
                )
            ],
        ),
    )

    stream_events = stream_events_from_adk_event(event, job_id=JOB_ID)

    assert len(stream_events) == 1
    assert stream_events[0].event_type == "tool_response"
    assert stream_events[0].tool == "score_candidate_against_requirements"
    assert stream_events[0].status == "success"
    assert stream_events[0].payload["response_keys"] == [
        "gap_skills",
        "overall_score",
    ]
    assert "rag" not in str(stream_events[0].payload)


def test_append_runtime_event_writes_state_and_live_sink(
    tool_context: ToolContext,
) -> None:
    captured: list[AgentStreamEvent] = []
    event = AgentStreamEvent(
        event_type="tool_call",
        job_id=JOB_ID,
        tool="query_github_learning_resources",
        status="started",
    )
    register_runtime_event_sink(JOB_ID, captured.append)

    try:
        append_runtime_event(tool_context, event)
    finally:
        unregister_runtime_event_sink(JOB_ID)

    assert captured == [event]
    assert tool_context.state["runtime_events"] == [event.model_dump(mode="json")]
