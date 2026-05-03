"""Helpers for converting ADK events into safe application stream events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from google.adk.events import Event
from google.genai import types

from models.match import AgentStreamEvent, MatchOutput
from modules.utils.trace import get_state

RUNTIME_EVENTS_STATE_KEY = "runtime_events"
_RUNTIME_EVENT_SINKS: dict[str, Callable[[AgentStreamEvent], None]] = {}


def register_runtime_event_sink(
    job_id: str,
    sink: Callable[[AgentStreamEvent], None],
) -> None:
    """Register an in-process sink for live nested runtime events."""
    _RUNTIME_EVENT_SINKS[job_id] = sink


def unregister_runtime_event_sink(job_id: str) -> None:
    """Remove an in-process nested runtime event sink."""
    _RUNTIME_EVENT_SINKS.pop(job_id, None)


def append_runtime_event(context: object, event: AgentStreamEvent) -> None:
    """Persist a sanitized stream event in mutable ADK state."""
    state = get_state(context)
    if state is None:
        return

    raw_events = state.get(RUNTIME_EVENTS_STATE_KEY)
    if isinstance(raw_events, list):
        events = cast("list[dict[str, Any]]", raw_events)
    else:
        events = []
        state[RUNTIME_EVENTS_STATE_KEY] = events
    events.append(event.model_dump(mode="json"))

    sink = _RUNTIME_EVENT_SINKS.get(event.job_id)
    if sink is not None:
        sink(event)


def stream_events_from_adk_event(
    event: Event,
    *,
    job_id: str,
) -> list[AgentStreamEvent]:
    """Convert one ADK event into zero or more sanitized stream events."""
    events: list[AgentStreamEvent] = []
    author = event.author

    for function_call in event.get_function_calls():
        events.append(_tool_call_event(job_id, author, function_call))

    for function_response in event.get_function_responses():
        events.append(_tool_response_event(job_id, author, function_response))

    if event.actions.state_delta:
        events.append(
            AgentStreamEvent(
                event_type="state_delta",
                job_id=job_id,
                author=author,
                payload={"keys": sorted(event.actions.state_delta.keys())},
            )
        )

    if event.error_code or event.error_message:
        events.append(
            AgentStreamEvent(
                event_type="run_failed",
                job_id=job_id,
                author=author,
                status="error",
                payload={
                    "error_code": event.error_code,
                    "error_message": event.error_message,
                },
            )
        )

    text = _event_text(event)
    if text:
        event_type = "final_response" if event.is_final_response() else "model_response"
        events.append(
            AgentStreamEvent(
                event_type=event_type,
                job_id=job_id,
                author=author,
                status="success" if event_type == "final_response" else None,
                payload={"text": text, "partial": bool(event.partial)},
            )
        )

    return events


def match_output_from_adk_event(event: Event) -> MatchOutput | None:
    """Extract the final MatchOutput from a finalize_match_output tool response."""
    for function_response in event.get_function_responses():
        if function_response.name != "finalize_match_output":
            continue
        output = _match_output_from_payload(function_response.response)
        if output is not None:
            return output
    return None


def resequence_event(event: AgentStreamEvent, sequence: int) -> AgentStreamEvent:
    """Return a copy with the caller-assigned stream sequence."""
    return event.model_copy(update={"sequence": sequence})


def _tool_call_event(
    job_id: str,
    author: str | None,
    function_call: types.FunctionCall,
) -> AgentStreamEvent:
    args = function_call.args if isinstance(function_call.args, dict) else {}
    return AgentStreamEvent(
        event_type="tool_call",
        job_id=job_id,
        author=author,
        tool=function_call.name,
        status="started",
        payload={
            "call_id": function_call.id,
            "arg_keys": sorted(args.keys()),
        },
    )


def _tool_response_event(
    job_id: str,
    author: str | None,
    function_response: types.FunctionResponse,
) -> AgentStreamEvent:
    response = (
        function_response.response
        if isinstance(function_response.response, dict)
        else {}
    )
    has_error = "error" in response
    payload: dict[str, Any] = {
        "call_id": function_response.id,
        "response_keys": sorted(response.keys()),
    }
    if has_error and isinstance(response.get("error"), dict):
        error_payload = cast("dict[str, Any]", response["error"])
        payload["error_type"] = error_payload.get("error_type")
        payload["retriable"] = error_payload.get("retriable")

    return AgentStreamEvent(
        event_type="tool_response",
        job_id=job_id,
        author=author,
        tool=function_response.name,
        status="error" if has_error else "success",
        payload=payload,
    )


def _event_text(event: Event) -> str:
    if event.content is None or not event.content.parts:
        return ""
    return "\n".join(
        part.text for part in event.content.parts if part.text and not part.thought
    ).strip()


def _match_output_from_payload(payload: object) -> MatchOutput | None:
    candidates = _payload_candidates(payload)
    for candidate in candidates:
        if isinstance(candidate, MatchOutput):
            return candidate
        if isinstance(candidate, dict):
            try:
                return MatchOutput.model_validate(candidate)
            except ValueError:
                continue
    return None


def _payload_candidates(payload: object) -> list[object]:
    candidates = [payload]
    if not isinstance(payload, dict):
        return candidates

    typed_payload = cast("dict[str, object]", payload)
    for key in ("result", "output", "response"):
        if key in typed_payload:
            candidates.append(typed_payload[key])
    return candidates
