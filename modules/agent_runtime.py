"""ADK runner orchestration for career-match jobs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, MutableMapping
from typing import cast
from uuid import uuid4

from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from agents.career_intelligence.agent import root_agent
from agents.career_intelligence.builder import CAREER_INTELLIGENCE_AGENT_NAME
from agents.career_intelligence.prompts import CAREER_MATCH_USER_PROMPT_TEMPLATE
from models.match import AgentRunState, AgentStreamEvent, MatchOutput
from modules.candidates.schemas import CandidateProfileInputSchema
from modules.utils.adk_events import (
    match_output_from_adk_event,
    register_runtime_event_sink,
    resequence_event,
    stream_events_from_adk_event,
    unregister_runtime_event_sink,
)

CAREER_MATCH_APP_NAME = "career_match"
DEFAULT_USER_ID = "career_match_worker"


class CareerMatchRunInput(BaseModel):
    """Input required to run one ADK career-match invocation."""

    candidate_profile: CandidateProfileInputSchema
    job_url_or_text: str = Field(min_length=1)
    job_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    job_market_context: str = "unknown"
    user_id: str = DEFAULT_USER_ID


async def stream_career_match_events(
    run_input: CareerMatchRunInput,
    *,
    agent: Agent = root_agent,
    session_service: BaseSessionService | None = None,
) -> AsyncIterator[AgentStreamEvent]:
    """Run the root ADK agent and yield sanitized events for every step."""
    queue: asyncio.Queue[AgentStreamEvent | None] = asyncio.Queue()
    sequence = 0
    producer_error: Exception | None = None

    def enqueue(event: AgentStreamEvent) -> None:
        nonlocal sequence
        sequence += 1
        queue.put_nowait(_with_sequence(event, sequence))

    async def produce_events() -> None:
        nonlocal producer_error
        register_runtime_event_sink(run_input.job_id, enqueue)
        try:
            await _produce_career_match_events(
                run_input=run_input,
                agent=agent,
                session_service=session_service,
                enqueue=enqueue,
            )
        except Exception as error:
            producer_error = error
            enqueue(
                AgentStreamEvent(
                    event_type="run_failed",
                    job_id=run_input.job_id,
                    author=CAREER_INTELLIGENCE_AGENT_NAME,
                    status="error",
                    payload={
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                )
            )
        finally:
            unregister_runtime_event_sink(run_input.job_id)
            queue.put_nowait(None)

    producer = asyncio.create_task(produce_events())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
        await producer
    finally:
        if not producer.done():
            producer.cancel()
    if producer_error is not None:
        raise producer_error


async def _produce_career_match_events(
    *,
    run_input: CareerMatchRunInput,
    agent: Agent,
    session_service: BaseSessionService | None,
    enqueue: Callable[[AgentStreamEvent], None],
) -> None:
    service = session_service or InMemorySessionService()
    session = await service.create_session(
        app_name=CAREER_MATCH_APP_NAME,
        user_id=run_input.user_id,
        session_id=run_input.job_id,
        state=AgentRunState(job_id=run_input.job_id).model_dump(
            mode="json",
            exclude_none=True,
        ),
    )
    runner = Runner(
        app_name=CAREER_MATCH_APP_NAME,
        agent=agent,
        session_service=service,
    )

    enqueue(
        AgentStreamEvent(
            event_type="run_started",
            job_id=run_input.job_id,
            author=CAREER_INTELLIGENCE_AGENT_NAME,
            status="started",
            payload={"session_id": session.id},
        )
    )

    final_output: MatchOutput | None = None
    async for event in runner.run_async(
        user_id=run_input.user_id,
        session_id=session.id,
        new_message=_build_user_message(run_input),
    ):
        extracted_output = match_output_from_adk_event(event)
        if extracted_output is not None:
            final_output = extracted_output

        for stream_event in stream_events_from_adk_event(
            event,
            job_id=run_input.job_id,
        ):
            enqueue(stream_event)

    if final_output is None:
        final_output = await _output_from_session_state(
            service=service,
            user_id=run_input.user_id,
            session_id=session.id,
        )

    enqueue(
        AgentStreamEvent(
            event_type="run_completed",
            job_id=run_input.job_id,
            author=CAREER_INTELLIGENCE_AGENT_NAME,
            status="success",
            payload={"output": final_output.model_dump(mode="json")},
        )
    )


def _build_user_message(run_input: CareerMatchRunInput) -> types.Content:
    prompt = CAREER_MATCH_USER_PROMPT_TEMPLATE.format(
        job_id=run_input.job_id,
        job_market_context=run_input.job_market_context,
        candidate_profile_json=run_input.candidate_profile.model_dump_json(indent=2),
        job_url_or_text=run_input.job_url_or_text,
    )
    return types.Content(role="user", parts=[types.Part.from_text(text=prompt)])


async def _session_state(
    *,
    service: BaseSessionService,
    user_id: str,
    session_id: str,
) -> MutableMapping[str, object]:
    session = await service.get_session(
        app_name=CAREER_MATCH_APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    if session is None or not isinstance(session.state, MutableMapping):
        return {}
    return cast("MutableMapping[str, object]", session.state)


async def _output_from_session_state(
    *,
    service: BaseSessionService,
    user_id: str,
    session_id: str,
) -> MatchOutput:
    state = await _session_state(
        service=service,
        user_id=user_id,
        session_id=session_id,
    )
    raw_output = state.get("final_match_output")
    if isinstance(raw_output, MatchOutput):
        return raw_output
    if isinstance(raw_output, dict):
        return MatchOutput.model_validate(raw_output)
    raise RuntimeError("ADK run did not call finalize_match_output successfully.")


def _with_sequence(event: AgentStreamEvent, sequence: int) -> AgentStreamEvent:
    return resequence_event(event, sequence)
