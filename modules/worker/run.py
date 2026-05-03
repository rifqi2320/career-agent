"""RabbitMQ worker entrypoint for async career-match jobs."""

from __future__ import annotations

import asyncio

from models.match import AgentTrace, AgentStreamEvent, MatchOutput, ToolCallTrace
from modules.agent_runtime import CareerMatchRunInput, stream_career_match_events
from modules.logging import logging
from modules.matches.repository import (
    ClaimedMatchJob,
    claim_match_job,
    complete_match_job,
    fail_or_retry_match_job,
    list_due_pending_match_job_ids,
    requeue_stale_processing_jobs,
)
from modules.task_queue.rabbitmq import consume_match_jobs, publish_match_job
from modules.candidates.schemas import CandidateProfileInputSchema

RETRY_SCHEDULER_INTERVAL_SECONDS = 10


async def main() -> None:
    """Start consuming match jobs from RabbitMQ."""
    logging.info("career match worker starting")
    recovered = requeue_stale_processing_jobs()
    if recovered:
        logging.warning("recovered stale processing jobs | count=%d", recovered)
    scheduler = asyncio.create_task(_publish_due_jobs_forever())
    try:
        await consume_match_jobs(process_job_id)
    finally:
        scheduler.cancel()


async def process_job_id(job_id: str) -> None:
    """Claim and process one queued job ID."""
    claimed_job = claim_match_job(job_id)
    if claimed_job is None:
        logging.info("match job skipped | job_id=%s reason=not_pending", job_id)
        return

    logging.info(
        "match job claimed | job_id=%s attempt=%d max_attempts=%d",
        claimed_job.job_id,
        claimed_job.attempt,
        claimed_job.max_attempts,
    )
    partial_events: list[AgentStreamEvent] = []
    try:
        output = None
        async for event in stream_career_match_events(_run_input(claimed_job)):
            partial_events.append(event)
            if event.event_type == "run_completed":
                raw_output = event.payload.get("output")
                if isinstance(raw_output, dict):
                    output = raw_output
        if output is None:
            raise RuntimeError("Agent run did not produce a final output.")

        match_output = MatchOutput.model_validate(output)
        complete_match_job(job_id=claimed_job.job_id, output=match_output)
        logging.info("match job completed | job_id=%s", claimed_job.job_id)
    except Exception as error:
        await _handle_job_failure(
            claimed_job=claimed_job,
            error=error,
            partial_trace=_partial_trace(partial_events),
        )


def _run_input(claimed_job: ClaimedMatchJob) -> CareerMatchRunInput:
    return CareerMatchRunInput(
        candidate_profile=CandidateProfileInputSchema.model_validate(
            claimed_job.candidate_profile
        ),
        job_url_or_text=claimed_job.job_input,
        job_id=claimed_job.job_id,
        job_market_context=claimed_job.job_market_context,
    )


async def _handle_job_failure(
    *,
    claimed_job: ClaimedMatchJob,
    error: Exception,
    partial_trace: AgentTrace,
) -> None:
    retry_decision = fail_or_retry_match_job(
        job_id=claimed_job.job_id,
        error_detail=f"{type(error).__name__}: {error}",
        agent_trace=partial_trace,
    )
    logging.exception(
        "match job failed | job_id=%s attempt=%d retry=%s delay_seconds=%d",
        claimed_job.job_id,
        claimed_job.attempt,
        retry_decision.should_retry,
        retry_decision.delay_seconds,
    )


async def _publish_due_jobs_forever() -> None:
    while True:
        await _publish_due_jobs_once()
        await asyncio.sleep(RETRY_SCHEDULER_INTERVAL_SECONDS)


async def _publish_due_jobs_once() -> None:
    due_job_ids = list_due_pending_match_job_ids()
    for job_id in due_job_ids:
        await publish_match_job(job_id)


def _partial_trace(events: list[AgentStreamEvent]) -> AgentTrace:
    tool_calls: list[ToolCallTrace] = []
    for event in events:
        if event.event_type != "tool_response" or event.tool is None:
            continue
        error_type = event.payload.get("error_type")
        tool_calls.append(
            ToolCallTrace(
                tool=event.tool,
                status="error" if event.status == "error" else "success",
                latency_ms=0,
                error_type=error_type if isinstance(error_type, str) else None,
            )
        )
    return AgentTrace(tool_calls=tool_calls)


if __name__ == "__main__":
    asyncio.run(main())
