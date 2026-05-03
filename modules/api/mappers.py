"""Mapping helpers for FastAPI response schemas."""

from __future__ import annotations

from typing import cast

from models.job import MatchJob, MatchJobStatus
from models.match import MatchOutput
from modules.api.schemas import MatchJobResponse


def match_job_response(job: MatchJob) -> MatchJobResponse:
    """Map a persisted match job to its HTTP response schema."""
    result = match_output_or_none(job.result)
    return MatchJobResponse(
        job_id=job.id,
        candidate_id=job.candidate_id,
        status=MatchJobStatus(job.status),
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        error_detail=job.error_detail,
        result=result,
        agent_trace=trace_payload(job.agent_trace, result),
    )


def match_output_or_none(payload: dict[str, object] | None) -> MatchOutput | None:
    """Return a validated match output when a job has a stored result."""
    if payload is None:
        return None
    return MatchOutput.model_validate(payload)


def trace_payload(
    stored_trace: dict[str, object] | None,
    result: MatchOutput | None,
) -> dict[str, object] | None:
    """Return the stored trace, falling back to the result trace for old rows."""
    if stored_trace is not None:
        return stored_trace
    if result is None:
        return None
    return cast("dict[str, object]", result.agent_trace.model_dump(mode="json"))
