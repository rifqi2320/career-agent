"""FastAPI entrypoint for async career matching."""

from __future__ import annotations

import json
from typing import cast

from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import ValidationError as PydanticValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from models.job import MatchJob, MatchJobStatus
from models.match import MatchOutput
from modules.api.schemas import (
    CandidateCreateRequest,
    CandidateCreateResponse,
    MatchCreateRequest,
    MatchCreateResponse,
    MatchJobResponse,
    MatchListResponse,
    MatchQueuedItem,
)
from modules.candidates.profile import (
    profile_from_structured_payload,
    profile_from_text,
    text_from_pdf_bytes,
)
from modules.error.common import ToolInputError
from modules.matches.repository import (
    create_candidate_profile,
    create_match_jobs,
    get_match_job,
    list_match_jobs,
    requeue_failed_match_job,
)
from modules.task_queue.rabbitmq import publish_match_job
from modules.tools.score_candidate_against_requirements import (
    CandidateProfileInputSchema,
)

app = FastAPI(title="Career Agent API", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Return process health for container checks."""
    return {"status": "ok"}


@app.post(
    "/api/v1/candidate",
    response_model=CandidateCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_candidate(request: Request) -> CandidateCreateResponse:
    """Ingest a candidate profile from JSON, text, or PDF upload."""
    profile, source_type = await _profile_from_request(request)
    candidate = create_candidate_profile(
        profile=profile.model_dump(mode="json"),
        source_type=source_type,
    )
    return CandidateCreateResponse(candidate_id=candidate.id, profile=profile)


@app.post(
    "/api/v1/matches",
    response_model=MatchCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_matches(payload: MatchCreateRequest) -> MatchCreateResponse:
    """Create pending match jobs and enqueue one durable RabbitMQ task per JD."""
    job_inputs = [job.strip() for job in payload.job_descriptions if job.strip()]
    if not job_inputs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="job_descriptions must contain at least one non-empty item.",
        )
    try:
        jobs = create_match_jobs(
            candidate_id=payload.candidate_id,
            job_inputs=job_inputs,
            job_market_context=payload.job_market_context,
        )
        for job in jobs:
            await publish_match_job(job.id)
    except ToolInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    return MatchCreateResponse(
        jobs=[MatchQueuedItem(job_id=job.id, status="pending") for job in jobs]
    )


@app.get("/api/v1/matches/{job_id}", response_model=MatchJobResponse)
async def read_match(job_id: str) -> MatchJobResponse:
    """Return one match job status and result."""
    job = get_match_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found."
        )
    return _match_job_response(job)


@app.get("/api/v1/matches", response_model=MatchListResponse)
async def read_matches(
    *,
    limit: int = Query(ge=1, le=100),
    offset: int = Query(ge=0),
    status_filter: MatchJobStatus | None = Query(default=None, alias="status"),
) -> MatchListResponse:
    """Return a paginated list of match jobs."""
    jobs = list_match_jobs(limit=limit, offset=offset, status=status_filter)
    return MatchListResponse(
        items=[_match_job_response(job) for job in jobs],
        limit=limit,
        offset=offset,
    )


@app.post("/api/v1/matches/{job_id}/requeue", response_model=MatchQueuedItem)
async def requeue_match(job_id: str) -> MatchQueuedItem:
    """Admin endpoint to requeue a failed match job."""
    try:
        requeued = requeue_failed_match_job(job_id)
    except ToolInputError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
    if not requeued:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only failed jobs can be requeued.",
        )
    await publish_match_job(job_id)
    return MatchQueuedItem(job_id=job_id, status="pending")


async def _profile_from_request(
    request: Request,
) -> tuple[CandidateProfileInputSchema, str]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        return await _profile_from_multipart(request)

    try:
        payload = CandidateCreateRequest.model_validate(await request.json())
    except (PydanticValidationError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Expected JSON body with `profile` or `resume_text`.",
        ) from error

    if payload.profile is not None:
        return payload.profile, "structured_json"
    if payload.resume_text is not None:
        return _profile_from_text_or_400(payload.resume_text), "text"
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Either `profile` or `resume_text` is required.",
    )


async def _profile_from_multipart(
    request: Request,
) -> tuple[CandidateProfileInputSchema, str]:
    form = await request.form()
    raw_profile = form.get("profile")
    if isinstance(raw_profile, str) and raw_profile.strip():
        try:
            profile_payload = json.loads(raw_profile)
            return profile_from_structured_payload(profile_payload), "structured_json"
        except (json.JSONDecodeError, ToolInputError) as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error

    raw_text = form.get("resume_text")
    if isinstance(raw_text, str) and raw_text.strip():
        return _profile_from_text_or_400(raw_text), "text"

    upload = form.get("resume_file")
    if isinstance(upload, StarletteUploadFile):
        pdf_bytes = await upload.read()
        text = _pdf_text_or_400(pdf_bytes)
        return _profile_from_text_or_400(text), "pdf"

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Multipart body must include profile, resume_text, or resume_file.",
    )


def _profile_from_text_or_400(text: str) -> CandidateProfileInputSchema:
    try:
        return profile_from_text(text)
    except ToolInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error


def _pdf_text_or_400(pdf_bytes: bytes) -> str:
    try:
        return text_from_pdf_bytes(pdf_bytes)
    except ToolInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error


def _match_job_response(job: MatchJob) -> MatchJobResponse:
    result = _match_output_or_none(job.result)
    return MatchJobResponse(
        job_id=job.id,
        candidate_id=job.candidate_id,
        status=MatchJobStatus(job.status),
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        error_detail=job.error_detail,
        result=result,
        agent_trace=_trace_payload(job.agent_trace, result),
    )


def _match_output_or_none(payload: dict[str, object] | None) -> MatchOutput | None:
    if payload is None:
        return None
    return MatchOutput.model_validate(payload)


def _trace_payload(
    stored_trace: dict[str, object] | None,
    result: MatchOutput | None,
) -> dict[str, object] | None:
    if stored_trace is not None:
        return stored_trace
    if result is None:
        return None
    return cast("dict[str, object]", result.agent_trace.model_dump(mode="json"))
