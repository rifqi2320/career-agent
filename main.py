"""FastAPI entrypoint for async career matching."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request, status

from models.job import MatchJobStatus
from modules.api.ingestion import profile_from_request
from modules.api.mappers import match_job_response
from modules.api.schemas import (
    CandidateCreateResponse,
    MatchCreateRequest,
    MatchCreateResponse,
    MatchJobResponse,
    MatchListResponse,
    MatchQueuedItem,
)
from modules.error.common import ToolInputError
from modules.matches.repository import (
    create_candidate_profile,
    create_match_jobs,
    get_match_job,
    list_match_jobs,
    requeue_failed_match_job,
)
from modules.task_queue.rabbitmq import publish_match_job, publish_match_jobs

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
    profile, source_type = await profile_from_request(request)
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
        await publish_match_jobs([job.id for job in jobs])
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
    return match_job_response(job)


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
        items=[match_job_response(job) for job in jobs],
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
