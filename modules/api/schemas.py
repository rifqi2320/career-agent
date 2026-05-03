"""Pydantic schemas for the FastAPI surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models.job import MatchJobStatus
from models.match import MatchOutput
from modules.tools.score_candidate_against_requirements import (
    CandidateProfileInputSchema,
)


class CandidateCreateRequest(BaseModel):
    """JSON request for creating a candidate profile."""

    resume_text: str | None = Field(default=None, min_length=1)
    profile: CandidateProfileInputSchema | None = None


class CandidateCreateResponse(BaseModel):
    """Response after candidate profile ingestion."""

    candidate_id: str
    profile: CandidateProfileInputSchema


class MatchCreateRequest(BaseModel):
    """Request to enqueue one match job per JD."""

    candidate_id: str = Field(min_length=1)
    job_descriptions: list[str] = Field(min_length=1, max_length=10)
    job_market_context: str = Field(default="unknown", min_length=1)


class MatchQueuedItem(BaseModel):
    """One queued job response item."""

    job_id: str
    status: Literal["pending"]


class MatchCreateResponse(BaseModel):
    """Response returned immediately after enqueueing jobs."""

    jobs: list[MatchQueuedItem]


class MatchJobResponse(BaseModel):
    """HTTP representation of one match job."""

    job_id: str
    candidate_id: str
    status: MatchJobStatus
    attempts: int
    max_attempts: int
    error_detail: str | None = None
    result: MatchOutput | None = None
    agent_trace: dict[str, object] | None = None


class MatchListResponse(BaseModel):
    """Paginated match job list response."""

    items: list[MatchJobResponse]
    limit: int
    offset: int
