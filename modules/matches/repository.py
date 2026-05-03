"""Persistence operations for candidate profiles and async match jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from models.job import CandidateProfile, MatchJob, MatchJobStatus
from models.match import AgentTrace, MatchOutput
from modules.database.client import SessionLocal
from modules.error.common import ToolInputError


@dataclass(frozen=True, slots=True)
class ClaimedMatchJob:
    """Data needed by a worker after it atomically claims a job."""

    job_id: str
    candidate_profile: dict[str, object]
    job_input: str
    job_market_context: str
    attempt: int
    max_attempts: int


def create_candidate_profile(
    *,
    profile: dict[str, object],
    source_type: str,
) -> CandidateProfile:
    """Persist one structured candidate profile."""
    candidate = CandidateProfile(
        id=str(uuid4()),
        profile=profile,
        source_type=source_type,
    )
    with SessionLocal() as session:
        session.add(candidate)
        session.commit()
        session.refresh(candidate)
        session.expunge(candidate)
    return candidate


def get_candidate_profile(candidate_id: str) -> CandidateProfile | None:
    """Return one stored candidate profile by ID."""
    with SessionLocal() as session:
        candidate = session.get(CandidateProfile, candidate_id)
        if candidate is None:
            return None
        session.expunge(candidate)
        return candidate


def create_match_jobs(
    *,
    candidate_id: str,
    job_inputs: list[str],
    job_market_context: str,
) -> list[MatchJob]:
    """Create pending match jobs for a stored candidate profile."""
    if len(job_inputs) > 10:
        raise ToolInputError("At most 10 job descriptions may be submitted.")
    jobs = [
        MatchJob(
            id=str(uuid4()),
            candidate_id=candidate_id,
            status=MatchJobStatus.PENDING.value,
            job_input=job_input,
            job_market_context=job_market_context,
            attempts=0,
            max_attempts=3,
        )
        for job_input in job_inputs
    ]
    with SessionLocal() as session:
        if session.get(CandidateProfile, candidate_id) is None:
            raise ToolInputError("candidate_id does not exist.")
        session.add_all(jobs)
        session.commit()
        for job in jobs:
            session.refresh(job)
            session.expunge(job)
    return jobs


def get_match_job(job_id: str) -> MatchJob | None:
    """Return one match job."""
    with SessionLocal() as session:
        job = session.get(MatchJob, job_id)
        if job is None:
            return None
        session.expunge(job)
        return job


def list_match_jobs(
    *,
    limit: int,
    offset: int,
    status: MatchJobStatus | None = None,
) -> list[MatchJob]:
    """Return a paginated match-job list, optionally filtered by status."""
    query: Select[tuple[MatchJob]] = select(MatchJob).order_by(
        MatchJob.created_at.desc()
    )
    if status is not None:
        query = query.where(MatchJob.status == status.value)
    query = query.limit(limit).offset(offset)
    with SessionLocal() as session:
        jobs = list(session.scalars(query).all())
        for job in jobs:
            session.expunge(job)
        return jobs


def claim_match_job(job_id: str) -> ClaimedMatchJob | None:
    """Atomically claim one pending job for a worker.

    PostgreSQL `FOR UPDATE SKIP LOCKED` prevents two worker processes from taking the
    same pending row if duplicate RabbitMQ messages arrive.
    """
    with SessionLocal() as session, session.begin():
        job = _select_pending_job_for_update(session, job_id)
        if job is None:
            return None
        candidate = session.get(CandidateProfile, job.candidate_id)
        if candidate is None:
            job.status = MatchJobStatus.FAILED.value
            job.error_detail = "Candidate profile was not found."
            job.failed_at = _utcnow()
            return None

        job.status = MatchJobStatus.PROCESSING.value
        job.attempts += 1
        job.processing_started_at = _utcnow()
        job.error_detail = None
        return ClaimedMatchJob(
            job_id=job.id,
            candidate_profile=dict(candidate.profile),
            job_input=job.job_input,
            job_market_context=job.job_market_context,
            attempt=job.attempts,
            max_attempts=job.max_attempts,
        )


def complete_match_job(*, job_id: str, output: MatchOutput) -> None:
    """Mark a processing job completed with validated agent output."""
    payload = output.model_dump(mode="json")
    with SessionLocal() as session, session.begin():
        job = session.get(MatchJob, job_id)
        if job is None:
            raise ToolInputError("match job does not exist.")
        job.status = MatchJobStatus.COMPLETED.value
        job.result = payload
        job.agent_trace = output.agent_trace.model_dump(mode="json")
        job.error_detail = None
        job.completed_at = _utcnow()


def fail_or_retry_match_job(
    *,
    job_id: str,
    error_detail: str,
    agent_trace: AgentTrace | None = None,
) -> bool:
    """Store failure details and return whether the job should be requeued."""
    with SessionLocal() as session, session.begin():
        job = session.get(MatchJob, job_id)
        if job is None:
            raise ToolInputError("match job does not exist.")
        job.error_detail = error_detail
        if agent_trace is not None:
            job.agent_trace = agent_trace.model_dump(mode="json")
        if job.attempts >= job.max_attempts:
            job.status = MatchJobStatus.FAILED.value
            job.failed_at = _utcnow()
            return False
        job.status = MatchJobStatus.PENDING.value
        return True


def requeue_failed_match_job(job_id: str) -> bool:
    """Move a failed job back to pending for an explicit admin retry."""
    with SessionLocal() as session, session.begin():
        job = session.get(MatchJob, job_id)
        if job is None:
            raise ToolInputError("match job does not exist.")
        if job.status != MatchJobStatus.FAILED.value:
            return False
        job.status = MatchJobStatus.PENDING.value
        job.attempts = 0
        job.error_detail = None
        job.failed_at = None
        return True


def _select_pending_job_for_update(
    session: Session,
    job_id: str,
) -> MatchJob | None:
    statement = (
        select(MatchJob)
        .where(
            MatchJob.id == job_id,
            MatchJob.status == MatchJobStatus.PENDING.value,
        )
        .with_for_update(skip_locked=True)
    )
    return session.scalar(statement)


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)
