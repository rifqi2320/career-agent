"""Persistence operations for candidate profiles and async match jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re
from uuid import uuid4

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from models.job import CandidateProfile, MatchJob, MatchJobStatus
from models.match import AgentTrace, MatchOutput
from modules.database.client import create_session
from modules.error.common import ToolInputError

BASE_RETRY_DELAY_SECONDS = 60
MAX_RETRY_DELAY_SECONDS = 900
RETRY_DELAY_PATTERN = re.compile(r"retry(?:Delay| in)[^0-9]*(?P<seconds>\d+)s")


@dataclass(frozen=True, slots=True)
class ClaimedMatchJob:
    """Data needed by a worker after it atomically claims a job."""

    job_id: str
    candidate_profile: dict[str, object]
    job_input: str
    job_market_context: str
    attempt: int
    max_attempts: int


@dataclass(frozen=True, slots=True)
class MatchRetryDecision:
    """Outcome after storing a failed match-job attempt."""

    should_retry: bool
    delay_seconds: int = 0
    next_attempt_at: datetime | None = None


def create_candidate_profile(
    *,
    profile: dict[str, object],
    source_type: str,
) -> CandidateProfile:
    """Persist one structured candidate profile."""
    if not profile:
        raise ToolInputError("candidate profile must not be empty.")
    normalized_source_type = source_type.strip()
    if not normalized_source_type:
        raise ToolInputError("source_type must not be empty.")
    candidate = CandidateProfile(
        id=str(uuid4()),
        profile=profile,
        source_type=normalized_source_type,
    )
    with create_session() as session:
        session.add(candidate)
        session.commit()
        session.refresh(candidate)
        session.expunge(candidate)
    return candidate


def create_match_jobs(
    *,
    candidate_id: str,
    job_inputs: list[str],
    job_market_context: str,
) -> list[MatchJob]:
    """Create pending match jobs for a stored candidate profile."""
    normalized_candidate_id = candidate_id.strip()
    if not normalized_candidate_id:
        raise ToolInputError("candidate_id must not be empty.")
    normalized_job_inputs = [
        job_input.strip() for job_input in job_inputs if job_input.strip()
    ]
    if not normalized_job_inputs:
        raise ToolInputError("job_inputs must include at least one non-empty item.")
    if len(normalized_job_inputs) > 10:
        raise ToolInputError("At most 10 job descriptions may be submitted.")
    normalized_job_market_context = job_market_context.strip() or "unknown"
    jobs = [
        MatchJob(
            id=str(uuid4()),
            candidate_id=normalized_candidate_id,
            status=MatchJobStatus.PENDING.value,
            job_input=job_input,
            job_market_context=normalized_job_market_context,
            attempts=0,
            max_attempts=3,
            next_attempt_at=None,
        )
        for job_input in normalized_job_inputs
    ]
    with create_session() as session:
        if session.get(CandidateProfile, normalized_candidate_id) is None:
            raise ToolInputError("candidate_id does not exist.")
        session.add_all(jobs)
        session.commit()
        for job in jobs:
            session.refresh(job)
            session.expunge(job)
    return jobs


def get_match_job(job_id: str) -> MatchJob | None:
    """Return one match job."""
    with create_session() as session:
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
    with create_session() as session:
        jobs = list(session.scalars(query).all())
        for job in jobs:
            session.expunge(job)
        return jobs


def claim_match_job(job_id: str) -> ClaimedMatchJob | None:
    """Atomically claim one pending job for a worker.

    PostgreSQL `FOR UPDATE SKIP LOCKED` prevents two worker processes from taking the
    same pending row if duplicate RabbitMQ messages arrive.
    """
    with create_session() as session, session.begin():
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
        job.next_attempt_at = None
        job.error_detail = None
        return ClaimedMatchJob(
            job_id=job.id,
            candidate_profile=dict(candidate.profile),
            job_input=job.job_input,
            job_market_context=job.job_market_context,
            attempt=job.attempts,
            max_attempts=job.max_attempts,
        )


def requeue_stale_processing_jobs(
    *,
    older_than_seconds: int = 1800,
    limit: int = 100,
) -> int:
    """Move stale processing jobs back to pending for worker crash recovery."""
    if older_than_seconds <= 0:
        raise ToolInputError("older_than_seconds must be greater than 0.")
    if limit <= 0:
        raise ToolInputError("limit must be greater than 0.")

    cutoff = _utcnow().timestamp() - older_than_seconds
    cutoff_at = datetime.fromtimestamp(cutoff, tz=UTC)
    with create_session() as session, session.begin():
        statement = (
            select(MatchJob)
            .where(
                MatchJob.status == MatchJobStatus.PROCESSING.value,
                MatchJob.processing_started_at < cutoff_at,
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        jobs = list(session.scalars(statement).all())
        for job in jobs:
            if job.attempts >= job.max_attempts:
                job.status = MatchJobStatus.FAILED.value
                job.failed_at = _utcnow()
                job.error_detail = (
                    job.error_detail or "Job was abandoned while processing."
                )
                continue
            job.status = MatchJobStatus.PENDING.value
            job.error_detail = "Job was recovered after stale processing state."
            job.processing_started_at = None
            job.next_attempt_at = _utcnow()
    return len(jobs)


def list_due_pending_match_job_ids(*, limit: int = 100) -> list[str]:
    """Return pending job IDs whose retry schedule is due."""
    if limit <= 0:
        raise ToolInputError("limit must be greater than 0.")

    now = _utcnow()
    statement = (
        select(MatchJob.id)
        .where(
            MatchJob.status == MatchJobStatus.PENDING.value,
            or_(MatchJob.next_attempt_at.is_(None), MatchJob.next_attempt_at <= now),
        )
        .order_by(MatchJob.created_at.asc())
        .limit(limit)
    )
    with create_session() as session:
        return list(session.scalars(statement).all())


def complete_match_job(*, job_id: str, output: MatchOutput) -> None:
    """Mark a processing job completed with validated agent output."""
    payload = output.model_dump(mode="json")
    with create_session() as session, session.begin():
        job = session.get(MatchJob, job_id)
        if job is None:
            raise ToolInputError("match job does not exist.")
        job.status = MatchJobStatus.COMPLETED.value
        job.result = payload
        job.agent_trace = output.agent_trace.model_dump(mode="json")
        job.error_detail = None
        job.next_attempt_at = None
        job.completed_at = _utcnow()


def fail_or_retry_match_job(
    *,
    job_id: str,
    error_detail: str,
    agent_trace: AgentTrace | None = None,
) -> MatchRetryDecision:
    """Store failure details and return the retry schedule decision."""
    with create_session() as session, session.begin():
        job = session.get(MatchJob, job_id)
        if job is None:
            raise ToolInputError("match job does not exist.")
        job.error_detail = error_detail
        if agent_trace is not None:
            job.agent_trace = agent_trace.model_dump(mode="json")
        if job.attempts >= job.max_attempts:
            job.status = MatchJobStatus.FAILED.value
            job.failed_at = _utcnow()
            job.next_attempt_at = None
            return MatchRetryDecision(should_retry=False)
        delay_seconds = _retry_delay_seconds(
            error_detail=error_detail,
            attempt=job.attempts,
        )
        next_attempt_at = _utcnow() + timedelta(seconds=delay_seconds)
        job.status = MatchJobStatus.PENDING.value
        job.next_attempt_at = next_attempt_at
        job.processing_started_at = None
        return MatchRetryDecision(
            should_retry=True,
            delay_seconds=delay_seconds,
            next_attempt_at=next_attempt_at,
        )


def requeue_failed_match_job(job_id: str) -> bool:
    """Move a failed job back to pending for an explicit admin retry."""
    with create_session() as session, session.begin():
        job = session.get(MatchJob, job_id)
        if job is None:
            raise ToolInputError("match job does not exist.")
        if job.status != MatchJobStatus.FAILED.value:
            return False
        job.status = MatchJobStatus.PENDING.value
        job.attempts = 0
        job.error_detail = None
        job.failed_at = None
        job.next_attempt_at = None
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
            or_(
                MatchJob.next_attempt_at.is_(None),
                MatchJob.next_attempt_at <= _utcnow(),
            ),
        )
        .with_for_update(skip_locked=True)
    )
    return session.scalar(statement)


def _retry_delay_seconds(*, error_detail: str, attempt: int) -> int:
    exponential_delay = min(
        BASE_RETRY_DELAY_SECONDS * (2 ** max(0, attempt - 1)),
        MAX_RETRY_DELAY_SECONDS,
    )
    hinted_delay = _retry_delay_hint_seconds(error_detail)
    if hinted_delay is None:
        return exponential_delay
    return min(max(exponential_delay, hinted_delay), MAX_RETRY_DELAY_SECONDS)


def _retry_delay_hint_seconds(error_detail: str) -> int | None:
    match = RETRY_DELAY_PATTERN.search(error_detail)
    if match is None:
        return None
    return int(match.group("seconds"))


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)
