from __future__ import annotations

import pytest

from modules.error.common import ToolInputError
from modules.matches.repository import (
    MAX_RETRY_DELAY_SECONDS,
    create_candidate_profile,
    create_match_jobs,
    requeue_stale_processing_jobs,
    _retry_delay_seconds,
)


def test_create_candidate_profile_rejects_empty_profile() -> None:
    with pytest.raises(ToolInputError, match="candidate profile"):
        create_candidate_profile(profile={}, source_type="text")


def test_create_candidate_profile_rejects_empty_source_type() -> None:
    with pytest.raises(ToolInputError, match="source_type"):
        create_candidate_profile(profile={"skills": ["python"]}, source_type="  ")


def test_create_match_jobs_rejects_empty_candidate_id() -> None:
    with pytest.raises(ToolInputError, match="candidate_id"):
        create_match_jobs(
            candidate_id=" ",
            job_inputs=["Build APIs"],
            job_market_context="unknown",
        )


def test_create_match_jobs_rejects_empty_job_inputs() -> None:
    with pytest.raises(ToolInputError, match="job_inputs"):
        create_match_jobs(
            candidate_id="11111111-1111-4111-8111-111111111111",
            job_inputs=[" ", ""],
            job_market_context="unknown",
        )


def test_requeue_stale_processing_jobs_rejects_invalid_options() -> None:
    with pytest.raises(ToolInputError, match="older_than_seconds"):
        requeue_stale_processing_jobs(older_than_seconds=0)

    with pytest.raises(ToolInputError, match="limit"):
        requeue_stale_processing_jobs(limit=0)


def test_retry_delay_uses_exponential_backoff() -> None:
    assert _retry_delay_seconds(error_detail="boom", attempt=1) == 60
    assert _retry_delay_seconds(error_detail="boom", attempt=2) == 120


def test_retry_delay_respects_quota_hint_and_cap() -> None:
    assert (
        _retry_delay_seconds(error_detail="429 retryDelay': '300s'", attempt=1) == 300
    )
    assert (
        _retry_delay_seconds(error_detail="429 retryDelay': '9999s'", attempt=1)
        == MAX_RETRY_DELAY_SECONDS
    )
