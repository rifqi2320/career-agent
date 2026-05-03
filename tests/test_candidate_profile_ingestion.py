from __future__ import annotations

import pytest

from modules.candidates.profile import profile_from_text
from modules.error.common import ToolInputError


def test_profile_from_text_extracts_structured_match_profile() -> None:
    profile = profile_from_text(
        """
        Senior AI Engineer
        7 years building Python, FastAPI, PostgreSQL, RabbitMQ, Docker, RAG,
        and LLM systems.

        Skills: Python, Google ADK, FastAPI, PostgreSQL
        """
    )

    assert profile.years_experience == 7
    assert profile.seniority_level == "senior"
    assert profile.domain == "ai"
    assert "python" in profile.skills
    assert "rabbitmq" in profile.skills
    assert "google adk" in profile.skills


def test_profile_from_text_rejects_empty_input() -> None:
    with pytest.raises(ToolInputError, match="must not be empty"):
        profile_from_text("   ")
