from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import cast

os.environ.setdefault("CAREER_AGENT_DISABLE_FILE_LOGS", "1")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://career@localhost:15432/career_agent",
)

from google.adk.tools import ToolContext
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_ROOT = PROJECT_ROOT / "tests" / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class DummyToolContext:
    """Minimal tool-context test double with mutable state."""

    state: dict[str, object] = field(default_factory=dict)


@pytest.fixture
def tool_context() -> ToolContext:
    """Return a fresh tool context per test."""
    return cast("ToolContext", DummyToolContext())


@pytest.fixture
def sample_candidate_profile_text() -> str:
    """Load sample candidate profile text fixture."""
    return (
        TEST_DATA_ROOT / "candidate_profile" / "sample_candidate_profile_1.md"
    ).read_text(encoding="utf-8")


@pytest.fixture
def sample_jobreq_text() -> str:
    """Load sample job requirement text fixture."""
    return (TEST_DATA_ROOT / "jobreq" / "sample_jobreq.md").read_text(encoding="utf-8")
