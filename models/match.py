"""Schemas for Part A career-match outputs and runtime trace."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from models.confidence import ConfidenceLevel


class LearningResourceType(StrEnum):
    """Allowed learning resource categories."""

    COURSE = "course"
    PROJECT = "project"
    CERT = "cert"
    DOC = "doc"


class MatchDimensionScores(BaseModel):
    """Dimension-level candidate/job fit scores."""

    skills: int = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=100)
    seniority_fit: int = Field(ge=0, le=100)


class LearningResource(BaseModel):
    """Learning resource included in a final learning plan."""

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    estimated_hours: int = Field(ge=0)
    type: LearningResourceType


class LearningPlanItem(BaseModel):
    """One prioritized skill-development action."""

    skill: str = Field(min_length=1)
    priority_rank: int = Field(ge=1)
    estimated_match_gain_pct: int = Field(ge=0, le=100)
    resources: list[LearningResource] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class ToolCallTrace(BaseModel):
    """Trace entry for one tool boundary crossing."""

    tool: str = Field(min_length=1)
    status: str = Field(pattern="^(success|error|skipped)$")
    latency_ms: int = Field(ge=0)
    error_type: str | None = None
    message: str | None = None


class AgentTrace(BaseModel):
    """Trace data populated by runtime orchestration, not by the LLM."""

    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    total_llm_calls: int = Field(default=0, ge=0)
    fallbacks_triggered: int = Field(default=0, ge=0)


class MatchOutput(BaseModel):
    """Final validated Part A output shape."""

    job_id: UUID
    overall_score: int = Field(ge=0, le=100)
    confidence: ConfidenceLevel
    dimension_scores: MatchDimensionScores
    matched_skills: list[str] = Field(default_factory=list)
    gap_skills: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)
    learning_plan: list[LearningPlanItem] = Field(default_factory=list)
    agent_trace: AgentTrace


class AgentRunState(BaseModel):
    """Typed snapshot of state carried across Part A tool calls."""

    job_id: UUID
    last_requirements: dict[str, object] | None = None
    last_score: dict[str, object] | None = None
    last_prioritized_skill_gaps: dict[str, object] | None = None
    resources_by_skill: dict[str, dict[str, object]] = Field(default_factory=dict)
    agent_trace: AgentTrace = Field(default_factory=AgentTrace)
