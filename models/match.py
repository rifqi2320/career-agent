"""Schemas for Part A career-match outputs and runtime trace."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from models.confidence import ConfidenceLevel
from models.resource_type import ResourceType


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
    type: ResourceType


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


class AgentStreamEvent(BaseModel):
    """Sanitized event emitted while an ADK runner invocation progresses."""

    event_type: Literal[
        "run_started",
        "model_response",
        "tool_call",
        "tool_response",
        "state_delta",
        "final_response",
        "run_completed",
        "run_failed",
    ]
    job_id: str = Field(min_length=1)
    sequence: int = Field(default=0, ge=0)
    author: str | None = None
    tool: str | None = None
    status: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class MatchOutput(BaseModel):
    """Final validated Part A output shape."""

    job_id: str = Field(min_length=1)
    overall_score: int = Field(ge=0, le=100)
    confidence: ConfidenceLevel
    dimension_scores: MatchDimensionScores
    matched_skills: list[str] = Field(default_factory=list)
    gap_skills: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)
    learning_plan: list[LearningPlanItem] = Field(default_factory=list)
    agent_trace: AgentTrace

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: ConfidenceLevel) -> ConfidenceLevel:
        """Reject unknown confidence values in the final output schema."""
        if value is ConfidenceLevel.UNKNOWN:
            raise ValueError("confidence must be one of: low, medium, high.")
        return value


class AgentRunState(BaseModel):
    """Typed snapshot of state carried across Part A tool calls."""

    job_id: str = Field(min_length=1)
    last_requirements: dict[str, object] | None = None
    last_score: dict[str, object] | None = None
    last_prioritized_skill_gaps: dict[str, object] | None = None
    resources_by_skill: dict[str, dict[str, object]] = Field(default_factory=dict)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    total_llm_calls: int = Field(default=0, ge=0)
    fallbacks_triggered: int = Field(default=0, ge=0)
