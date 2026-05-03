"""Schemas for skill learning-resource research."""

from __future__ import annotations

from pydantic import BaseModel, Field

from models.confidence import ConfidenceLevel
from models.resource_type import ResourceType


class SkillResourceItemSchema(BaseModel):
    """Learning resource selected for a specific skill gap."""

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    estimated_hours: int = Field(ge=0)
    type: ResourceType


class ResearchSkillResourcesOutputSchema(BaseModel):
    """Ranked learning resources for one target skill."""

    resources: list[SkillResourceItemSchema] = Field(
        default_factory=list,
        min_length=3,
        max_length=5,
    )
    relevance_score: int = Field(ge=0, le=100)
    confidence_score: int = Field(default=0, ge=0, le=100)
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN


class CandidateResourceSchema(BaseModel):
    """Candidate learning resource available to the internal research agent."""

    title: str
    abstracts: str | None = None
    url: str
    estimated_hours: int = Field(ge=0)
    type: ResourceType
    skill_name: str
    seniority_context: str
    source: str | None = None


class GitHubReadmeSchema(BaseModel):
    """README content fetched for a GitHub repository candidate."""

    repository: str
    url: str
    readme_text: str
