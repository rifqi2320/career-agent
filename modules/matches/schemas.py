"""Schemas for candidate/job match tool inputs and scoring outputs."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field

from models.confidence import ConfidenceLevel
from models.match import MatchDimensionScores

ScoreDimensionsSchema = MatchDimensionScores


class RequirementsInputSchema(BaseModel):
    """Minimal structured job requirements used by candidate scoring."""

    required_skills: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("required_skills", "skills"),
    )
    seniority_level: str = Field(
        default="unknown",
        validation_alias=AliasChoices("seniority_level", "seniority"),
    )
    domain: str = "unknown"
    responsibilities: list[str] = Field(default_factory=list)


class ScoreCandidateOutputSchema(BaseModel):
    """Structured score returned by the candidate scoring tool."""

    overall_score: int = Field(ge=0, le=100)
    dimension_scores: ScoreDimensionsSchema
    matched_skills: list[str] = Field(default_factory=list)
    gap_skills: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel
    confidence_score: int = Field(default=0, ge=0, le=100)
