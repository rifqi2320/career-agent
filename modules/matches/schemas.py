"""Schemas for candidate/job matching workflows."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field
from pydantic import model_validator

from models.confidence import ConfidenceLevel
from models.match import MatchDimensionScores
from modules.error.common import ValidationError

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


class ExtractJDRequirementOutputSchema(BaseModel):
    """Structured job requirements extracted from raw JD text or a URL."""

    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    seniority_level: str = "unknown"
    domain: str = "unknown"
    responsibilities: list[str] = Field(default_factory=list)
    confidence_score: int = Field(default=0, ge=0, le=100)
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN


class ScoreCandidateOutputSchema(BaseModel):
    """Structured score returned by the candidate scoring tool."""

    overall_score: int = Field(ge=0, le=100)
    dimension_scores: ScoreDimensionsSchema
    matched_skills: list[str] = Field(default_factory=list)
    gap_skills: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel
    confidence_score: int = Field(default=0, ge=0, le=100)


class PrioritizedSkillSchema(BaseModel):
    """One skill gap ranked by expected match gain."""

    skill: str = Field(min_length=1)
    priority_rank: int = Field(ge=1)
    estimated_match_gain_pct: int = Field(ge=0, le=100)
    rationale: str = Field(min_length=1)


class PrioritizeSkillGapsOutputSchema(BaseModel):
    """Structured output for prioritized skill gaps."""

    prioritized_skills: list[PrioritizedSkillSchema] = Field(default_factory=list)
    confidence_score: int = Field(default=0, ge=0, le=100)
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN

    @model_validator(mode="after")
    def validate_rank_integrity(self) -> "PrioritizeSkillGapsOutputSchema":
        """Ensure ranks and skills form one clean ordered set."""
        if not self.prioritized_skills:
            raise ValidationError("prioritized_skills must not be empty.")

        normalized_skills = {
            skill_item.skill.strip().casefold()
            for skill_item in self.prioritized_skills
        }
        if len(normalized_skills) != len(self.prioritized_skills):
            raise ValidationError("prioritized_skills contains duplicate skills.")

        ranks = [skill_item.priority_rank for skill_item in self.prioritized_skills]
        if len(set(ranks)) != len(ranks):
            raise ValidationError("priority_rank values must be unique.")

        expected_ranks = list(range(1, len(self.prioritized_skills) + 1))
        if sorted(ranks) != expected_ranks:
            raise ValidationError("priority_rank values must be consecutive from 1..N.")
        return self
