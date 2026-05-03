from __future__ import annotations

from time import perf_counter

from google.adk.tools import ToolContext

from models.confidence import ConfidenceMetrics, ConfidenceLevel, calibrate_confidence
from modules.error.common import RetryableModelOutputError, ToolInputError
from modules.logging import logging
from modules.resources.research_agent import run_resource_research_agent
from modules.resources.schemas import (
    CandidateResourceSchema,
    ResearchSkillResourcesOutputSchema,
    ResourceType,
    SkillResourceItemSchema,
)
from modules.utils.trace import increment_llm_calls, store_tool_result_by_key

__all__ = [
    "CandidateResourceSchema",
    "ResearchSkillResourcesOutputSchema",
    "ResourceType",
    "SkillResourceItemSchema",
    "research_skill_resources",
]


def _calibrate_research_confidence(
    candidate_count: int,
    selected_count: int,
    relevance_score: int,
) -> ConfidenceMetrics:
    """Compute confidence from result quality and retrieval completeness."""
    if candidate_count == 0:
        return ConfidenceMetrics(confidence_score=0, confidence=ConfidenceLevel.LOW)

    retrieval_ratio = min(1.0, selected_count / min(5, candidate_count))
    adjusted_score = relevance_score * 0.8 + retrieval_ratio * 20
    return calibrate_confidence(adjusted_score)


async def research_skill_resources(
    skill_name: str,
    seniority_context: str = "unknown",
    *,
    context: ToolContext,
) -> ResearchSkillResourcesOutputSchema:
    """Research and rank resources using an internal DB/GitHub research agent."""
    started_at = perf_counter()
    logging.info(
        "research_skill_resources started | skill_name=%s seniority_context=%s",
        skill_name,
        seniority_context,
    )

    normalized_skill = skill_name.strip()
    if not normalized_skill:
        raise ToolInputError("skill_name must not be empty.")

    increment_llm_calls(context)
    try:
        resources = await _research_skill_resources_agent(
            skill_name=normalized_skill,
            seniority_context=seniority_context,
        )
    except RetryableModelOutputError:
        logging.warning("research_skill_resources retryable agent output failure")
        raise

    confidence = _calibrate_research_confidence(
        candidate_count=len(resources.resources),
        selected_count=len(resources.resources),
        relevance_score=resources.relevance_score,
    )
    result_payload = resources.model_copy(
        update={
            "confidence_score": confidence.confidence_score,
            "confidence": confidence.confidence,
        }
    )
    context.state["last_resources_research"] = result_payload.model_dump()
    store_tool_result_by_key(
        context,
        state_key="resources_by_skill",
        item_key=normalized_skill.casefold(),
        value=result_payload.model_dump(),
    )
    elapsed_ms = int((perf_counter() - started_at) * 1000)
    logging.info(
        "research_skill_resources success | selected_resources=%d relevance_score=%d confidence_score=%d elapsed_ms=%d",
        len(result_payload.resources),
        result_payload.relevance_score,
        result_payload.confidence_score,
        elapsed_ms,
    )
    return result_payload


async def _research_skill_resources_agent(
    *,
    skill_name: str,
    seniority_context: str,
) -> ResearchSkillResourcesOutputSchema:
    """Run the internal resource research agent."""
    return await run_resource_research_agent(
        skill_name=skill_name,
        seniority_context=seniority_context,
    )
