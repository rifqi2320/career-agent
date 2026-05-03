from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from google.adk.tools import ToolContext

from models.match import AgentStreamEvent
from models.confidence import ConfidenceMetrics, ConfidenceLevel, calibrate_confidence
from modules.error.common import (
    RetryableModelOutputError,
    ToolExecutionError,
    ToolInputError,
    ToolTimeoutError,
)
from modules.logging import logging
from modules.resources.repository import list_skill_resources
from modules.resources.research_agent import run_resource_research_agent
from modules.resources.schemas import (
    CandidateResourceSchema,
    ResearchSkillResourcesOutputSchema,
    ResourceType,
    SkillResourceItemSchema,
)
from modules.utils.adk_events import append_runtime_event
from modules.utils.trace import (
    increment_fallbacks,
    increment_llm_calls,
    store_tool_result_by_key,
)

__all__ = [
    "CandidateResourceSchema",
    "ResearchSkillResourcesOutputSchema",
    "ResourceType",
    "SkillResourceItemSchema",
    "get_curated_skill_resources",
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

    resources = await _research_skill_resources_agent_with_timeout_policy(
        skill_name=normalized_skill,
        seniority_context=seniority_context,
        context=context,
    )

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


def get_curated_skill_resources(
    skill_name: str,
    seniority_context: str = "unknown",
    *,
    context: ToolContext,
) -> ResearchSkillResourcesOutputSchema:
    """Return DB-curated learning resources as a deterministic fallback."""
    normalized_skill = skill_name.strip()
    if not normalized_skill:
        raise ToolInputError("skill_name must not be empty.")

    increment_fallbacks(context)
    result = list_skill_resources(
        skill_name=normalized_skill,
        seniority_context=seniority_context,
        limit=5,
    )
    if result.is_err():
        raise ToolExecutionError(
            f"Failed to fetch curated skill resources: {result.error}",
            original_error=result.error,
        )

    rows = result.value or []
    if len(rows) < 3:
        raise ToolInputError(
            "Curated fallback found fewer than 3 resources for this skill."
        )

    resources = [
        SkillResourceItemSchema(
            title=row.title,
            url=row.url,
            estimated_hours=row.estimated_hours,
            type=ResourceType(row.resource_type),
        )
        for row in rows[:5]
    ]
    output = ResearchSkillResourcesOutputSchema(
        resources=resources,
        relevance_score=65,
    )
    confidence = _calibrate_research_confidence(
        candidate_count=len(resources),
        selected_count=len(resources),
        relevance_score=output.relevance_score,
    )
    result_payload = output.model_copy(
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
    return result_payload


async def _research_skill_resources_agent_with_timeout_policy(
    *,
    skill_name: str,
    seniority_context: str,
    context: ToolContext,
) -> ResearchSkillResourcesOutputSchema:
    """Run resource research, retrying one timeout before curated fallback."""
    parent_job_id = _parent_job_id(context)

    def event_sink(event: AgentStreamEvent) -> None:
        append_runtime_event(context, event)

    last_timeout: ToolTimeoutError | None = None
    for attempt in range(1, 3):
        increment_llm_calls(context)
        try:
            return await _research_skill_resources_agent(
                skill_name=skill_name,
                seniority_context=seniority_context,
                parent_job_id=parent_job_id,
                event_sink=event_sink,
            )
        except ToolTimeoutError as error:
            last_timeout = error
            logging.warning(
                "research_skill_resources timeout | attempt=%d",
                attempt,
            )
        except RetryableModelOutputError:
            logging.warning("research_skill_resources retryable agent output failure")
            raise

    logging.warning(
        "research_skill_resources using curated fallback after repeated timeout"
    )
    try:
        return get_curated_skill_resources(
            skill_name=skill_name,
            seniority_context=seniority_context,
            context=context,
        )
    except ToolInputError:
        if last_timeout is not None:
            raise last_timeout
        raise


async def _research_skill_resources_agent(
    *,
    skill_name: str,
    seniority_context: str,
    parent_job_id: str | None = None,
    event_sink: Callable[[AgentStreamEvent], None] | None = None,
) -> ResearchSkillResourcesOutputSchema:
    """Run the internal resource research agent."""
    return await run_resource_research_agent(
        skill_name=skill_name,
        seniority_context=seniority_context,
        parent_job_id=parent_job_id,
        event_sink=event_sink,
    )


def _parent_job_id(context: ToolContext) -> str | None:
    raw_job_id = context.state.get("job_id")
    if raw_job_id is None:
        return None
    job_id = str(raw_job_id).strip()
    return job_id or None
