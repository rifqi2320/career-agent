from __future__ import annotations

from enum import StrEnum
from time import perf_counter

from google.adk.tools import ToolContext
from jinja2 import Template
from pydantic import BaseModel, Field
from safe_result import safe_async

from models.llm import LlmConfig
from modules.config.llm import LlmProfile, get_llm_config
from modules.error.common import RetryableModelOutputError
from modules.logging import logging
from modules.resources.repository import list_skill_resources
from modules.tools.wrapper import wrap_safe_tool
from modules.utils import generate_structured_output

RESOURCE_LIMIT = 30

RESEARCH_SKILL_RESOURCES_SYSTEM_PROMPT = """
You are a career learning-resource ranker.

Given a target skill, seniority context, and candidate resources from the database,
return strict JSON with this schema:
- resources: list of up to 5 items
  - title: string
  - url: string
  - estimated_hours: integer
  - type: one of course|project|cert|doc
- relevance_score: integer 0-100

Rules:
- Only select resources from the provided candidate list.
- Prefer resources that match the skill and seniority context.
- Keep resource ordering from most to least useful.
- Be conservative if candidate resources are weakly related.
""".strip()

RESEARCH_SKILL_RESOURCES_USER_PROMPT_TEMPLATE = Template(
    """
Target skill: {{ skill_name }}
Seniority context: {{ seniority_context }}

Candidate resources (JSON):
{{ candidate_resources_json }}
""".strip()
)


class ResourceType(StrEnum):
    COURSE = "course"
    PROJECT = "project"
    CERT = "cert"
    DOC = "doc"


class SkillResourceItemSchema(BaseModel):
    title: str
    url: str
    estimated_hours: int = Field(ge=0)
    type: ResourceType


class ResearchSkillResourcesOutputSchema(BaseModel):
    resources: list[SkillResourceItemSchema] = Field(default_factory=list, max_length=5)
    relevance_score: int = Field(ge=0, le=100)


class CandidateResourceSchema(BaseModel):
    title: str
    abstracts: str | None = None
    url: str
    estimated_hours: int = Field(ge=0)
    type: ResourceType
    skill_name: str
    seniority_context: str
    source: str | None = None


@safe_async
async def _research_skill_resources(
    skill_name: str,
    seniority_context: str = "unknown",
    *,
    context: ToolContext,
) -> ResearchSkillResourcesOutputSchema:
    """Research and rank learning resources for a skill using DB candidates + LLM."""
    started_at = perf_counter()
    logging.info(
        "research_skill_resources started | skill_name=%s seniority_context=%s",
        skill_name,
        seniority_context,
    )

    normalized_skill = skill_name.strip()
    if not normalized_skill:
        raise ValueError("skill_name must not be empty.")

    candidate_result = list_skill_resources(
        skill_name=normalized_skill,
        seniority_context=seniority_context,
        limit=RESOURCE_LIMIT,
    )
    if candidate_result.is_err():
        raise RuntimeError(
            f"Failed to fetch skill resources from database: {candidate_result.error}"
        )

    candidate_rows = candidate_result.value
    if candidate_rows is None or len(candidate_rows) == 0:
        raise ValueError(
            "No skill resources found in database for this query. Seed `skill_resources` first."
        )

    candidate_resources = [
        CandidateResourceSchema(
            title=row.title,
            abstracts=row.abstracts,
            url=row.url,
            estimated_hours=row.estimated_hours,
            type=ResourceType(row.resource_type),
            skill_name=row.skill_name,
            seniority_context=row.seniority_context or "unknown",
            source=row.source,
        )
        for row in candidate_rows
    ]

    llm_config: LlmConfig = get_llm_config(profile=LlmProfile.MAIN)
    result = await _research_skill_resources_llm(
        skill_name=normalized_skill,
        seniority_context=seniority_context,
        candidate_resources=candidate_resources,
        llm_config=llm_config,
    )

    if result.is_ok() and result.value is not None:
        context.state["last_resources_research"] = result.value.model_dump()
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        logging.info(
            "research_skill_resources success | selected_resources=%d relevance_score=%d elapsed_ms=%d",
            len(result.value.resources),
            result.value.relevance_score,
            elapsed_ms,
        )
        return result.value

    if isinstance(result.error, RetryableModelOutputError):
        logging.warning(
            "research_skill_resources retryable model output failure | error=%s",
            result.error,
        )
        raise result.error

    raise RuntimeError(f"LLM resource research failed: {result.error}")


@safe_async
async def _research_skill_resources_llm(
    *,
    skill_name: str,
    seniority_context: str,
    candidate_resources: list[CandidateResourceSchema],
    llm_config: LlmConfig,
) -> ResearchSkillResourcesOutputSchema:
    """Use LLM to select and rank the best resources from DB candidates."""
    user_prompt = RESEARCH_SKILL_RESOURCES_USER_PROMPT_TEMPLATE.render(
        skill_name=skill_name,
        seniority_context=seniority_context,
        candidate_resources_json=[
            item.model_dump(mode="json") for item in candidate_resources
        ],
    )

    result = await generate_structured_output(
        llm_config=llm_config,
        system_prompt=RESEARCH_SKILL_RESOURCES_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=ResearchSkillResourcesOutputSchema,
    )
    return result.unwrap()


research_skill_resources = wrap_safe_tool(_research_skill_resources)
