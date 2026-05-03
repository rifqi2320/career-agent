"""Internal ADK agent for learning-resource research."""

from __future__ import annotations

import asyncio
import json
import re
from uuid import uuid4

from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from modules.builder.llm.builder import build_llm
from modules.config.llm import LlmProfile, get_llm_config
from modules.config.resources import load_resource_research_config
from modules.error.common import (
    DependencyError,
    RetryableModelOutputError,
    ToolTimeoutError,
)
from modules.resources.schemas import ResearchSkillResourcesOutputSchema
from modules.resources.tools import (
    query_github_learning_resources,
    query_github_repository_readme,
    query_skill_resource_db,
)

RESOURCE_RESEARCH_AGENT_NAME = "resource_research_agent"
RESOURCE_RESEARCH_APP_NAME = "career_resource_research"
RESOURCE_RESEARCH_OUTPUT_KEY = "resource_research_output"

RESOURCE_RESEARCH_AGENT_INSTRUCTION = """
You research learning resources for a career skill gap.

Before answering, call the retrieval tools:
- query_skill_resource_db for curated resources from the project database.
- query_github_learning_resources for GitHub repositories, examples, tutorials, and project references.
- query_github_repository_readme for GitHub repositories that look promising or ambiguous.

GitHub search workflow:
- Inspect README content for GitHub repositories before selecting them when the title and description are not enough to judge quality.
- If a GitHub repository is low quality, stale, unrelated, too generic, mostly empty, or lacks useful learning material, ignore it.
- If GitHub search returns weak results, call query_github_learning_resources again with a more specific related query such as the target skill plus "lab", "training", "detection", "examples", "tutorial", or a relevant tool name.
- Do not include a GitHub resource just to satisfy the minimum count if curated database resources are stronger.

Return only a JSON object that matches the configured response schema.

Selection rules:
- Choose at least 3 and at most 5 resources.
- Only choose resources returned by the tools.
- Do not alter titles, URLs, or resource types from the tool data.
- Preserve estimated_hours when the tool returned a value greater than 0.
- When estimated_hours is 0, estimate a realistic study time in hours from the resource title, description, type, and scope.
- Rank strongest resources first.
- Use the curated database as the primary quality signal.
- If the database returns relevant resources, include at least one curated database resource unless all database results are clearly irrelevant.
- Use GitHub resources when they add practical or portfolio value beyond the curated resources.
- Consider target skill relevance, seniority context, credibility, practical value, and time cost.
- Use unique titles and URLs.
- If the combined tool results contain fewer than 3 usable resources, do not invent resources.
- Set relevance_score from 0 to 100 for the selected set.
""".strip()


async def run_resource_research_agent(
    *,
    skill_name: str,
    seniority_context: str,
) -> ResearchSkillResourcesOutputSchema:
    """Run the internal ADK resource researcher and return validated output."""
    research_config = load_resource_research_config()
    model_result = build_llm(get_llm_config(profile=LlmProfile.MAIN))
    if model_result.is_err():
        raise DependencyError(
            f"Failed to build resource research model: {model_result.error}"
        ) from model_result.error
    model = model_result.value
    if model is None:
        raise DependencyError("Failed to build resource research model: empty result")

    agent = Agent(
        name=RESOURCE_RESEARCH_AGENT_NAME,
        description=(
            "Researches and ranks learning resources from curated DB and GitHub."
        ),
        model=model,
        instruction=RESOURCE_RESEARCH_AGENT_INSTRUCTION,
        tools=[
            query_skill_resource_db,
            query_github_learning_resources,
            query_github_repository_readme,
        ],
        output_schema=ResearchSkillResourcesOutputSchema,
        output_key=RESOURCE_RESEARCH_OUTPUT_KEY,
    )
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(
        app_name=RESOURCE_RESEARCH_APP_NAME,
        user_id="research_skill_resources",
        session_id=str(uuid4()),
    )
    runner = Runner(
        app_name=RESOURCE_RESEARCH_APP_NAME,
        agent=agent,
        session_service=session_service,
    )

    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=(
                    f"Target skill: {skill_name}\n"
                    f"Seniority context: {seniority_context}\n"
                    "Research and return the best resources."
                )
            )
        ],
    )

    try:
        async with asyncio.timeout(research_config.agent_timeout_seconds):
            final_text = await _run_agent_to_final_text(
                runner=runner,
                session_id=session.id,
                message=message,
            )
    except TimeoutError as error:
        raise ToolTimeoutError(
            "Resource research agent timed out.",
            original_error=error,
        ) from error

    return _parse_resource_research_output(final_text)


async def _run_agent_to_final_text(
    *,
    runner: Runner,
    session_id: str,
    message: types.Content,
) -> str:
    """Run an ADK runner invocation and return the final response text."""
    final_text = ""
    async for event in runner.run_async(
        user_id="research_skill_resources",
        session_id=session_id,
        new_message=message,
    ):
        if not event.is_final_response():
            continue
        if event.author != RESOURCE_RESEARCH_AGENT_NAME:
            continue
        if event.content is None or not event.content.parts:
            continue
        final_text = "".join(
            part.text for part in event.content.parts if part.text and not part.thought
        ).strip()

    if not final_text:
        raise RetryableModelOutputError("Resource research agent returned no output.")
    return final_text


def _parse_resource_research_output(
    response_text: str,
) -> ResearchSkillResourcesOutputSchema:
    """Parse and validate the internal agent's final JSON output."""
    text = response_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise RetryableModelOutputError(
            "Resource research agent output failed JSON parsing."
        ) from error

    try:
        return ResearchSkillResourcesOutputSchema.model_validate(payload)
    except ValueError as error:
        raise RetryableModelOutputError(
            "Resource research agent output failed schema validation."
        ) from error
