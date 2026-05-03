"""Builder for the career intelligence ADK agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset

from models.config.llm import LlmProfilesConfig
from modules.builder.llm.builder import build_llm
from modules.error.common import DependencyError
from modules.tools.extract_jd_requirements import extract_jd_requirements
from modules.tools.finalize_match_output import finalize_match_output
from modules.tools.prioritise_skill_gaps import prioritise_skill_gaps
from modules.tools.research_skill_resources import (
    get_curated_skill_resources,
    research_skill_resources,
)
from modules.tools.score_candidate_against_requirements import (
    score_candidate_against_requirements,
)
from modules.utils.callback import (
    handle_after_tool_callback,
    handle_before_model_callback,
    handle_before_tool_callback,
    handle_tool_error_callback,
)

from .prompts import ROOT_AGENT_INSTRUCTION

CAREER_INTELLIGENCE_AGENT_NAME = "career_intelligence"
CAREER_INTELLIGENCE_AGENT_DESCRIPTION = (
    "career-intelligence: a focused assistant for career analysis, "
    "positioning, interview preparation, and job-search strategy."
)
AgentTool = Callable[..., Any] | BaseTool | BaseToolset

CAREER_INTELLIGENCE_TOOLS: list[AgentTool] = [
    extract_jd_requirements,
    score_candidate_against_requirements,
    prioritise_skill_gaps,
    research_skill_resources,
    get_curated_skill_resources,
    finalize_match_output,
]


def build_career_intelligence_agent(config: LlmProfilesConfig) -> Agent:
    """Build the ADK root agent from validated project configuration."""
    model = _build_agent_model(config)
    return Agent(
        model=model,
        name=CAREER_INTELLIGENCE_AGENT_NAME,
        description=CAREER_INTELLIGENCE_AGENT_DESCRIPTION,
        instruction=ROOT_AGENT_INSTRUCTION,
        tools=CAREER_INTELLIGENCE_TOOLS,
        before_tool_callback=handle_before_tool_callback,
        after_tool_callback=handle_after_tool_callback,
        on_tool_error_callback=handle_tool_error_callback,
        before_model_callback=handle_before_model_callback,
    )


def _build_agent_model(config: LlmProfilesConfig) -> BaseLlm:
    """Build the main model used by the root agent."""
    result = build_llm(config.main)
    if result.is_err():
        raise DependencyError(f"Failed to build root agent model: {result.error}")
    model = result.value
    if model is None:
        raise DependencyError("Failed to build root agent model: empty result")
    return model
