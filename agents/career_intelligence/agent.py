"""Root ADK agent definition for career intelligence."""

from google.adk.agents.llm_agent import Agent

from modules.tools.extract_jd_requirements import extract_jd_requirements
from modules.tools.prioritise_skill_gaps import prioritise_skill_gaps
from modules.tools.research_skill_resources import research_skill_resources
from modules.tools.score_candidate_against_requirements import (
    score_candidate_against_requirements,
)

from .config import settings
from .prompts import ROOT_AGENT_INSTRUCTION

root_agent = Agent(
    model=settings.model,
    name="career_intelligence",
    description=(
        "career-intelligence: a focused assistant for career analysis, "
        "positioning, interview preparation, and job-search strategy."
    ),
    instruction=ROOT_AGENT_INSTRUCTION,
    tools=[
        extract_jd_requirements,
        score_candidate_against_requirements,
        prioritise_skill_gaps,
        research_skill_resources,
    ],
)
