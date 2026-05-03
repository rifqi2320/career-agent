"""Root ADK agent definition for career intelligence."""

from google.adk.agents.llm_agent import Agent

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
)
