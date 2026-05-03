"""Root ADK agent definition for career intelligence."""

from .builder import build_career_intelligence_agent
from .config import settings

root_agent = build_career_intelligence_agent(settings)
