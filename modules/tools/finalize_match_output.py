"""ADK tool wrapper for final career-match output creation."""

from __future__ import annotations

from google.adk.tools import ToolContext

from modules.error.common import ToolInputError
from modules.matches.finalization import build_final_match_output
from modules.matches.state import require_match_state


def finalize_match_output(
    *,
    context: ToolContext,
    job_id: str | None = None,
    reasoning: str | None = None,
    max_learning_plan_items: int = 3,
) -> dict[str, object]:
    """Validate and return the final career match output from prior tool results."""
    try:
        state = require_match_state(context)
    except ToolInputError as error:
        raise ToolInputError(
            "Tool context state is required to finalize match output."
        ) from error

    return build_final_match_output(
        context=context,
        state=state,
        job_id=job_id,
        reasoning=reasoning,
        max_learning_plan_items=max_learning_plan_items,
    )
