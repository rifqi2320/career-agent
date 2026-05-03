"""Build the final Part A match output from tool state."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast
from uuid import uuid4

from google.adk.tools import ToolContext

from models.confidence import ConfidenceLevel
from models.match import (
    LearningPlanItem,
    LearningResource,
    LearningResourceType,
    MatchDimensionScores,
    MatchOutput,
)
from modules.error.common import ToolInputError
from modules.utils.trace import build_agent_trace, get_state


def finalize_match_output(
    *,
    context: ToolContext,
    job_id: str | None = None,
    reasoning: str | None = None,
    max_learning_plan_items: int = 3,
) -> dict[str, object]:
    """Validate and return the final career match output from prior tool results."""
    state = get_state(context)
    if state is None:
        raise ToolInputError("Tool context state is required to finalize match output.")

    score = _require_mapping(state.get("last_score"), "last_score")
    output_job_id = _resolve_job_id(state, job_id)
    gap_skills = _string_list(score.get("gap_skills"))
    matched_skills = _string_list(score.get("matched_skills"))
    learning_plan = _build_learning_plan(
        state=state,
        gap_skills=gap_skills,
        max_items=max_learning_plan_items,
    )
    final_reasoning = reasoning or _build_reasoning(
        overall_score=_int_field(score, "overall_score"),
        confidence=ConfidenceLevel(score.get("confidence", ConfidenceLevel.UNKNOWN)),
        matched_count=len(matched_skills),
        gap_count=len(gap_skills),
    )

    output = MatchOutput(
        job_id=output_job_id,
        overall_score=_int_field(score, "overall_score"),
        confidence=ConfidenceLevel(score.get("confidence", ConfidenceLevel.UNKNOWN)),
        dimension_scores=MatchDimensionScores.model_validate(
            _require_mapping(score.get("dimension_scores"), "dimension_scores")
        ),
        matched_skills=matched_skills,
        gap_skills=gap_skills,
        reasoning=final_reasoning,
        learning_plan=learning_plan,
        agent_trace=build_agent_trace(context),
    )
    payload = output.model_dump(mode="json")
    state["final_match_output"] = payload
    return payload


def _resolve_job_id(
    state: MutableMapping[str, object],
    explicit_job_id: str | None,
) -> str:
    raw_job_id = explicit_job_id or state.get("job_id")
    resolved_job_id = str(raw_job_id).strip() if raw_job_id is not None else ""
    if not resolved_job_id:
        resolved_job_id = str(uuid4())
    state["job_id"] = resolved_job_id
    return resolved_job_id


def _build_learning_plan(
    *,
    state: MutableMapping[str, object],
    gap_skills: list[str],
    max_items: int,
) -> list[LearningPlanItem]:
    raw_prioritized = state.get("last_prioritized_skill_gaps")
    prioritized_payload = (
        _require_mapping(raw_prioritized, "last_prioritized_skill_gaps")
        if raw_prioritized is not None
        else {}
    )
    raw_items = prioritized_payload.get("prioritized_skills")
    prioritized_items = raw_items if isinstance(raw_items, list) else []
    plan_source = prioritized_items or _fallback_prioritized_items(gap_skills)

    learning_plan: list[LearningPlanItem] = []
    for raw_item in plan_source[: max(0, max_items)]:
        if not isinstance(raw_item, dict):
            continue
        item = _coerce_plan_item(raw_item, state)
        if item is not None:
            learning_plan.append(item)
    return learning_plan


def _coerce_plan_item(
    raw_item: dict[str, object],
    state: MutableMapping[str, object],
) -> LearningPlanItem | None:
    raw_skill = raw_item.get("skill")
    if not isinstance(raw_skill, str) or not raw_skill.strip():
        return None
    skill = raw_skill.strip()
    priority_rank = _coerce_int(raw_item.get("priority_rank"), default=1)
    estimated_gain = _coerce_int(raw_item.get("estimated_match_gain_pct"), default=0)
    raw_rationale = raw_item.get("rationale")
    rationale = (
        raw_rationale.strip()
        if isinstance(raw_rationale, str) and raw_rationale.strip()
        else f"Closing {skill} improves coverage of an identified role gap."
    )

    return LearningPlanItem(
        skill=skill,
        priority_rank=priority_rank,
        estimated_match_gain_pct=estimated_gain,
        resources=_resources_for_skill(state, skill),
        rationale=rationale,
    )


def _resources_for_skill(
    state: MutableMapping[str, object],
    skill: str,
) -> list[LearningResource]:
    resources_by_skill = state.get("resources_by_skill")
    raw_research = None
    if isinstance(resources_by_skill, dict):
        typed_resources_by_skill = cast("dict[str, object]", resources_by_skill)
        raw_research = typed_resources_by_skill.get(skill.casefold())
    if raw_research is None:
        raw_research = state.get("last_resources_research")
    if not isinstance(raw_research, dict):
        return []

    research_payload = cast("dict[str, object]", raw_research)
    raw_resources = research_payload.get("resources")
    if not isinstance(raw_resources, list):
        return []

    resources: list[LearningResource] = []
    for raw_resource in raw_resources:
        if not isinstance(raw_resource, dict):
            continue
        resource = _coerce_resource(cast("dict[str, object]", raw_resource))
        if resource is not None:
            resources.append(resource)
    return resources


def _coerce_resource(raw_resource: dict[str, object]) -> LearningResource | None:
    title = raw_resource.get("title")
    url = raw_resource.get("url")
    resource_type = raw_resource.get("type")
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        parsed_type = LearningResourceType(str(resource_type))
    except ValueError:
        return None
    return LearningResource(
        title=title.strip(),
        url=url.strip(),
        estimated_hours=_coerce_int(raw_resource.get("estimated_hours"), default=0),
        type=parsed_type,
    )


def _fallback_prioritized_items(gap_skills: list[str]) -> list[dict[str, object]]:
    return [
        {
            "skill": skill,
            "priority_rank": index,
            "estimated_match_gain_pct": 0,
            "rationale": f"Address {skill} because it remains an unclosed role gap.",
        }
        for index, skill in enumerate(gap_skills, start=1)
    ]


def _build_reasoning(
    *,
    overall_score: int,
    confidence: ConfidenceLevel,
    matched_count: int,
    gap_count: int,
) -> str:
    return (
        f"The candidate scored {overall_score}/100 with {confidence.value} confidence "
        f"based on {matched_count} matched skills and {gap_count} remaining gaps. "
        "The learning plan prioritizes the gaps expected to improve role fit most."
    )


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ToolInputError(f"`context.state['{field_name}']` is required.")
    return cast("dict[str, object]", value)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _int_field(payload: dict[str, object], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int):
        raise ToolInputError(f"`{field_name}` must be an integer.")
    return value


def _coerce_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    return default
