"""Build and validate final career-match outputs from run state."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

from models.confidence import ConfidenceLevel
from models.match import (
    AgentRunState,
    LearningPlanItem,
    LearningResource,
    MatchDimensionScores,
    MatchOutput,
)
from models.resource_type import ResourceType
from modules.error.common import ToolInputError
from modules.matches.state import (
    FINAL_MATCH_OUTPUT_KEY,
    JOB_ID_KEY,
    LAST_PRIORITIZED_SKILL_GAPS_KEY,
    LAST_SCORE_KEY,
    RESOURCES_BY_SKILL_KEY,
    int_field,
    require_mapping,
    state_snapshot,
    string_list,
)
from modules.utils.trace import build_agent_trace


def build_final_match_output(
    *,
    context: object,
    state: MutableMapping[str, object],
    job_id: str | None,
    reasoning: str | None,
    max_learning_plan_items: int,
) -> dict[str, object]:
    """Validate state and return the final serialized match output."""
    score = require_mapping(state, LAST_SCORE_KEY)
    output_job_id = _resolve_job_id(state, job_id)
    gap_skills = string_list(score.get("gap_skills"))
    matched_skills = string_list(score.get("matched_skills"))
    confidence = ConfidenceLevel(score.get("confidence", ConfidenceLevel.UNKNOWN))
    _validate_final_confidence(confidence)
    _validate_run_state(state)
    _require_prioritized_gaps_if_needed(state=state, gap_skills=gap_skills)
    _require_low_confidence_follow_up(
        state=state,
        confidence=confidence,
        gap_skills=gap_skills,
    )
    learning_plan = _build_learning_plan(
        state=state,
        max_items=max_learning_plan_items,
    )
    overall_score = int_field(score, "overall_score")
    final_reasoning = _ensure_confidence_reasoning(
        reasoning
        or _build_reasoning(
            overall_score=overall_score,
            confidence=confidence,
            matched_count=len(matched_skills),
            gap_count=len(gap_skills),
        ),
        overall_score=overall_score,
        confidence=confidence,
    )

    output = MatchOutput(
        job_id=output_job_id,
        overall_score=overall_score,
        confidence=confidence,
        dimension_scores=MatchDimensionScores.model_validate(
            require_mapping(
                cast("MutableMapping[str, object]", score),
                "dimension_scores",
            )
        ),
        matched_skills=matched_skills,
        gap_skills=gap_skills,
        reasoning=final_reasoning,
        learning_plan=learning_plan,
        agent_trace=build_agent_trace(context),
    )
    payload = output.model_dump(mode="json")
    state[FINAL_MATCH_OUTPUT_KEY] = payload
    return payload


def _resolve_job_id(
    state: MutableMapping[str, object],
    explicit_job_id: str | None,
) -> str:
    raw_job_id = explicit_job_id or state.get(JOB_ID_KEY)
    resolved_job_id = str(raw_job_id).strip() if raw_job_id is not None else ""
    if not resolved_job_id:
        raise ToolInputError("job_id is required to finalize match output.")
    state[JOB_ID_KEY] = resolved_job_id
    return resolved_job_id


def _validate_run_state(state: MutableMapping[str, object]) -> None:
    """Validate the typed ADK run-state payload before final output creation."""
    try:
        AgentRunState.model_validate(state_snapshot(state))
    except ValueError as error:
        raise ToolInputError("ADK run state is invalid.") from error


def _validate_final_confidence(confidence: ConfidenceLevel) -> None:
    if confidence is ConfidenceLevel.UNKNOWN:
        raise ToolInputError("confidence must be one of: low, medium, high.")


def _require_prioritized_gaps_if_needed(
    *,
    state: MutableMapping[str, object],
    gap_skills: list[str],
) -> None:
    """Require prioritization before producing a learning plan for gaps."""
    if not gap_skills:
        return
    _prioritized_items(state)


def _require_low_confidence_follow_up(
    *,
    state: MutableMapping[str, object],
    confidence: ConfidenceLevel,
    gap_skills: list[str],
) -> None:
    """Ensure low-confidence scores are not finalized without more signal."""
    if confidence is not ConfidenceLevel.LOW or not gap_skills:
        return

    top_gap = _top_prioritized_skill(state)
    if top_gap is None:
        raise ToolInputError(
            "Low-confidence finalization requires prioritized skill gaps."
        )
    if not _resources_for_skill(state, top_gap):
        raise ToolInputError(
            "Low-confidence finalization requires resources for the highest-priority gap."
        )


def _prioritized_items(state: MutableMapping[str, object]) -> list[dict[str, object]]:
    raw_prioritized = require_mapping(state, LAST_PRIORITIZED_SKILL_GAPS_KEY)
    raw_items = raw_prioritized.get("prioritized_skills")
    if not isinstance(raw_items, list) or not raw_items:
        raise ToolInputError(
            "`context.state['last_prioritized_skill_gaps'].prioritized_skills` "
            "must include at least one item when skill gaps exist."
        )

    items: list[dict[str, object]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ToolInputError("Prioritized skill gap items must be JSON objects.")
        items.append(cast("dict[str, object]", raw_item))
    return items


def _top_prioritized_skill(state: MutableMapping[str, object]) -> str | None:
    prioritized = sorted(
        _prioritized_items(state),
        key=lambda item: _coerce_int(item.get("priority_rank"), default=999),
    )
    for item in prioritized:
        raw_skill = item.get("skill")
        if isinstance(raw_skill, str) and raw_skill.strip():
            return raw_skill.strip()
    return None


def _build_learning_plan(
    *,
    state: MutableMapping[str, object],
    max_items: int,
) -> list[LearningPlanItem]:
    raw_prioritized = state.get(LAST_PRIORITIZED_SKILL_GAPS_KEY)
    prioritized_payload = (
        require_mapping(state, LAST_PRIORITIZED_SKILL_GAPS_KEY)
        if raw_prioritized is not None
        else {}
    )
    raw_items = prioritized_payload.get("prioritized_skills")
    prioritized_items = raw_items if isinstance(raw_items, list) else []

    learning_plan: list[LearningPlanItem] = []
    for raw_item in prioritized_items[: max(0, max_items)]:
        if not isinstance(raw_item, dict):
            continue
        item = _coerce_plan_item(cast("dict[str, object]", raw_item), state)
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
    resources_by_skill = state.get(RESOURCES_BY_SKILL_KEY)
    raw_research = None
    if isinstance(resources_by_skill, dict):
        typed_resources_by_skill = cast("dict[str, object]", resources_by_skill)
        raw_research = typed_resources_by_skill.get(skill.casefold())
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
        parsed_type = ResourceType(str(resource_type))
    except ValueError:
        return None
    return LearningResource(
        title=title.strip(),
        url=url.strip(),
        estimated_hours=_coerce_int(raw_resource.get("estimated_hours"), default=0),
        type=parsed_type,
    )


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


def _ensure_confidence_reasoning(
    reasoning: str,
    *,
    overall_score: int,
    confidence: ConfidenceLevel,
) -> str:
    if confidence is not ConfidenceLevel.LOW:
        return reasoning
    if "low confidence" in reasoning.casefold():
        return reasoning
    return (
        f"{reasoning} This is a low confidence result because the available "
        f"evidence only supports a {overall_score}/100 calibrated match score."
    )


def _coerce_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    return default
