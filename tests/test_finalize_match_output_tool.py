from __future__ import annotations

from uuid import UUID

from google.adk.tools import ToolContext
import pytest

from models.confidence import ConfidenceLevel
from modules.error.common import ToolInputError
from modules.tools.finalize_match_output import finalize_match_output


def test_finalize_match_output_builds_valid_a3_payload(
    tool_context: ToolContext,
) -> None:
    tool_context.state.update(
        {
            "job_id": "11111111-1111-4111-8111-111111111111",
            "last_score": {
                "overall_score": 82,
                "confidence": "medium",
                "dimension_scores": {
                    "skills": 80,
                    "experience": 85,
                    "seniority_fit": 78,
                },
                "matched_skills": ["python", "apis"],
                "gap_skills": ["rag"],
            },
            "last_prioritized_skill_gaps": {
                "prioritized_skills": [
                    {
                        "skill": "rag",
                        "priority_rank": 1,
                        "estimated_match_gain_pct": 12,
                        "rationale": "RAG is the highest leverage missing skill.",
                    }
                ]
            },
            "resources_by_skill": {
                "rag": {
                    "resources": [
                        {
                            "title": "Build a RAG App",
                            "url": "https://example.com/rag",
                            "estimated_hours": 10,
                            "type": "project",
                        }
                    ],
                    "relevance_score": 90,
                }
            },
            "tool_calls": [
                {
                    "tool": "extract_jd_requirements",
                    "status": "success",
                    "latency_ms": 12,
                }
            ],
            "total_llm_calls": 3,
            "fallbacks_triggered": 0,
        }
    )

    result = finalize_match_output(context=tool_context)

    assert result.job_id == UUID("11111111-1111-4111-8111-111111111111")
    assert result.overall_score == 82
    assert result.confidence is ConfidenceLevel.MEDIUM
    assert result.dimension_scores.skills == 80
    assert result.learning_plan[0].skill == "rag"
    assert result.learning_plan[0].resources[0].type == "project"
    assert result.agent_trace.total_llm_calls == 3
    assert result.agent_trace.tool_calls[0].tool == "extract_jd_requirements"


def test_finalize_match_output_requires_score_state(tool_context: ToolContext) -> None:
    with pytest.raises(ToolInputError, match="last_score"):
        finalize_match_output(context=tool_context)
