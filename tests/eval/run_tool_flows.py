from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, cast

from google.adk.tools import ToolContext
from pydantic import BaseModel

from models.llm import LlmConfig
from modules.config.llm import LlmProfile, get_llm_config
from modules.logging import logging
from modules.tools.extract_jd_requirements import extract_jd_requirements
from modules.tools.prioritise_skill_gaps import prioritise_skill_gaps
from modules.tools.research_skill_resources import research_skill_resources
from modules.tools.score_candidate_against_requirements import (
    CandidateProfileInputSchema,
    score_candidate_against_requirements,
)
from modules.utils import generate_structured_output

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "tests" / "data"
CANDIDATE_DIR = DATA_DIR / "candidate_profile"
JOBREQ_DIR = DATA_DIR / "jobreq"
RESULTS_DIR = DATA_DIR / "results"

RUNS_PER_CANDIDATE = 5
JOB_MARKET_CONTEXT = "indonesia fintech ai engineer market"

PROFILE_PARSE_SYSTEM_PROMPT = """
You extract a candidate profile into strict JSON.

Return this schema only:
- skills: list[str]
- years_experience: number or null
- seniority_level: string
- domain: string

Rules:
- Extract skills explicitly evidenced in the profile text.
- Normalize skills to lowercase snake-like short terms where possible.
- Deduplicate and sort skills alphabetically for deterministic output.
- years_experience should be a conservative estimate from explicit date ranges; use null if unclear.
- If uncertain, use "unknown" for seniority_level and domain.
- Output JSON only.
""".strip()

PROFILE_PARSE_USER_PROMPT_TEMPLATE = """
Candidate profile text:
{profile_text}
""".strip()


class SerializableToolResult(BaseModel):
    status: str
    error_type: str | None = None
    error: str | None = None
    output: object | None = None


@dataclass
class DummyToolContext:
    state: dict[str, object] = field(default_factory=dict)


def _load_text_with_indirection(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").strip()
    candidate_pointer = (PROJECT_ROOT / raw).resolve()
    if raw.endswith(".md") and candidate_pointer.exists():
        return candidate_pointer.read_text(encoding="utf-8")
    return raw


def _serialize_value(value: Any) -> SerializableToolResult:
    output = value.model_dump() if hasattr(value, "model_dump") else value
    return SerializableToolResult(status="ok", output=output)


def _serialize_error(error: Exception) -> SerializableToolResult:
    return SerializableToolResult(
        status="err",
        error_type=type(error).__name__,
        error=str(error),
        output=None,
    )


async def _parse_candidate_profile(
    profile_text: str,
    llm_config: LlmConfig,
) -> CandidateProfileInputSchema:
    result = await generate_structured_output(
        llm_config=llm_config,
        system_prompt=PROFILE_PARSE_SYSTEM_PROMPT,
        user_prompt=PROFILE_PARSE_USER_PROMPT_TEMPLATE.format(profile_text=profile_text),
        schema=CandidateProfileInputSchema,
    )
    if result.is_err():
        raise RuntimeError(f"Failed to parse candidate profile: {result.error}")
    parsed = result.value
    if parsed is None:
        raise RuntimeError("Failed to parse candidate profile: empty result.")
    return parsed


def _first_research_skill(
    score_payload: dict[str, object] | None,
    prioritize_payload: dict[str, object] | None,
) -> str:
    if prioritize_payload is not None:
        prioritized = prioritize_payload.get("prioritized_skills")
        if isinstance(prioritized, list) and prioritized:
            first_item = prioritized[0]
            if isinstance(first_item, dict):
                prioritized_skill = cast("dict[str, object]", first_item)
                first_skill = prioritized_skill.get("skill")
                if isinstance(first_skill, str) and first_skill.strip():
                    return first_skill.strip()

    if score_payload is not None:
        gap_skills = score_payload.get("gap_skills")
        if isinstance(gap_skills, list) and gap_skills:
            first_gap = gap_skills[0]
            if isinstance(first_gap, str) and first_gap.strip():
                return first_gap.strip()

    return "python"


def _seniority_from_requirements(requirements_payload: dict[str, object] | None) -> str:
    if requirements_payload is None:
        return "unknown"
    seniority = requirements_payload.get("seniority_level")
    if isinstance(seniority, str) and seniority.strip():
        return seniority.strip()
    return "unknown"


def _canonical_tool_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


async def run() -> Path:
    started = datetime.now(tz=UTC)
    batch_id = started.strftime("%Y%m%d-%H%M%S")
    batch_dir = RESULTS_DIR / batch_id
    by_tool_dir = batch_dir / "by_tool"
    by_flow_dir = batch_dir / "by_flow"
    batch_dir.mkdir(parents=True, exist_ok=True)

    jobreq_files = sorted(JOBREQ_DIR.glob("*.md"))
    if not jobreq_files:
        raise RuntimeError(f"No jobreq files found in {JOBREQ_DIR}")
    jobreq_text = _load_text_with_indirection(jobreq_files[0])

    candidate_files = sorted(CANDIDATE_DIR.glob("*.md"))
    if not candidate_files:
        raise RuntimeError(f"No candidate profiles found in {CANDIDATE_DIR}")

    llm_config_main = get_llm_config(profile=LlmProfile.MAIN)
    parsed_profiles: dict[str, CandidateProfileInputSchema] = {}
    for candidate_file in candidate_files:
        profile_text = _load_text_with_indirection(candidate_file)
        parsed_profiles[candidate_file.stem] = await _parse_candidate_profile(
            profile_text=profile_text,
            llm_config=llm_config_main,
        )

    per_tool_variants: dict[str, dict[str, set[str]]] = {
        "extract_jd_requirements": {},
        "score_candidate_against_requirements": {},
        "prioritise_skill_gaps": {},
        "research_skill_resources": {},
    }

    for candidate_file in candidate_files:
        candidate_key = candidate_file.stem
        profile = parsed_profiles[candidate_key]
        for run_index in range(1, RUNS_PER_CANDIDATE + 1):
            logging.info(
                "eval flow run | candidate=%s run=%d/%d",
                candidate_key,
                run_index,
                RUNS_PER_CANDIDATE,
            )
            context = cast("ToolContext", DummyToolContext())

            try:
                extract_result = await extract_jd_requirements(
                    url_or_text=jobreq_text,
                    context=context,
                )
                extract_payload = _serialize_value(extract_result).model_dump(
                    mode="json"
                )
            except Exception as error:  # noqa: BLE001
                extract_payload = _serialize_error(error).model_dump(mode="json")

            try:
                score_result = await score_candidate_against_requirements(
                    candidate_profile=profile,
                    requirements=None,
                    context=context,
                )
                score_payload = _serialize_value(score_result).model_dump(mode="json")
            except Exception as error:  # noqa: BLE001
                score_payload = _serialize_error(error).model_dump(mode="json")

            try:
                prioritize_result = await prioritise_skill_gaps(
                    gap_skills=None,
                    job_market_context=JOB_MARKET_CONTEXT,
                    context=context,
                )
                prioritize_payload = _serialize_value(prioritize_result).model_dump(
                    mode="json"
                )
            except Exception as error:  # noqa: BLE001
                prioritize_payload = _serialize_error(error).model_dump(mode="json")

            extracted_requirements = (
                extract_payload.get("output")
                if isinstance(extract_payload.get("output"), dict)
                else None
            )
            scored = (
                score_payload.get("output")
                if isinstance(score_payload.get("output"), dict)
                else None
            )
            prioritized = (
                prioritize_payload.get("output")
                if isinstance(prioritize_payload.get("output"), dict)
                else None
            )

            research_skill = _first_research_skill(scored, prioritized)
            seniority_context = _seniority_from_requirements(extracted_requirements)
            try:
                research_result = await research_skill_resources(
                    skill_name=research_skill,
                    seniority_context=seniority_context,
                    context=context,
                )
                research_payload = _serialize_value(research_result).model_dump(
                    mode="json"
                )
            except Exception as error:  # noqa: BLE001
                research_payload = _serialize_error(error).model_dump(mode="json")

            run_key = f"{candidate_key}__run_{run_index:02d}"
            tool_payloads: dict[str, dict[str, Any]] = {
                "extract_jd_requirements": extract_payload,
                "score_candidate_against_requirements": score_payload,
                "prioritise_skill_gaps": prioritize_payload,
                "research_skill_resources": research_payload,
            }
            for tool_name, payload in tool_payloads.items():
                tool_path = by_tool_dir / tool_name / f"{run_key}.json"
                _write_json(tool_path, payload)
                candidate_variants = per_tool_variants[tool_name].setdefault(
                    candidate_key, set()
                )
                candidate_variants.add(_canonical_tool_payload(payload))

            flow_payload = {
                "candidate": candidate_key,
                "run_index": run_index,
                "jobreq_file": jobreq_files[0].name,
                "parsed_profile": profile.model_dump(mode="json"),
                "research_skill_selected": research_skill,
                "outputs": tool_payloads,
            }
            _write_json(by_flow_dir / f"{run_key}.json", flow_payload)

    comparison: dict[str, Any] = {
        "batch_id": batch_id,
        "runs_per_candidate": RUNS_PER_CANDIDATE,
        "candidate_count": len(candidate_files),
        "jobreq_file": jobreq_files[0].name,
        "per_tool_candidate_unique_variants": {},
    }

    needs_stricter_prompt = False
    for tool_name, candidate_map in per_tool_variants.items():
        comparison["per_tool_candidate_unique_variants"][tool_name] = {}
        for candidate_key, variants in candidate_map.items():
            unique_count = len(variants)
            comparison["per_tool_candidate_unique_variants"][tool_name][candidate_key] = {
                "unique_runs": unique_count,
                "is_consistent": unique_count == 1,
            }
            if unique_count > 1:
                needs_stricter_prompt = True

    comparison["needs_stricter_prompt"] = needs_stricter_prompt
    _write_json(batch_dir / "comparison.json", comparison)

    summary_lines = [
        f"batch_id: {batch_id}",
        f"candidates: {len(candidate_files)}",
        f"runs_per_candidate: {RUNS_PER_CANDIDATE}",
        f"jobreq_file: {jobreq_files[0].name}",
        "",
        "unique variants by tool/candidate:",
    ]
    for tool_name, candidate_map in comparison[
        "per_tool_candidate_unique_variants"
    ].items():
        summary_lines.append(f"- {tool_name}")
        for candidate_key, details in candidate_map.items():
            summary_lines.append(
                f"  - {candidate_key}: {details['unique_runs']} unique run payload(s)"
            )
    summary_lines.append("")
    summary_lines.append(
        f"needs_stricter_prompt: {comparison['needs_stricter_prompt']}"
    )
    (batch_dir / "SUMMARY.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    latest_pointer = RESULTS_DIR / "LATEST"
    latest_pointer.write_text(f"{batch_id}\n", encoding="utf-8")
    return batch_dir


if __name__ == "__main__":
    output_dir = asyncio.run(run())
    print(f"Wrote results to: {output_dir}")
