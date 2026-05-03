"""Candidate profile extraction helpers for the FastAPI entrypoint."""

from __future__ import annotations

from io import BytesIO
import re
from typing import Final

from pypdf import PdfReader

from modules.candidates.schemas import CandidateProfileInputSchema
from modules.error.common import ToolInputError

_COMMON_SKILLS: Final[tuple[str, ...]] = (
    "agent development kit",
    "airflow",
    "aws",
    "azure",
    "docker",
    "fastapi",
    "flask",
    "gemini",
    "github actions",
    "google adk",
    "kubernetes",
    "langchain",
    "langgraph",
    "llm",
    "machine learning",
    "postgresql",
    "prompt engineering",
    "python",
    "rabbitmq",
    "rag",
    "react",
    "redis",
    "sql",
    "terraform",
    "typescript",
    "vector database",
)
_SECTION_SPLIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\n,;|/]+")
_YEARS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<years>\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\b",
    re.IGNORECASE,
)


def profile_from_structured_payload(
    payload: object,
) -> CandidateProfileInputSchema:
    """Validate an already-structured profile payload."""
    try:
        return CandidateProfileInputSchema.model_validate(payload)
    except ValueError as error:
        raise ToolInputError("candidate profile payload is invalid.") from error


def profile_from_text(text: str) -> CandidateProfileInputSchema:
    """Extract the minimal structured profile the ADK matching tool needs."""
    normalized_text = text.strip()
    if not normalized_text:
        raise ToolInputError("candidate resume text must not be empty.")
    lower_text = normalized_text.casefold()
    return CandidateProfileInputSchema(
        skills=_extract_skills(normalized_text, lower_text),
        years_experience=_extract_years_experience(normalized_text),
        seniority_level=_extract_seniority(lower_text),
        domain=_extract_domain(lower_text),
    )


def text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from a PDF upload."""
    if not pdf_bytes:
        raise ToolInputError("candidate resume PDF must not be empty.")
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        page_text = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:
        raise ToolInputError("candidate resume PDF could not be parsed.") from error
    text = "\n".join(page_text).strip()
    if not text:
        raise ToolInputError("candidate resume PDF did not contain extractable text.")
    return text


def _extract_skills(text: str, lower_text: str) -> list[str]:
    explicit_skills = _extract_skills_from_section(text)
    discovered = {skill for skill in _COMMON_SKILLS if skill in lower_text}
    discovered.update(explicit_skills)
    return sorted(discovered)


def _extract_skills_from_section(text: str) -> set[str]:
    lines = text.splitlines()
    skills: set[str] = set()
    in_skills_section = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_skills_section:
                break
            continue
        lower_line = stripped.casefold()
        if lower_line.startswith(("skills", "technical skills", "core skills")):
            in_skills_section = True
            _, _, after_colon = stripped.partition(":")
            if after_colon:
                skills.update(_split_skill_items(after_colon))
            continue
        if in_skills_section and _looks_like_new_section(stripped):
            break
        if in_skills_section:
            skills.update(_split_skill_items(stripped))
    return skills


def _split_skill_items(text: str) -> set[str]:
    items: set[str] = set()
    for raw_item in _SECTION_SPLIT_PATTERN.split(text):
        item = raw_item.strip(" -\t*").casefold()
        if 1 < len(item) <= 64:
            items.add(item)
    return items


def _looks_like_new_section(line: str) -> bool:
    return line.endswith(":") and len(line.split()) <= 4


def _extract_years_experience(text: str) -> float | None:
    matches = [
        float(match.group("years"))
        for match in _YEARS_PATTERN.finditer(text)
        if match.group("years")
    ]
    if not matches:
        return None
    return max(matches)


def _extract_seniority(lower_text: str) -> str:
    if any(token in lower_text for token in ("principal", "staff", "lead engineer")):
        return "lead"
    if "senior" in lower_text:
        return "senior"
    if "junior" in lower_text:
        return "junior"
    return "unknown"


def _extract_domain(lower_text: str) -> str:
    if any(
        token in lower_text for token in ("llm", "rag", "agent", "machine learning")
    ):
        return "ai"
    if any(token in lower_text for token in ("security", "soc", "siem")):
        return "security"
    if any(token in lower_text for token in ("fintech", "banking", "payment")):
        return "fintech"
    return "unknown"
