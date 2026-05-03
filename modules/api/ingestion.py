"""FastAPI request parsing for candidate profile ingestion."""

from __future__ import annotations

import json

from fastapi import HTTPException, Request, status
from pydantic import ValidationError as PydanticValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from modules.api.schemas import CandidateCreateRequest
from modules.candidates.profile import (
    profile_from_structured_payload,
    profile_from_text,
    text_from_pdf_bytes,
)
from modules.candidates.schemas import CandidateProfileInputSchema
from modules.error.common import ToolInputError


async def profile_from_request(
    request: Request,
) -> tuple[CandidateProfileInputSchema, str]:
    """Parse one candidate profile from JSON, text, or multipart upload."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        return await _profile_from_multipart(request)

    try:
        payload = CandidateCreateRequest.model_validate(await request.json())
    except (PydanticValidationError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Expected JSON body with `profile` or `resume_text`.",
        ) from error

    if payload.profile is not None:
        return payload.profile, "structured_json"
    if payload.resume_text is not None:
        return _profile_from_text_or_400(payload.resume_text), "text"
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Either `profile` or `resume_text` is required.",
    )


async def _profile_from_multipart(
    request: Request,
) -> tuple[CandidateProfileInputSchema, str]:
    form = await request.form()
    raw_profile = form.get("profile")
    if isinstance(raw_profile, str) and raw_profile.strip():
        try:
            profile_payload = json.loads(raw_profile)
            return profile_from_structured_payload(profile_payload), "structured_json"
        except (json.JSONDecodeError, ToolInputError) as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error

    raw_text = form.get("resume_text")
    if isinstance(raw_text, str) and raw_text.strip():
        return _profile_from_text_or_400(raw_text), "text"

    upload = form.get("resume_file")
    if isinstance(upload, StarletteUploadFile):
        pdf_bytes = await upload.read()
        text = _pdf_text_or_400(pdf_bytes)
        return _profile_from_text_or_400(text), "pdf"

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Multipart body must include profile, resume_text, or resume_file.",
    )


def _profile_from_text_or_400(text: str) -> CandidateProfileInputSchema:
    try:
        return profile_from_text(text)
    except ToolInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error


def _pdf_text_or_400(pdf_bytes: bytes) -> str:
    try:
        return text_from_pdf_bytes(pdf_bytes)
    except ToolInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
