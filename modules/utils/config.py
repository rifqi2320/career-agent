"""Utility helpers for reading and validating project config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from safe_result import safe

ModelT = TypeVar("ModelT", bound=BaseModel)


@safe
def read_text_file(path: Path) -> str:
    """Read UTF-8 text from a file path."""
    return path.read_text(encoding="utf-8")


@safe
def parse_json_text(raw_text: str) -> object:
    """Parse JSON text into Python objects."""
    return json.loads(raw_text)


@safe
def validate_model(payload: object, model_type: type[ModelT]) -> ModelT:
    """Validate payload into a Pydantic model type."""
    return model_type.model_validate(payload)
