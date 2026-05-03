"""Pydantic models for root-level LLM profile configuration."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from models.llm import LlmConfig


class LlmProfile(StrEnum):
    """Named LLM profiles for routing tool calls."""

    SMALL = "small"
    MAIN = "main"


class LlmProfilesConfig(BaseModel):
    """Project-level LLM profile config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    small: LlmConfig
    main: LlmConfig
