"""Shared confidence calibration model and utilities."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ConfidenceLevel(StrEnum):
    """Confidence buckets with deterministic ordering."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> ConfidenceLevel:
        """Fallback for unknown or malformed values."""
        return cls.UNKNOWN


def _clamp(value: float) -> int:
    """Clamp a confidence-like value to `[0, 100]` integers."""
    return max(0, min(100, int(round(value))))


def map_score_to_level(score: int) -> ConfidenceLevel:
    """Map a numeric score into a confidence band."""
    if score >= 85:
        return ConfidenceLevel.HIGH
    if score >= 60:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def calibrate_confidence(
    score: int | float,
    *,
    evidence_ratio: float = 1.0,
    penalties: int = 0,
) -> "ConfidenceMetrics":
    """Apply a small deterministic calibration to a model score."""
    normalized = _clamp(score * evidence_ratio - penalties)
    return ConfidenceMetrics(
        confidence_score=normalized,
        confidence=map_score_to_level(normalized),
    )


class ConfidenceMetrics(BaseModel):
    """Reusable confidence output for tool responses."""

    confidence_score: int = Field(ge=0, le=100)
    confidence: ConfidenceLevel
