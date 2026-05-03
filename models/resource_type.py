"""Shared learning-resource type enum."""

from __future__ import annotations

from enum import StrEnum


class ResourceType(StrEnum):
    """Supported learning-resource categories."""

    COURSE = "course"
    PROJECT = "project"
    CERT = "cert"
    DOC = "doc"
