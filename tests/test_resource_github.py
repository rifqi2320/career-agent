from __future__ import annotations

import base64
import json
from typing import Any

from modules.config.resources import ResourceResearchConfig
from modules.resources.github import (
    fetch_github_repository_readme,
    search_github_learning_resources,
)
from modules.resources.schemas import ResourceType


class FakeHttpResponse:
    """Small context-manager response for urllib tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_search_github_learning_resources_maps_repository_candidates(
    monkeypatch,
) -> None:
    payload = {
        "items": [
            {
                "full_name": "example/python-projects",
                "html_url": "https://github.com/example/python-projects",
                "description": "Practice Python through projects.",
                "stargazers_count": 500,
                "language": "Python",
                "updated_at": "2026-05-01T00:00:00Z",
            }
        ]
    }
    captured_urls: list[str] = []

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        captured_urls.append(request.full_url)
        assert timeout == 3.0
        assert request.headers["User-agent"] == "career-agent-resource-research"
        return FakeHttpResponse(payload)

    monkeypatch.setattr("modules.resources.github.urlopen", fake_urlopen)

    result = search_github_learning_resources(
        skill_name="python",
        seniority_context="junior",
        limit=1,
        config=ResourceResearchConfig(
            github_api_base="https://api.github.test",
            github_timeout_seconds=3.0,
        ),
    )

    assert result.is_ok()
    candidates = result.value
    assert candidates is not None
    assert len(candidates) == 1
    assert candidates[0].title == "example/python-projects"
    assert candidates[0].type == ResourceType.PROJECT
    assert candidates[0].estimated_hours == 0
    assert candidates[0].source == "github_api"
    assert captured_urls
    assert captured_urls[0].startswith("https://api.github.test/search/repositories?")


def test_fetch_github_repository_readme_decodes_bounded_text(monkeypatch) -> None:
    readme_text = "# Python Projects\n\nPractice Python with focused labs."
    payload = {
        "html_url": "https://github.com/example/python-projects/blob/main/README.md",
        "encoding": "base64",
        "content": base64.b64encode(readme_text.encode("utf-8")).decode("utf-8"),
    }
    captured_urls: list[str] = []

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        captured_urls.append(request.full_url)
        assert timeout == 3.0
        return FakeHttpResponse(payload)

    monkeypatch.setattr("modules.resources.github.urlopen", fake_urlopen)

    result = fetch_github_repository_readme(
        repository="example/python-projects",
        config=ResourceResearchConfig(
            github_api_base="https://api.github.test",
            github_timeout_seconds=3.0,
        ),
    )

    assert result.is_ok()
    readme = result.value
    assert readme is not None
    assert readme.repository == "example/python-projects"
    assert readme.readme_text == readme_text
    assert captured_urls == [
        "https://api.github.test/repos/example/python-projects/readme"
    ]
