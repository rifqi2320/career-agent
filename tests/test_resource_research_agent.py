from __future__ import annotations

import pytest

from modules.error.common import RetryableModelOutputError
from modules.resources.research_agent import _parse_resource_research_output
from modules.resources.schemas import ResourceType


def test_parse_resource_research_output_accepts_minimum_three_resources() -> None:
    result = _parse_resource_research_output(
        """
        {
          "resources": [
            {
              "title": "Python Docs",
              "url": "https://docs.python.org/3/",
              "estimated_hours": 8,
              "type": "doc"
            },
            {
              "title": "Python Project",
              "url": "https://github.com/example/python-project",
              "estimated_hours": 0,
              "type": "project"
            },
            {
              "title": "Python Course",
              "url": "https://example.com/python-course",
              "estimated_hours": 20,
              "type": "course"
            }
          ],
          "relevance_score": 91
        }
        """
    )

    assert len(result.resources) == 3
    assert result.resources[0].type == ResourceType.DOC
    assert result.relevance_score == 91


def test_parse_resource_research_output_rejects_fewer_than_three_resources() -> None:
    with pytest.raises(RetryableModelOutputError, match="schema validation"):
        _parse_resource_research_output(
            """
            {
              "resources": [
                {
                  "title": "Python Docs",
                  "url": "https://docs.python.org/3/",
                  "estimated_hours": 8,
                  "type": "doc"
                }
              ],
              "relevance_score": 80
            }
            """
        )
