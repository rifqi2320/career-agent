from __future__ import annotations

from modules.config.resources import load_resource_research_config


def test_load_resource_research_config_accepts_github_pat_token(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_PAT_TOKEN", "pat-token")

    config = load_resource_research_config()

    assert config.github_token == "pat-token"
