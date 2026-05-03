from __future__ import annotations

from agents.career_intelligence.builder import (
    CAREER_INTELLIGENCE_AGENT_NAME,
    CAREER_INTELLIGENCE_TOOLS,
    build_career_intelligence_agent,
)
from modules.config.llm import DEFAULT_CONFIG_PATH, load_project_llm_config
from modules.utils.callback import (
    handle_after_tool_callback,
    handle_before_model_callback,
    handle_before_tool_callback,
    handle_tool_error_callback,
)


def test_default_config_loads_from_configs_directory() -> None:
    config = load_project_llm_config(DEFAULT_CONFIG_PATH)

    assert DEFAULT_CONFIG_PATH.name == "default.json"
    assert DEFAULT_CONFIG_PATH.parent.name == "configs"
    assert config.main.model_name.value


def test_agent_builder_registers_required_tools() -> None:
    config = load_project_llm_config(DEFAULT_CONFIG_PATH)

    agent = build_career_intelligence_agent(config)

    assert agent.name == CAREER_INTELLIGENCE_AGENT_NAME
    assert agent.tools == CAREER_INTELLIGENCE_TOOLS
    assert agent.before_tool_callback == handle_before_tool_callback
    assert agent.after_tool_callback == handle_after_tool_callback
    assert agent.on_tool_error_callback == handle_tool_error_callback
    assert agent.before_model_callback == handle_before_model_callback
    assert [getattr(tool, "__name__") for tool in agent.tools] == [
        "extract_jd_requirements",
        "score_candidate_against_requirements",
        "prioritise_skill_gaps",
        "research_skill_resources",
        "finalize_match_output",
    ]
