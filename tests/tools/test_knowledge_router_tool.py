"""Tests for the knowledge router tool wrapper and prompt guidance."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def test_knowledge_router_tools_register_with_registry():
    import tools.knowledge_router_tool  # noqa: F401
    from tools.registry import registry

    assert registry.get_entry("knowledge_route_decision") is not None
    assert registry.get_entry("knowledge_write") is not None
    assert set(registry.get_tool_names_for_toolset("knowledge")) >= {
        "knowledge_route_decision",
        "knowledge_write",
    }


def test_builtin_discovery_imports_knowledge_and_siyuan_tool_modules():
    from tools.registry import discover_builtin_tools, registry

    modules = discover_builtin_tools()

    assert "tools.knowledge_router_tool" in modules
    assert "tools.siyuan_tool" in modules
    assert registry.get_entry("siyuan_search") is not None


def test_knowledge_route_decision_handler_returns_json():
    from tools.knowledge_router_tool import handle_knowledge_route_decision

    payload = json.loads(
        handle_knowledge_route_decision(
            {"content_type": "runbook", "title": "Backup", "content": "# Backup"}
        )
    )

    assert payload["success"] is True
    assert payload["decision"]["destination"] == "static_knowledge"
    assert payload["decision"]["requires_title"] is True


def test_knowledge_write_dry_run_does_not_call_router_write(monkeypatch):
    from tools import knowledge_router_tool

    router = MagicMock()
    monkeypatch.setattr(knowledge_router_tool, "build_default_router", lambda: router)

    payload = json.loads(
        knowledge_router_tool.handle_knowledge_write(
            {
                "content_type": "task_log",
                "content": "temporary test output",
                "dry_run": True,
            }
        )
    )

    assert payload["success"] is True
    assert payload["written"] is False
    assert payload["decision"]["destination"] == "none"
    router.write.assert_not_called()


def test_knowledge_write_calls_default_router_for_real_write(monkeypatch):
    from knowledge.types import KnowledgeWriteResult
    from tools import knowledge_router_tool

    router = MagicMock()
    router.write.return_value = KnowledgeWriteResult(
        success=True,
        destination="dynamic_memory",
        action="retain",
        written=True,
        id="mem-1",
    )
    monkeypatch.setattr(knowledge_router_tool, "build_default_router", lambda: router)

    payload = json.loads(
        knowledge_router_tool.handle_knowledge_write(
            {"content_type": "lesson", "content": "Prefer official APIs."}
        )
    )

    assert payload["success"] is True
    assert payload["written"] is True
    assert payload["destination"] == "dynamic_memory"
    router.write.assert_called_once()


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _agent_with_tools(*tool_names: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


def test_knowledge_router_guidance_injected_only_when_tool_loaded():
    from agent.prompt_builder import KNOWLEDGE_ROUTER_GUIDANCE

    with_tools = _agent_with_tools("knowledge_write", "knowledge_route_decision")
    assert KNOWLEDGE_ROUTER_GUIDANCE in with_tools._build_system_prompt()

    without_tools = _agent_with_tools("web_search")
    assert KNOWLEDGE_ROUTER_GUIDANCE not in without_tools._build_system_prompt()
