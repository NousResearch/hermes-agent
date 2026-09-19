"""Behavior contract for config-gated autonomous model selection."""

import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
from agent.tool_dispatch_helpers import _NEVER_PARALLEL_TOOLS
from toolsets import _HERMES_CORE_TOOLS
from tools.registry import registry
from tools import model_selection_tool as selection
from tools.delegate_tool_toolsets import DELEGATE_BLOCKED_TOOLS


class FakeAgent:
    def __init__(self):
        self.model = "gpt-5.6-terra"
        self.provider = "openai-codex"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.api_key = ""
        self.api_mode = "codex_responses"
        self.reasoning_config = {"enabled": True, "effort": "medium"}
        self._primary_runtime = {"reasoning_config": dict(self.reasoning_config)}
        self.switch_calls = []
        self.notices = []
        self.session_updates = []
        self.notice_callback = self.notices.append
        self.model_selection_callback = self.session_updates.append

    def switch_model(self, **kwargs):
        self.switch_calls.append(kwargs)
        self.model = kwargs["new_model"]
        self.provider = kwargs["new_provider"]
        self.base_url = kwargs["base_url"]
        self.api_mode = kwargs["api_mode"]


@pytest.fixture
def route_config():
    return {
        "enabled": True,
        "routes": {
            "luna": {
                "target": "luna",
                "default_reasoning": "low",
                "allowed_reasoning": ["low"],
                "description": "Bounded extraction and formatting.",
            },
            "terra": {
                "target": "terra",
                "default_reasoning": "medium",
                "allowed_reasoning": ["medium", "high"],
                "description": "Ordinary engineering.",
            },
            "sol": {
                "target": "sol",
                "default_reasoning": "high",
                "allowed_reasoning": ["high", "xhigh"],
                "description": "High-consequence or ambiguous HomeLab work.",
            },
        },
    }


def resolved(model="gpt-5.6-sol"):
    return SimpleNamespace(
        success=True,
        new_model=model,
        target_provider="openai-codex",
        api_key="",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        runtime_capabilities={"reasoning": True},
        error_message="",
    )


def test_tool_is_config_gated_core_inline_barrier():
    entry = registry.get_entry("select_model")
    assert entry is not None
    assert entry.check_fn is not None
    assert "select_model" in _HERMES_CORE_TOOLS
    assert "select_model" in INLINE_TOOL_EXECUTORS
    assert "select_model" in _NEVER_PARALLEL_TOOLS
    assert "select_model" in DELEGATE_BLOCKED_TOOLS


def test_success_switches_model_applies_reasoning_and_reports(monkeypatch, route_config):
    agent = FakeAgent()
    monkeypatch.setattr(selection, "_load_selection_config", lambda: route_config)
    monkeypatch.setattr(selection, "_resolve_switch_result", lambda agent, target: resolved())

    payload = json.loads(selection.select_model_tool(
        agent,
        route="sol",
        reason="Reviewing a MikroTik firewall boundary.",
        reasoning_effort="xhigh",
        reasoning_reason="The firewall review is unusually ambiguous and benefits from deeper analysis.",
    ))

    assert payload == {
        "success": True,
        "changed": True,
        "route": "sol",
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "reasoning_effort": "xhigh",
        "reasoning_reason": "The firewall review is unusually ambiguous and benefits from deeper analysis.",
        "reason": "Reviewing a MikroTik firewall boundary.",
        "message": (
            "Model selected: Sol (gpt-5.6-sol) — Reviewing a MikroTik firewall boundary. "
            "Reasoning: xhigh — The firewall review is unusually ambiguous and benefits from deeper analysis."
        ),
    }
    assert agent.switch_calls == [{
        "new_model": "gpt-5.6-sol",
        "new_provider": "openai-codex",
        "api_key": "",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
        "capabilities": {"reasoning": True},
    }]
    assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}
    assert agent._primary_runtime["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
    assert agent._model_selected_in_tool_batch is True
    assert agent.session_updates[0]["runtime"]["model"] == "gpt-5.6-sol"
    assert agent.session_updates[0]["message"] == payload["message"]
    assert agent.notices == []


def test_route_default_reasoning_is_used(monkeypatch, route_config):
    agent = FakeAgent()
    monkeypatch.setattr(selection, "_load_selection_config", lambda: route_config)
    monkeypatch.setattr(selection, "_resolve_switch_result", lambda agent, target: resolved("gpt-5.6-luna"))

    payload = json.loads(selection.select_model_tool(
        agent, route="luna", reason="Formatting a bounded inventory."
    ))

    assert payload["reasoning_effort"] == "low"
    assert payload["reasoning_reason"] == "Low is the configured default for the Luna route."
    assert agent.reasoning_config == {"enabled": True, "effort": "low"}


@pytest.mark.parametrize(
    ("route", "effort", "error"),
    [
        ("opus", None, "Route 'opus' is not allowlisted"),
        ("luna", "xhigh", "Reasoning effort 'xhigh' is not allowed for route 'luna'"),
    ],
)
def test_invalid_selection_is_fail_closed(monkeypatch, route_config, route, effort, error):
    agent = FakeAgent()
    monkeypatch.setattr(selection, "_load_selection_config", lambda: route_config)

    payload = json.loads(selection.select_model_tool(
        agent, route=route, reason="A real reason.", reasoning_effort=effort
    ))

    assert payload["success"] is False
    assert error in payload["error"]
    assert agent.switch_calls == []
    assert agent.model == "gpt-5.6-terra"
    assert agent.notices == []
    assert agent.session_updates == []


def test_empty_reason_is_rejected_before_resolution(monkeypatch, route_config):
    agent = FakeAgent()
    monkeypatch.setattr(selection, "_load_selection_config", lambda: route_config)

    payload = json.loads(selection.select_model_tool(agent, route="sol", reason="  "))

    assert payload == {"success": False, "error": "A non-empty model-selection reason is required."}
    assert agent.switch_calls == []


def test_surface_without_report_callback_fails_before_resolution(monkeypatch, route_config):
    agent = FakeAgent()
    agent.model_selection_callback = None
    monkeypatch.setattr(selection, "_load_selection_config", lambda: route_config)

    payload = json.loads(selection.select_model_tool(
        agent, route="sol", reason="A high-consequence boundary change."
    ))

    assert payload["success"] is False
    assert "does not support reported model selection" in payload["error"]
    assert agent.switch_calls == []


def test_same_runtime_changes_reasoning_without_resetting_model(monkeypatch, route_config):
    agent = FakeAgent()
    agent.model = "gpt-5.6-sol"
    monkeypatch.setattr(selection, "_load_selection_config", lambda: route_config)
    monkeypatch.setattr(selection, "_resolve_switch_result", lambda agent, target: resolved())

    payload = json.loads(selection.select_model_tool(
        agent, route="sol", reason="Security review requires the high-consequence route."
    ))

    assert payload["success"] is True
    assert payload["changed"] is True
    assert agent.switch_calls == []
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}


def test_real_alias_resolution_uses_profile_config(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  provider: openai\n"
        "  default: gpt-current\n"
        "  aliases:\n"
        "    test-sol: openai/gpt-selected\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-secret")
    token = set_hermes_home_override(tmp_path)
    try:
        agent = FakeAgent()
        agent.provider = "openai"
        agent.model = "gpt-current"
        agent.api_key = "test-key-not-a-secret"
        result = selection._resolve_switch_result(agent, "test-sol")
    finally:
        reset_hermes_home_override(token)

    assert result.success is True
    assert result.target_provider == "openai"
    assert result.new_model == "gpt-selected"
