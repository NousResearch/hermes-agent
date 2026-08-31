"""CLI-safe mode switching through the config_set tool."""

from __future__ import annotations

import json
from types import SimpleNamespace

from hermes_cli.mode_prompts import PLAN_PROMPT
from tools.config_set_tool import config_set_tool


def test_config_set_updates_the_calling_agent_without_a_gateway_session(monkeypatch):
    agent = SimpleNamespace(agent_mode="plan", ephemeral_system_prompt=PLAN_PROMPT)
    monkeypatch.setattr(
        "tui_gateway.server.handle_request",
        lambda request: {"result": {"key": "mode", "value": request["params"]["value"]}},
    )

    result = json.loads(
        config_set_tool(
            key="mode",
            value="auto",
            session_id="cli-session-not-in-gateway",
            agent=agent,
        )
    )

    assert result["result"]["value"] == "auto"
    assert agent.agent_mode == "auto"
    assert agent.ephemeral_system_prompt is None


def test_invalid_mode_does_not_mutate_the_calling_agent(monkeypatch):
    agent = SimpleNamespace(agent_mode="plan", ephemeral_system_prompt=PLAN_PROMPT)
    called = False

    def gateway(_request):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr("tui_gateway.server.handle_request", gateway)
    result = json.loads(config_set_tool(key="mode", value="bogus", agent=agent))

    assert "error" in result
    assert called is False
    assert agent.agent_mode == "plan"
    assert agent.ephemeral_system_prompt == PLAN_PROMPT
